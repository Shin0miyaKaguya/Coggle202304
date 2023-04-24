import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from typing import *
from torchvision.datasets.folder import *
from sklearn.preprocessing import normalize
import cv2
import torch.nn.functional as F
from torch.utils.data import Dataset
import glob
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

if torch.cuda.is_available():
	device = torch.device('cuda:0')
else:
	device = torch.device('cpu')


class FilterableImageFolder(ImageFolder):
	def __init__(
			self,
			root: str,
			transform: Optional[Callable] = None,
			target_transform: Optional[Callable] = None,
			loader: Callable[[str], Any] = default_loader,
			is_valid_file: Optional[Callable[[str], bool]] = None,
			valid_classes: List = None
	):
		self.valid_classes = valid_classes
		super(FilterableImageFolder, self).__init__(root, transform, target_transform, loader, is_valid_file)

	def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
		classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
		classes = [valid_class for valid_class in classes if valid_class in self.valid_classes]
		if not classes:
			raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

		class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
		return classes, class_to_idx


class SiameseNetwork(nn.Module):
	def __init__(self):
		super(SiameseNetwork, self).__init__()
		self.cnn1 = nn.Sequential(
			nn.ReflectionPad2d(1),
			nn.Conv2d(1, 4, kernel_size=3),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(4),
			# nn.BatchNorm2d(4)中参数4为通道数
			nn.ReflectionPad2d(1),
			nn.Conv2d(4, 8, kernel_size=3),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(8),

			nn.ReflectionPad2d(1),
			nn.Conv2d(8, 8, kernel_size=3),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(8),

		)

		self.fc1 = nn.Sequential(
			nn.Linear(8 * 224 * 224, 500),
			nn.ReLU(inplace=True),

			nn.Linear(500, 500),
			nn.ReLU(inplace=True),

			nn.Linear(500, 5))

	def forward_once(self, x):
		output = self.cnn1(x)
		output = output.view(output.size(0), -1)
		output = self.fc1(output)
		return output

	def forward(self, input1, input2):
		output1 = self.forward_once(input1)
		output2 = self.forward_once(input2)
		return output1, output2


class ContrastiveLoss(torch.nn.Module):
	"""
	Contrastive loss function.
	Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	"""

	def __init__(self, margin=2.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin

	def forward(self, output1, output2, label):
		euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
		loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
									  (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

		return loss_contrastive

path_to_data = '.\\data'  # gallery images
BATCH_SIZE = 128

# build dataloader
# preprocess = transforms.Compose([
# 	transforms.Resize(256),
# 	transforms.CenterCrop(224),
# 	transforms.ToTensor(),
# 	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# test_data = FilterableImageFolder(root=path_to_data, transform=preprocess,valid_classes=['dataset'])
# source_names = test_data.samples
# data_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)
# query_data = FilterableImageFolder(root=path_to_data, transform=preprocess,valid_classes=['query'])
# query_loader = torch.utils.data.DataLoader(query_data, batch_size=BATCH_SIZE)
# query_names = query_data.samples



"""孪生网络Dataset"""

transform = transforms.Compose([
		transforms.RandomVerticalFlip(p=0.5),
		])


class TrainDataset(Dataset):
	def __init__(self,root,transforms = None):
		super(TrainDataset, self).__init__()
		self.imgs_path = root
		self.transforms = transforms

	def __getitem__(self, index):
		path = self.imgs_path[index]
		x1 = cv2.imread(path, 0)
		is_diff = np.random.randint(0,1)
		x1 = Image.fromarray(x1)
		if is_diff == 0:
			x2 = transform(x1)
		else:
			index1 = np.random.randint(0,len(self.imgs_path))
			while index1 == index:
				index1 = np.random.randint(0,len(self.imgs_path))
			path1 = self.imgs_path[index1]
			x2 = cv2.imread(path1, 0)
			x2 = Image.fromarray(x2)
		if self.transforms:
			x1,x2 = self.transforms(x1),self.transforms(x2)
		return x1,x2,int(is_diff)

	def __len__(self):
		return len(self.imgs_path)


class TestDataset(Dataset):
	def __init__(self,root,transforms = None):
		super(TestDataset, self).__init__()
		self.root = root
		self.transforms = transforms
		self.samples = root

	def __getitem__(self, index):
		query_path = self.root[index]
		x1 = cv2.imread(query_path, 0)
		x1 = Image.fromarray(x1)
		if self.transforms:
			x1 = self.transforms(x1)
		return x1

	def __len__(self):
		return len(self.root)

train_set = TrainDataset(glob.glob(".\data\dataset\*.jpg"), transforms=transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

test_set = TestDataset(root = glob.glob(".\data\query\*.jpg"), transforms=transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor()]))
test_set1 = TestDataset(root = glob.glob(".\data\dataset\*.jpg"), transforms=transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_set,batch_size=2500,shuffle=False, pin_memory=True)
test_loader1 = torch.utils.data.DataLoader(test_set1,batch_size=512,shuffle=False, pin_memory=True)

source_names = test_set1.samples
query_names = test_set.samples


counter = []
loss_history = []


def show_plot(iteration,loss):
	#绘制损失变化图
	plt.plot(iteration,loss)
	plt.show(block=True)


"""traning"""
def main():
	net = SiameseNetwork().to(device)  # 定义模型且移至GPU
	criterion = ContrastiveLoss()  # 定义损失函数
	optimizer = optim.Adam(net.parameters(), lr=0.0005)  # 定义优化器
	iteration_number = 0
	print("\t".join(["Epoch", "TrainLoss", "TestLoss"]))
	for epoch in tqdm(range(10)):
		net.train()
		train_loss = np.float("inf")
		for i, data in enumerate(train_loader, 0):
			img0, img1, label = data
			# img0维度为torch.Size([32, 1, 100, 100])，32是batch，label为torch.Size([32, 1])
			img0, img1, label = img0.to(device), img1.to(device), label.to(device)  # 数据移至GPU
			optimizer.zero_grad()
			output1, output2 = net(img0, img1)
			loss_contrastive = criterion(output1, output2, label)
			loss_contrastive.backward()
			optimizer.step()
			if i % 10 == 0:
				iteration_number += 10
				counter.append(iteration_number)
				loss_history.append(loss_contrastive.item())
				print(loss_contrastive.item())
			train_loss = np.min(train_loss,loss_contrastive.item())
		print("Epoch number: {} , Current loss: {:.4f}\n".format(epoch, train_loss))
	show_plot(counter, loss_history)
	torch.save(net,"model\simase.model")
# main()

net = torch.load("model\simase.model")


pre = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor()])
from sklearn.metrics.pairwise import euclidean_distances
queries,sources = [],[]
with torch.no_grad():
	all_dis = np.empty(shape=2500)
	all_idx = np.empty(shape=2500)
	for i,data in enumerate(test_loader,0):
		x1 = data
		o1,_ = net(x1,x1)
		for j,data1 in enumerate(test_loader1,0):
			o2,_= net(data1,data1)
			sims = euclidean_distances(o1,o2)
			if j==0:
				all_sims = sims
			else:
				all_sims = np.concatenate([all_sims,sims],axis=1)

res = np.argmin(all_sims,axis=1)
sources = [source_names[i].split(os.path.sep)[-1] for i in res]
queries = [x.split(os.path.sep)[-1] for x in query_names]

submit = pd.DataFrame({"source":sources,"query":queries})
submit.to_csv("submit/submit_task4_siamese.csv".format(), index=False)




# submit = pd.DataFrame({"source":sources,"query":queries})
# submit.to_csv("submit/submit_task4.csv", index=False)