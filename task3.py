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
import glob
from sklearn.preprocessing import normalize
import clip


# load model to device
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


class ResNet(nn.Module):
	#Pretrained Resnet18
	def __init__(self):
		super(ResNet, self).__init__()
		model = torchvision.models.resnet18(pretrained=True)
		model.fc = torch.nn.Linear(512, 512)
		torch.nn.init.zeros_(model.fc.bias)
		for param in model.parameters():
			param.requires_grad = False
		model.eval()
		self.model = model

	def forward(self, img):
		out = self.model(img)
		return out


class MyClip(nn.Module):
	def __init__(self):
		super(MyClip, self).__init__()
		imgmodel, preprocess = clip.load("ViT-B/32", device=device)
		imgmodel.eval()
		self.model = imgmodel

	def forward(self, img):
		out = self.model.encode_image(img)
		return out


# args
path_to_data = '.\\data'  # gallery images
BATCH_SIZE = 256
threshold = 0.8  # pick out if the cosine similarity > threshold.

# build dataloader
preprocess = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_data = FilterableImageFolder(root=path_to_data, transform=preprocess,valid_classes=['dataset'])
source_names = test_data.samples
data_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)
query_data = FilterableImageFolder(root=path_to_data, transform=preprocess,valid_classes=['query'])
query_loader = torch.utils.data.DataLoader(query_data, batch_size=BATCH_SIZE)
query_names = query_data.samples


model_names = ["ResNet","VIT","Clip"]
model_name = model_names[2]
imgmodel = None
if model_name =="VIT":
	imgmodel = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights)
	imgmodel.heads = torch.nn.Identity()
	imgmodel = imgmodel.to(device)

if model_name == "Clip":
	imgmodel =MyClip().to(device)

if model_name == "ResNet":
	imgmodel = ResNet().to(device)



source_feat = []
query_feat = []
# test and query
with torch.no_grad():
	# load query image
	for (x, y) in tqdm(data_loader, desc="Evaluating", leave=False):
		x = x.to(device)
		feat = imgmodel(x).detach().numpy()
		source_feat.append(feat)
	source_feat = np.vstack(source_feat)
	print(source_feat.shape)

	for (x, y) in tqdm(query_loader, desc="Evaluating", leave=False):
		x = x.to(device)
		feat = imgmodel(x).detach().numpy()
		query_feat.append(feat)
	query_feat = np.vstack(query_feat)
	print(query_feat.shape)

source_feat = normalize(source_feat)
query_feat = normalize(query_feat)

sim = np.dot(query_feat,source_feat.T)
res = np.argmax(sim,axis=1)

sources = [source_names[i][0].split(os.path.sep)[-1] for i in res]
queries = [x[0].split(os.path.sep)[-1] for x in query_names]

submit = pd.DataFrame({"source":sources,"query":queries})
submit.to_csv("submit/submit_task3_{}.csv".format(model_name), index=False)