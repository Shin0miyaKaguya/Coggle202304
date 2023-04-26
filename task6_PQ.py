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
import nanopq


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


# args
path_to_data = '.\\data'  # gallery images
BATCH_SIZE = len(glob.glob('.\data\dataset\*.jpg'))
threshold = 0.8  # pick out if the cosine similarity > threshold.

# build dataloader
preprocess = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.Grayscale(),
	transforms.ToTensor()
])
source_data = FilterableImageFolder(root=path_to_data, transform=preprocess,valid_classes=['dataset'])
source_names = source_data.samples
data_loader = torch.utils.data.DataLoader(source_data, batch_size=len(source_names))
query_data = FilterableImageFolder(root=path_to_data, transform=preprocess,valid_classes=['query'])
query_names = query_data.samples
query_loader = torch.utils.data.DataLoader(query_data, batch_size=1)

pq = nanopq.PQ(M=4, Ks=256, verbose=True)
for x,y in data_loader:
	data = x.view(x.size()[0],-1).detach().numpy()
	pq.fit(vecs=data, iter=20, seed=123)
	X_code = pq.encode(vecs=data)

sources = []
queries = [x[0].split(os.path.sep)[-1] for x in query_names]
for query,_ in query_loader:
	q = query.view(query.size()[0],-1).squeeze().detach().numpy()
	dists = pq.dtable(query=q).adist(codes=X_code)
	idx = np.argmin(dists)
	sources.append(source_names[idx][0].split(os.path.sep)[-1])

submit = pd.DataFrame({"source":sources,"query":queries})
submit.to_csv("submit/submit_task6_PQ.csv".format(), index=False)


"""PCA"""

