import cv2
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

dataset_feat = []
for path in glob.glob(".\dataset\*.jpg"):
	img = cv2.imread(path, 0)
	feat = cv2.calcHist(np.array([img]), [0], None, [256], [0, 256]).flatten()
	dataset_feat.append(feat)

dataset_feat = np.array(dataset_feat)
dataset_feat = normalize(dataset_feat)

query_feat = []
for path in glob.glob(".\query\*.jpg"):
	img = cv2.imread(path, 0)
	feat = cv2.calcHist(np.array([img]), [0], None, [256], [0, 256]).flatten()
	query_feat.append(feat)

query_feat = np.array(query_feat)
query_feat = normalize(query_feat)

dis = np.dot(query_feat, dataset_feat.T)
dataset_path = np.array(glob.glob('.\dataset\*.jpg'))

res = pd.DataFrame({
	'source': [x.split("\\")[-1] for x in dataset_path[dis.argmax(1)]],
	'query': [x.split('\\')[-1] for x in glob.glob('.\query\*.jpg')]
})
res.to_csv("submit.csv", index=False)
