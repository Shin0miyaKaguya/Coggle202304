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

sorted_index = dis.argsort(axis=1)[:,-10:]



sources = []
queries = []
sift = cv2.SIFT_create()
for idx,x in enumerate(glob.glob('.\query\*.jpg')):
	print(idx)
	query = x.split('\\')[-1]
	can_pics = dataset_path[sorted_index[idx]]
	all_mathces = []
	img1 = cv2.imread(x, 0)  # queryImage
	kp1, des1 = sift.detectAndCompute(img1, None)
	for pic in can_pics:
		img2 = cv2.imread(pic, 0)  # trainImage
		kp2, des2 = sift.detectAndCompute(img2, None)
		# FLANN parameters
		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
		search_params = dict(checks=50)  # or pass empty dictionary
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(des1, des2, k=2)
		ratio_thresh = 0.7
		good_matches = []
		for m, n in matches:
			if m.distance < ratio_thresh * n.distance:
				good_matches.append(m)
		all_mathces.append(len(good_matches))
	best_match = np.array(all_mathces).argmax()
	sources.append(can_pics[best_match].split('\\')[-1])
	queries.append(query)

res = pd.DataFrame({
	"source":sources,
	"query":queries
})

res.to_csv("submit_task2.csv", index=False)
