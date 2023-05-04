import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def read_result(path):
	with open(path, "r") as result_file:
		lines = result_file.read().splitlines()

	ids = []
	CIs = []
	CMs = []

	for line in lines[:-1]:
		img_id, CI, CM = process_line(line)
		
		
		if img_id == None:
			print("dropping nan")
			continue
		
		ids.append(img_id)
		CIs.append(CI)
		CMs.append(CM)

	return np.array(ids), np.array(CIs), np.array(CMs)

def process_line(line):

	parts = line.split(":")

	img_id = parts[0].split("_")[-2]
  
  
	try:

		CI = parts[1][1:-4].strip()
		CM = parts[2][:-1].strip()
		if CI == "nan" or CM == "nan":
			return None, None, None
		else:
			CI = float(CI)
			CM = float(CM)
	except:
		print(img_id)
		sys.exit()


	return img_id, CI, CM


ids, CIs, CMs = read_result("./brushifyforest_result.txt")


print(np.mean(CMs))


#CIs = CIs - CIs.min()
#CIs = CIs / CIs.max()

#CMs = CMs - CMs.min()
#CMs = CMs / CMs.max()
#print(CIs.shape)
#print(CIs)
#print("aaaaaaa")
#kmeans = KMeans(n_clusters=3, random_state=0).fit(CIs.reshape(-1))

#print(kmeans.cluster_centers_)

#plt.plot(CIs)
#plt.plot(CMs)
#plt.plot(np.abs(CIs - CMs))
#plt.scatter(kmeans.cluster_centers_)

#plt.show()
