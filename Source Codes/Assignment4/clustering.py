import util as u
import dset as d
import numpy as np
import sys
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

path = ".../IMM-Frontal Face DB SMALL"
fid, k = 7, 12
images = d.read_image_files(path, 0.1)
imagePatches = []
for i in range(len(images)):
    imagePatches.append(u.image_windows(images[i], stride=(40, 40)))

mergedwindows = np.vstack(imagePatches)
kmeans = KMeans(n_clusters=20).fit(mergedwindows)
agg = AgglomerativeClustering(n_clusters=20).fit(mergedwindows)
meanshift = MeanShift(n_jobs=-1, max_iter=20).fit(mergedwindows)

clusterid_km = [kmeans.predict(i) for i in imagePatches]
histograms_km = []
for i in range(len(images)):
    his, _ = np.histogram(clusterid_km[i], density=False)
    histograms_km.append(his)

agg_labels = agg.labels_
histograms_agg = []
i, j = 0, 35
while i < 2400:
    his, _ = np.histogram(agg_labels[i:j], density=False)
    histograms_agg.append(his)
    i += 35
    j += 35

clusterid_ms = [meanshift.predict(i) for i in imagePatches]
histograms_ms = []
for i in range(len(images)):
     his, _ = np.histogram(clusterid_ms[i], density=False)
     histograms_ms.append ( his )

NN_km = NearestNeighbors().fit(histograms_km)
distances_km, indices_km = NN_km.kneighbors(histograms_km[fid].reshape(1, -1), k)
NN_agg = NearestNeighbors().fit(histograms_agg)
distances_agg, indices_agg = NN_agg.kneighbors(histograms_agg[fid].reshape(1, -1), k)
NN_ms = NearestNeighbors().fit(histograms_ms)
distances_ms, indices_ms = NN_ms.kneighbors(histograms_ms[fid].reshape(1, -1), k)

u.plot_image_windows(vecs=kmeans.cluster_centers_, title="kmeans", size=(20, 20))
plt.show()
u.plot_image_windows(vecs=meanshift.cluster_centers_, title="meanshift", size=(20, 20))
plt.show()

sys.stdout = open("dis&indx.txt", "w")
print("distances and indices for each model", sep='\n')
print("The number of clusters found by the algorithm (kmeans) :", len(np.unique(kmeans.labels_)), sep='\n')
print("distances (kmeans) :", distances_km, sep='\n')
print("indices (kmeans) :", indices_km, sep='\n')
print("The number of clusters found by the algorithm (AgglomerativeClustering: )", agg.n_clusters_, sep='\n')
print("distances (AgglomerativeClustering :)", distances_agg, sep='\n')
print("indices (AgglomerativeClustering :)", indices_agg, sep='\n')
print("The number of clusters found by the algorithm (meanshift: )", len(np.unique(meanshift.labels_)), sep='\n')
print("distances (meanshift) :", distances_ms, sep='\n')
print("indices (meanshift) :", indices_ms, sep='\n')
sys.stdout.close()

x = np.arange(0, 12, 1)
fig, ax = plt.subplots(1, 3, figsize=(21, 4))
fig.suptitle('kmeans', fontsize=21)
ax[0].hist(histograms_km[fid], bins=x, align='left', edgecolor='w', color = "g")
ax[0].set_xticks(x, minor=False)
ax[1].hist(histograms_km[indices_km[0][k-1]], bins=x, align='left', edgecolor='w', color = "skyblue")
ax[1].set_xticks(x, minor=False)
ax[2].hist(histograms_km[fid] - histograms_km[indices_km[0][k-1]], bins=x, align='left', edgecolor='w', color = "r")
ax[2].set_xticks(x, minor=False)
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(21, 4))
fig.suptitle('AgglomerativeClustering', fontsize=21)
ax[0].hist(histograms_agg[fid], bins=x, align='left', edgecolor='w', color = "g")
ax[0].set_xticks(x, minor=False)
ax[1].hist(histograms_agg[indices_agg[0][k-1]], bins=x, align='left', edgecolor='w', color = "skyblue")
ax[1].set_xticks(x, minor=False)
ax[2].hist(histograms_agg[fid] - histograms_agg[indices_agg[0][k-1]], bins=x, align='left', edgecolor='w', color = "r")
ax[2].set_xticks(x, minor=False)
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(21, 4))
fig.suptitle('meanshift', fontsize=21)
ax[0].hist(histograms_ms[fid], bins=x, align='left', edgecolor='w', color = "g")
ax[0].set_xticks(x, minor=False)
ax[1].hist(histograms_ms[indices_ms[0][k-1]], bins=x, align='left', edgecolor='w', color = "skyblue")
ax[1].set_xticks(x, minor=False)
ax[2].hist(histograms_ms[fid] - histograms_ms[indices_ms[0][k-1]], bins=x, align='left', edgecolor='w', color = "r")
ax[2].set_xticks(x, minor=False)
plt.show()