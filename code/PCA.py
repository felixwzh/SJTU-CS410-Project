from sklearn.decomposition import PCA
import numpy as np

n_components=2000
n_components = 0.999
n_components = 0.99
n_components = 0.95
n_components = 0.90
n_components = 0.85


pca = PCA(n_components=n_components)

X = np.load("./../data/all_raw_data.npy")

pca.fit(X)

newX=pca.transform(X)

np.save("./../data/all_PCA_"+str(pca.n_components_)+".npy",newX)

print pca.explained_variance_ratio_
print pca.explained_variance_
print sum(pca.explained_variance_ratio_)

