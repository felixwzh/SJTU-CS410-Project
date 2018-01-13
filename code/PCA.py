from sklearn.decomposition import PCA
import numpy as np

n_components=3400
# n_components = 0.999
# n_components = 0.99
# n_components = 0.95
# n_components = 0.90
# n_components = 0.85

# n_components=[200,400,1200]
# for i in range(3):

pca = PCA(n_components=n_components)

X = np.load("./../data/all_raw_data.npy")

pca.fit(X)

newX=pca.transform(X)

np.save("./../data/all_PCA_"+str(pca.n_components_)+".npy",newX)

# print pca.explained_variance_ratio_
# print pca.explained_variance_
print pca.n_components_, sum(pca.explained_variance_ratio_)

