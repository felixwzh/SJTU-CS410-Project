import matplotlib.pyplot as plt  
from sklearn.decomposition import PCA  
from sklearn.datasets import load_iris  
import numpy as np
import pandas as pd
project_path='/Users/jerry/Desktop/Gene_chip/'

labelnum={
	'Sex-2':2,
	'MaterialType-2':2,
	'DiseaseState-16':16,
	'BioSourceType-7':7
}
def datainit():
	labelname=['Sex-2','MaterialType-2','DiseaseState-16','BioSourceType-7']
	data = np.load(project_path+'data/all_PCA_500.npy')
	print(data.shape)
	label_df = pd.read_csv(project_path+'data/all_label.csv')
	results={}
	for name in labelname:
		label=label_df[name]
		y_mask=np.zeros([label.shape[0],])
		y_mask=(label!=y_mask)
		label=label[y_mask].tolist()
		subdata=data[y_mask].tolist()
		results[name]=[subdata,label]
	return results

def plot(results):
	for key in results:
		
		x = np.array(results[key][0])  
		y = results[key][1]
		pca = PCA(n_components=2)  
		x = pca.fit_transform(x)
		x_min, x_max = np.min(x), np.max(x)
		reduced_x = ((x - x_min) / (x_max - x_min)).tolist()
		print(len(y))
		#plt.scatter(red_x, red_y, c='r', marker='')  
		#plt.scatter(blue_x, blue_y, c='b', marker='2')
		#fig = plt.figure()
   		ax = plt.subplot(111)
   		for i in range(len(reduced_x)):
			ax.text(reduced_x[i][0], reduced_x[i][1],str(y[i]),color=plt.cm.Set1(y[i]/10.),fontdict={'weight': 'bold', 'size': 5})
		
		plt.title(key+' Distribution')
		plt.ylabel('y')
		plt.xlabel('x')
		plt.show()

plot(datainit())

'''
data = load_iris()  
y = data.target  
X = data.data  
pca = PCA(n_components=2)  
reduced_X = pca.fit_transform(X)  

red_x, red_y = [], []  
blue_x, blue_y = [], []  
green_x, green_y = [], []  
for i in range(len(reduced_X)):  
    if y[i] == 0:  
        red_x.append(reduced_X[i][0])  
        red_y.append(reduced_X[i][1])  
    elif y[i] == 1:  
        blue_x.append(reduced_X[i][0])  
        blue_y.append(reduced_X[i][1])  
    else:  
        green_x.append(reduced_X[i][0])  
        green_y.append(reduced_X[i][1])  
  
plt.scatter(red_x, red_y, c='r', marker='x')  
plt.scatter(blue_x, blue_y, c='b', marker='D')  
plt.scatter(green_x, green_y, c='g', marker='.')  
plt.show()
'''  