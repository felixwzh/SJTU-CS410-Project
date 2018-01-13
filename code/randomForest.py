
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn import random_projection

# load data

# raw_flag=True
raw_flag=False
dimension=500
# dimension=1000
# dimension=2000
# dimension=5000 #0.999
# dimension=22283

if raw_flag:
    X = np.load("./../data/all_raw_data.npy")
    X = X[:,0:dimension]


else:
    X = np.load("./../data/all_PCA_"+str(dimension)+".npy")


# transformer = random_projection.SparseRandomProjection()
# X = transformer.fit_transform(X)

label_df = pd.read_csv('./../data/all_label.csv')
y = label_df['MaterialType-2']
print X.shape

# split data
kf = KFold(n_splits=5,random_state=1)
kf.get_n_splits(X)
print(kf)
macro_f1=[]
micro_f1=[]

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = SVC(kernel='linear')
    clf.fit(X, y)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    macro_f1.append(f1_score(y_test, y_pred, average='macro'))
    micro_f1.append(f1_score(y_test, y_pred, average='micro'))
    print 'macro f1', f1_score(y_test, y_pred, average='macro')
    print 'micro f1', f1_score(y_test, y_pred, average='micro')
print 'avg macro f1', sum(macro_f1)/len(macro_f1)
print 'avg micro f1', sum(micro_f1)/len(micro_f1)

with open('./../log/svm.log','a') as fout:
    fout.write('================\n')
    if raw_flag:
        fout.write('use raw data\n')
    else:
        fout.write('use PCA data\n')
    fout.write('dimension=' + str(dimension) + '\n')
    fout.write('avg macro f1='+str(sum(macro_f1) / len(macro_f1))+'\n')
    fout.write('avg micro f1='+str(sum(micro_f1) / len(micro_f1))+'\n')