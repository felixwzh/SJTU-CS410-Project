from sklearn.svm import SVC
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task_num', type=int,default=0) # task_list=['MaterialType-2','Sex-2','DiseaseState-16','BioSourceType-7']
parser.add_argument('-rf', '--raw_flag', type=bool, default=False) # `True` use raw data, `False` use pca data
parser.add_argument('-dm', '--dimension', type=int,default=22283)
parser.add_argument('-C', '--C', type=float, default=1)
parser.add_argument('-p', '--penalty', type=str, default='l2')
parser.add_argument('-svl', '--solver', type=str, default='sag')
#  {'newton-cg', 'lbfgs', 'sag', 'saga'},

# save args
args = parser.parse_args()

# configuration
task_list=['MaterialType-2','Sex-2','DiseaseState-16','BioSourceType-7']
param={
    'C':args.C,
    'penalty':args.penalty,
    'solver':args.solver,

    'max_iter': 100,
    'multi_class': 'multinomial',
    'n_jobs': -1,
    'random_state':1,
}
task_name=task_list[args.task_num]
raw_flag=False
print 'raw_flag',raw_flag
dimension=args.dimension



# load data
if raw_flag:
    X = np.load("./../data/all_raw_data.npy")
    X = X[:,0:dimension]

else:
    X = np.load("./../data/all_PCA_"+str(dimension)+".npy")

# todo another way to decrease dimension
# transformer = random_projection.SparseRandomProjection()
# X = transformer.fit_transform(X)

# load label
label_df = pd.read_csv('./../data/all_label.csv')
y = np.array(label_df[task_name].tolist())
y_mask=np.zeros([y.shape[0],])
y_mask=(y!=y_mask)

# mask data
y=y[y_mask]
X = X[y_mask]


print 'data shape:',X.shape

# split data random_state=1
kf = KFold(n_splits=5,random_state=1,shuffle=True)
kf.get_n_splits(X)
print(kf)

# store f1
macro_f1=[]
micro_f1=[]

for train_index, test_index in kf.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf= LogisticRegression(**param)
    clf.fit(X, y)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    macro_f1.append(f1_score(y_test, y_pred, average='macro'))
    micro_f1.append(f1_score(y_test, y_pred, average='micro'))
    print 'macro f1', f1_score(y_test, y_pred, average='macro')
    print 'micro f1', f1_score(y_test, y_pred, average='micro')

print 'avg macro f1', sum(macro_f1)/len(macro_f1)
print 'avg micro f1', sum(micro_f1)/len(micro_f1)


with open('./../log/LR.csv', 'a') as fout:
    fout.write(task_name+',')
    if raw_flag:
        fout.write('use raw data,')
    else:
        fout.write('use PCA data,')
    for key in param:
        fout.write(key+':'+str(param[key])+',')
    fout.write(str(dimension) + ',')
    fout.write(str(sum(macro_f1) / len(macro_f1)) + ',')
    fout.write(str(sum(micro_f1) / len(micro_f1)) + ',\n')
