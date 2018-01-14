from sklearn.svm import SVC
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import argparse
import xgboost as xgb


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task_num', type=int,default=0) # task_list=['MaterialType-2','Sex-2','DiseaseState-16','BioSourceType-7']
parser.add_argument('-rf', '--raw_flag', type=bool, default=False) # `True` use raw data, `False` use pca data
parser.add_argument('-dm', '--dimension', type=int,default=22283)

#  {'newton-cg', 'lbfgs', 'sag', 'saga'},

# save args
args = parser.parse_args()

# configuration
task_list=['MaterialType-2','Sex-2','DiseaseState-16','BioSourceType-7']
num_class_list=[2,2,16,7]

param = {
         # 'reg_alpha': 0.0001,
         # 'colsample_bytree': 0.8,
         # 'scale_pos_weight': 1,
         'learning_rate': 0.07,
         # 'min_child_weight': 11,
         'subsample': 0.8,
         # 'reg_lambda': 0.0049,
         'seed': 1,
         'objective': 'multi:softmax',
         'max_depth': 1,
         'gamma': 0.0,
         'silent': 1,
         'num_class': num_class_list[args.task_num]
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

y=y-1
print y

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

    y_test=y[test_index]

    dtrain = xgb.DMatrix(data=X[train_index], label=y[train_index])
    dtest = xgb.DMatrix(data=X[test_index], label=y[test_index])

    # evallist = [(dtrain, 'train'), (dtest, 'eval')]
    bst = xgb.train(param, dtrain, 100,
                    # evallist
                    )

    y_pred = bst.predict(dtest)


    macro_f1.append(f1_score(y_test, y_pred, average='macro'))
    micro_f1.append(f1_score(y_test, y_pred, average='micro'))
    print 'macro f1', f1_score(y_test, y_pred, average='macro')
    print 'micro f1', f1_score(y_test, y_pred, average='micro')

print 'avg macro f1', sum(macro_f1)/len(macro_f1)
print 'avg micro f1', sum(micro_f1)/len(micro_f1)


with open('./../log/xgb.csv', 'a') as fout:
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
