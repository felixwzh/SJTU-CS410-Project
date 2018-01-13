from sklearn.svm import SVC
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn import random_projection

label_df = pd.read_csv('./../data/all_label.csv')
y = np.array(label_df['BioSourceType-7'].tolist())
y_mask=np.zeros([y.shape[0],])
y_true=(y!=y_mask)
print y
print y_mask
print y_true
y_after=y[y_true]

print y_after




