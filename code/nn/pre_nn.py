from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import pandas as pd

import os
import pprint
import sys

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.utils import to_categorical
from keras import regularizers, initializers
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau 
from keras.layers.normalization import BatchNormalization

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn import random_projection

import pickle

from keras import backend as K


pp = pprint.PrettyPrinter()
flags = tf.app.flags
flags.DEFINE_string("gpu", "0", "GPU(s) to use. [0]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate [2.5e-4]")
flags.DEFINE_integer("batch_size", 50, "The number of batch images [4]")
flags.DEFINE_integer("save_step", 500, "The interval of saveing checkpoints[500]")
flags.DEFINE_string("checkpoint", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log", "summary", "log [log]")
flags.DEFINE_integer("epoch", 10, "Epoch[10]")
flags.DEFINE_integer("pre_epoch", 5, "Epoch[10]")

FLAGS = flags.FLAGS
dim=22283
pre_train_dim=(8196,1024,128)
#project_path='/Users/jerry/Desktop/Gene_chip/'
project_path=''

label_class={
	'MaterialType': 2,
	'Sex':2,
	'DiseaseState':16,
	'BioSourceType':7
	}

def readData(label):
	x = np.load(project_path+'data/all_raw_data.npy')
	print(x.shape)
	label_df = pd.read_csv(project_path+'data/all_label.csv')
	y = np.array(label_df[label])
	return x,y

def init(X, Y):
	y_mask=np.zeros([Y.shape[0],])
	y_mask=(Y!=y_mask)
	Y=Y[y_mask]-1
	print(Y)
	X = X[y_mask]
	num_all = int(X.shape[0])
	mask = np.random.permutation(num_all)
	X = X[mask]
	Y = Y[mask]
	return X, Y


def add_initializer(model, kernel_initializer = initializers.random_normal(stddev=0.01,seed=None), bias_initializer = initializers.Zeros()):
	for layer in model.layers:
		if hasattr(layer, "kernel_initializer"):
			layer.kernel_initializer = kernel_initializer
		if hasattr(layer, "bias_initializer"):
			layer.bias_initializer = bias_initializer


def add_regularizer(model, kernel_regularizer = regularizers.l2(), bias_regularizer = regularizers.l2()):
	for layer in model.layers:
		if hasattr(layer, "kernel_regularizer"):
			layer.kernel_regularizer = kernel_regularizer
		if hasattr(layer, "bias_regularizer"):
			layer.bias_regularizer = bias_regularizer

def genChipModel(label):
	inputs = Input(shape = (dim, ))
	hidden1 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(8196, activation = 'relu')(inputs)))
	hidden2 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(512, activation = 'relu')(hidden1)))
	#hidden3 = Dropout(0.5)(Dense(512, activation = 'relu')(hidden2))
	hidden3 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(128, activation = 'relu')(hidden2)))
	#hidden5 = Dropout(0.5)(Dense(128, activation = 'relu')(hidden4))
	predictions = Dense(label_class[label], activation = 'softmax')(hidden3)
	model = Model(inputs = inputs, outputs = predictions)
	add_initializer(model)
	add_regularizer(model)
	return model

def train(label):
	
	x, y = readData(label)
	x,y = init(x, y)
	print(x.shape)

	#weights=pre_train(x_train,x_test)
	weights=pickle.load(open(project_path+'premodel/pre_weights.pkl','r'))
	
	
	#model=genChipModel(label)
	kf = KFold(n_splits=5,random_state=1,shuffle=True)
	kf.get_n_splits(x)
	print(kf)
	macro_f1=[]
	micro_f1=[]
	for train_index, test_index in kf.split(x):
		model = genChipModel(label)
		model.layers[1].set_weights(weights[0])
		model.layers[2].set_weights(weights[1])
		model.layers[3].set_weights(weights[2])
		X_train, X_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]
		y_train_cat = to_categorical(y_train, num_classes=label_class[label])
		y_test_cat= to_categorical(y_test, num_classes=label_class[label])
		
		opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])#'categorical_crossentropy', metrics=['accuracy'])
		network=model.summary()

		model.fit(X_train, y_train_cat, epochs=FLAGS.epoch, batch_size=FLAGS.batch_size, validation_data=(X_test, y_test_cat))
		

		y_result=model.predict(X_test, batch_size=FLAGS.batch_size, verbose=0)
		y_pred=y_result.tolist()
		y_pred = [result.index(max(result)) for result in y_pred]

		macro_f1.append(f1_score(y_test, y_pred, average='macro'))
		micro_f1.append(f1_score(y_test, y_pred, average='micro'))
		print ('macro f1', f1_score(y_test, y_pred, average='macro'))
		print ('micro f1', f1_score(y_test, y_pred, average='micro'))
		
		K.clear_session()
	mac=sum(macro_f1)/len(macro_f1)
	mic=sum(micro_f1)/len(micro_f1)
	print ('avg macro f1', mac)
	print ('avg micro f1', mic)

		
	#log:
	f=open(project_path+'log/'+label+'.txt','wb')
	f.write('----------network----------\n')
	f.write('\n----------config-----------\n')
	f.write('batch_size : %d\n ' % FLAGS.batch_size)
	f.write('lr : %f\n' % FLAGS.learning_rate)
	f.write('epoch : %d\n' % FLAGS.epoch)
	f.write('task name : %s\n' % label)
	f.write('data size :%d\n' %x.shape[0])
	f.write('\n----------result-----------\n')
	f.write('avg macro f1 : %f' % mac)
	f.write('avg micro f1 : %f' % mic)
	f.close()

if __name__ == '__main__':
	label='DiseaseState'
	#labels=['Sex','MaterialType','DiseaseState','BioSourceType']
	#for label in labels:
	#	train(label)
	train(label)
