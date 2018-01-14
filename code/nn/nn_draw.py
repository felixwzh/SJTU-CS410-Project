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
from keras import backend as K

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn import random_projection

import matplotlib.pyplot as plt
import pickle
pp = pprint.PrettyPrinter()
flags = tf.app.flags
flags.DEFINE_string("gpu", "0", "GPU(s) to use. [0]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate [2.5e-4]")
flags.DEFINE_integer("batch_size", 5, "The number of batch images [4]")
flags.DEFINE_integer("save_step", 500, "The interval of saveing checkpoints[500]")
flags.DEFINE_string("checkpoint", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log", "summary", "log [log]")
flags.DEFINE_integer("epoch", 10, "Epoch[10]")
flags.DEFINE_integer("pre_epoch", 5, "Epoch[10]")

FLAGS = flags.FLAGS
dim=500
#pre_train_dim=(8196,1024,128)
project_path='/Users/jerry/Desktop/Gene_chip/'

label_class={
	'MaterialType-2': 2,
	'Sex-2':2,
	'DiseaseState-16':16,
	'BioSourceType-7':7
	}

def readData(label):
	x = np.load(project_path+'data/all_PCA_500.npy')
	print(x.shape)
	label_df = pd.read_csv(project_path+'data/all_label.csv')
	y = np.array(label_df[label])
	return x,y

def init(X, Y):
	y_mask=np.zeros([Y.shape[0],])
	y_mask=(Y!=y_mask)
	Y=Y[y_mask]-1
	X=X[y_mask]
	num_all = X.shape[0]
	num_train = int(0.8 * num_all)
	num_test = num_all - num_train
	# shuffle
	mask = np.random.permutation(num_all)
	X = X[mask]
	Y = Y[mask]
	# training data
	mask_train = range(num_train)
	X_train = X[mask_train]
	Y_train = Y[mask_train]
	#testing data
	mask_test = range(num_train, num_all)
	X_test = X[mask_test]
	Y_test = Y[mask_test]
	return X_train,Y_train,X_test,Y_test

def add_initializer(model, kernel_initializer = initializers.random_normal(stddev=0.01), bias_initializer = initializers.Zeros()):
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
	hidden1 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(256, activation = 'relu')(inputs)))
	hidden2 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(512, activation = 'relu')(hidden1)))
	#hidden3 = Dropout(0.5)(Dense(512, activation = 'relu')(hidden2))
	hidden3 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(256, activation = 'relu')(hidden2)))
	#hidden5 = Dropout(0.5)(Dense(128, activation = 'relu')(hidden4))
	predictions = Dense(label_class[label], activation = 'softmax')(hidden3)
	model = Model(inputs = inputs, outputs = predictions)
	add_initializer(model)
	add_regularizer(model)
	return model

def train(label):
	learning_rates=[0.0001,0.0005,0.001,0.01,0.03]
	x, y = readData(label)
	x_train,y_train,x_test,y_test = init(x, y)
	print(x.shape)
	acc=[]
	loss=[]

	y_train_cat = to_categorical(y_train, num_classes=label_class[label])
	y_test_cat= to_categorical(y_test, num_classes=label_class[label])
	#weights=pre_train(x_train,x_test)
	'''
	kf = KFold(n_splits=5,random_state=1,shuffle=True)
	kf.get_n_splits(x)
	print(kf)
	macro_f1=[]
	micro_f1=[]
	'''
	model=genChipModel(label)
	for learning_rate in learning_rates:
	
		opt = keras.optimizers.rmsprop(lr=learning_rate, decay=1e-6)
		

		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		history=model.fit(x_train, y_train_cat, epochs=FLAGS.epoch, batch_size=FLAGS.batch_size, verbose=1,validation_data=(x_test, y_test_cat))
		acc.append(history.history['acc'])
		loss.append(history.history['loss'])
	result={}
	result['loss']=loss
	result['acc']=acc
	output = open(project_path+'model/nn_result.pkl', 'wb')
	pickle.dump(result, output)
	output.close()

	fig = plt.figure(frameon=False)
	fig.set_size_inches(6, 4)

	plt.grid()
    # fig = pylab.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.yaxis.grid(color='gray', linestyle='dashed')
	ax.xaxis.grid(color='gray', linestyle='dashed')
	plt.plot(acc[0],'-ro', label="lr=0.0001")
	plt.plot(acc[1],'-go', label="lr=0.0005")
	plt.plot(acc[2],'-bo', label="lr=0.001")
	plt.plot(acc[3],'-yo', label="lr=0.01")
	plt.plot(acc[4],'-co', label="lr=0.03")
	#plt.plot(history.history['val_acc'])
	plt.xlabel('epoch')
	plt.ylabel('Accuracy')
	plt.legend(loc='lower right')
	plt.savefig(project_path+'plot/lr_acc2.pdf')
	plt.show()
	# summarize history for loss
	fig = plt.figure(frameon=False)
	fig.set_size_inches(6, 4)

	plt.grid()
    # fig = pylab.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.yaxis.grid(color='gray', linestyle='dashed')
	ax.xaxis.grid(color='gray', linestyle='dashed')
	plt.plot(loss[0],'-ro', label="lr=0.0001")
	plt.plot(loss[1],'-go', label="lr=0.0005")
	plt.plot(loss[2],'-bo', label="lr=0.001")
	plt.plot(loss[3],'-yo', label="lr=0.01")
	plt.plot(loss[4],'-co', label="lr=0.03")
	#plt.plot(history.history['val_acc'])
	plt.xlabel('epoch')
	plt.ylabel('Loss')
	plt.legend(loc='upper right')
	plt.savefig(project_path+'plot/lr_loss2.pdf')
	plt.show()

	

if __name__ == '__main__':
	label='BioSourceType-7'
	labels=['Sex','MaterialType','DiseaseState','BioSourceType']

	train(label)
