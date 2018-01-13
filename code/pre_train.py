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

import pickle
import tensorflow as tf
import os
import keras.backend.tensorflow_backend as KTF


pp = pprint.PrettyPrinter()
flags = tf.app.flags
flags.DEFINE_string("gpu", "0", "GPU(s) to use. [0]")
flags.DEFINE_float("learning_rate", 2.5e-3, "Learning rate [2.5e-4]")
flags.DEFINE_integer("batch_size", 200, "The number of batch images [4]")
flags.DEFINE_integer("save_step", 500, "The interval of saveing checkpoints[500]")
flags.DEFINE_string("checkpoint", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log", "summary", "log [log]")
flags.DEFINE_integer("epoch", 100, "Epoch[10]")
flags.DEFINE_integer("pre_epoch", 10, "Epoch[10]")

FLAGS = flags.FLAGS
dim=22283
pre_train_dim=(8196,512,128)

project_path='../'

def readData(label):
	x = np.load(project_path+'data/all_raw_data.npy')
	print(x.shape)
	label_df = pd.read_csv(project_path+'data/all_label.csv')
	y = np.array(label_df[label])
	return x,y

def init(X, Y):
	#x=np.array([X[i] for i in Y if i >0])
	#y=np.array([Y[i]-1 for i in Y if i >0])
	num_all = int(X.shape[0])
	num_train = int(0.8 * num_all)
	num_test = num_all - num_train
	# shuffle
	mask = np.random.permutation(num_all)
	X = X[mask]
	Y = Y[mask]
	# training data
	mask_train = range(num_train)
	X_train = X
	Y_train = Y
	#testing data
	mask_test = range(num_train, num_all)
	X_test = X[mask_test]
	Y_test = Y[mask_test]
	# Y_train, Y_test = np.expand_dims(Y_train, axis=1), np.expand_dims(Y_test, axis=1)
	print('All data shape: ', X.shape)
	return X_train, Y_train, X_test, Y_test

def add_regularizer(model, kernel_regularizer = regularizers.l2(), bias_regularizer = regularizers.l2()):
	for layer in model.layers:
		if hasattr(layer, "kernel_regularizer"):
			layer.kernel_regularizer = kernel_regularizer
		if hasattr(layer, "bias_regularizer"):
			layer.bias_regularizer = bias_regularizer

def pre_model1(pre_train_dim):

	inputs = Input(shape = (22283, ))
	hidden1 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(pre_train_dim[0], activation = 'relu')(inputs)))
	outputs = Dense(dim, activation = 'relu')(hidden1)
	model = Model(inputs = inputs, outputs = outputs)
	add_regularizer(model)
	return model

def pre_model2(pre_train_dim,weight):

	inputs = Input(shape = (22283, ))
	hidden1 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(pre_train_dim[0], activation = 'relu',trainable=False)(inputs)))
	hidden2 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(pre_train_dim[1], activation = 'relu')(hidden1)))
	hidden3 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(pre_train_dim[0], activation = 'relu')(hidden2)))
	outputs= Dense(dim, activation = 'relu',trainable=False)(hidden3)
	model = Model(inputs = inputs, outputs = outputs)

	add_regularizer(model)
	return model

def pre_model3(pre_train_dim,weight):
	inputs = Input(shape = (22283, ))
	hidden1 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(pre_train_dim[0], activation = 'relu',trainable=False)(inputs)))
	hidden2 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(pre_train_dim[1], activation = 'relu',trainable=False)(hidden1)))
	hidden3 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(pre_train_dim[2], activation = 'relu')(hidden2)))
	hidden4 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(pre_train_dim[1], activation = 'relu')(hidden3)))
	hidden5 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(pre_train_dim[0], activation = 'relu',trainable=False)(hidden4)))
	outputs= Dense(dim, activation = 'relu',trainable=False)(hidden5)
	model = Model(inputs = inputs, outputs = outputs)
	add_regularizer(model)
	return model
'''
def pre_model4(pre_train_dim,weight):
	inputs = Input(shape = (22283, ))
	hidden1 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(pre_train_dim[0], activation = 'relu',trainable=False)(inputs)))
	hidden2 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(pre_train_dim[1], activation = 'relu',trainable=False)(hidden1)))
	hidden3 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(pre_train_dim[2], activation = 'relu',trainable=False)(hidden2)))
	hidden4 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(pre_train_dim[3], activation = 'relu')(hidden3)))
	hidden5 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(pre_train_dim[2], activation = 'relu')(hidden4)))
	hidden6 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(pre_train_dim[1], activation = 'relu',trainable=False)(hidden5)))
	hidden7 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(pre_train_dim[0], activation = 'relu',trainable=False)(hidden6)))
	outputs= Dense(dim, activation = 'relu',trainable=False)(hidden5)
	model = Model(inputs = inputs, outputs = outputs)
	add_regularizer(model)
	return model
'''
def pre_train(x_train , x_test):
	batch_size_now=100
	weights=[]
	model_hidden1=pre_model1(pre_train_dim)
	model_hidden1.compile(loss='mse', optimizer='adam')
	model_hidden1.fit(x_train, x_train, epochs=FLAGS.pre_epoch, batch_size=batch_size_now, validation_data=(x_test, x_test))
	weights.append(model_hidden1.layers[1].get_weights())
	#model_hidden1.save(project_path+'model_hidden1.h5')
	model_hidden2=pre_model2(pre_train_dim,weights)
	model_hidden2.layers[1].set_weights(model_hidden1.layers[1].get_weights())
	model_hidden2.layers[-1].set_weights(model_hidden1.layers[-1].get_weights())
	model_hidden2.compile(loss='mse', optimizer='adam')
	model_hidden2.fit(x_train, x_train, epochs=FLAGS.pre_epoch, batch_size=batch_size_now, validation_data=(x_test, x_test))
	weights.append(model_hidden2.layers[2].get_weights())

	model_hidden3=pre_model3(pre_train_dim,weights)
	model_hidden3.layers[1].set_weights(model_hidden2.layers[1].get_weights())
	model_hidden3.layers[2].set_weights(model_hidden2.layers[2].get_weights())
	model_hidden3.layers[-2].set_weights(model_hidden2.layers[-2].get_weights())
	model_hidden3.layers[-1].set_weights(model_hidden2.layers[-1].get_weights())
	model_hidden3.compile(loss='mse', optimizer='adam')
	model_hidden3.fit(x_train, x_train, epochs=FLAGS.pre_epoch, batch_size=batch_size_now, validation_data=(x_test, x_test))
	weights.append(model_hidden3.layers[3].get_weights())
	'''
	model_hidden4=pre_model4(pre_train_dim,weights)
	#model_hidden4.layers[1].set_weights(model_hidden3.layers[1].get_weights())
	#model_hidden4.layers[2].set_weights(model_hidden3.layers[2].get_weights())
	#model_hidden4.layers[3].set_weights(model_hidden3.layers[3].get_weights())
	#model_hidden4.layers[-2].set_weights(model_hidden3.layers[-2].get_weights())
	#model_hidden4.layers[-3].set_weights(model_hidden3.layers[-3].get_weights())
	#model_hidden4.layers[-1].set_weights(model_hidden3.layers[-1].get_weights())
	model_hidden4.compile(loss='mse', optimizer='sgd')
	model_hidden4.fit(x_train, x_train, epochs=FLAGS.pre_epoch, batch_size=100, validation_data=(x_test, x_test))
	weights.append([model_hidden4.layers[4].get_weights(),model_hidden4.layers[-4].get_weights()])
	'''
	#save the weights of hidden layer:
	output = open(project_path+'model/pre_weights.pkl', 'wb')
	pickle.dump(weights, output)
	output.close()

	return weights

if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	# os.system('echo $CUDA_VISIBLE_DEVICES')
	# config = tf.ConfigProto(device_count={"CPU":16},
  # inter_op_parallelism_threads=1,
  # intra_op_parallelism_threads=1,)
	# session = tf.Session(config=config)
	# KTF.set_session(session)



	label='MaterialType-2'
	x, y = readData(label)

	x_train, y_train, x_test, y_test = init(x, y)
	print('----------------pre_train start---------------------')
	weights=pre_train(x_train,x_test)