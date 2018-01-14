from __future__ import division, print_function
import tensorflow as tf
import numpy as np
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

project_path='/Users/jerry/Desktop/Gene_chip/'
pp = pprint.PrettyPrinter()
flags = tf.app.flags
flags.DEFINE_string("gpu", "0", "GPU(s) to use. [0]")
flags.DEFINE_float("learning_rate", 2.5e-3, "Learning rate [2.5e-4]")
flags.DEFINE_integer("batch_size", 200, "The number of batch images [4]")
flags.DEFINE_integer("save_step", 500, "The interval of saveing checkpoints[500]")
flags.DEFINE_string("checkpoint", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log", "summary", "log [log]")
flags.DEFINE_integer("epoch", 100, "Epoch[10]")
dim=22283
flags.DEFINE_integer("pre_epoch", 5, "Epoch[10]")

FLAGS = flags.FLAGS
def GeneModel():
	inputs = Input(shape = (dim, ))
	hidden1 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(8196, activation = 'relu',trainable=False)(inputs)))
	hidden2 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(512, activation = 'relu',trainable=False)(hidden1)))
	#hidden3 = Dropout(0.5)(Dense(512, activation = 'relu')(hidden2))
	#hidden3 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(128, activation = 'relu')(hidden2)))
	#hidden5 = Dropout(0.5)(Dense(128, activation = 'relu')(hidden4))
	#predictions = Dense(label_class[label], activation = 'softmax')(hidden3)
	model = Model(inputs = inputs, outputs = hidden2)

	weights=pickle.load(open(project_path+'premodel/pre_weights.pkl','r'))
	model.layers[1].set_weights(weights[0])
	model.layers[2].set_weights(weights[1])
	return model

def dim_reduction():
	model=GeneModel()
	x = np.load(project_path+'data/all_raw_data.npy')
	print(x.shape)

	opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	x_reduction=model.predict(x, batch_size=FLAGS.batch_size, verbose=0)

	np.save("autoencode.npy", x_reduction)

if __name__ == '__main__':
	dim_reduction()

