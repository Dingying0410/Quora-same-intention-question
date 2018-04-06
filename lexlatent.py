import numpy as np
import scipy.io
import sys
import argparse
import random
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Reshape
from keras.layers.merge import Concatenate
from keras.layers import Merge, Flatten
from keras.layers.recurrent import LSTM
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint, RemoteMonitor
from itertools import izip_longest
from keras.layers import Conv2D, MaxPooling2D

def build_cnn(input_shape):
	model = Sequential()
	model.add(Conv2D(64, kernel_size=(3,3),
                 activation='relu',
                 input_shape=input_shape))
	# model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, kernel_size=(3,3),
                 activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.1))
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.2))
	return model

if __name__=="__main__":
	
	num_epochs= 50
	batch_size= 200

	print "Loading training data"
	with open('lexlatenttrain_set.txt.ls') as f:
		data= f.readlines()
	train1= []
	train2= []
	for i in range(0, len(data), 2):
		# print data[i].split()
		train1.append(data[i].split())
		train2.append(data[i+1].split())

	train_label=[]
	for line in open('data/train_set.txt'):
		line = line.strip()
		pieces= line.split('\t')
		train_label.append(int(pieces[2]))

	print "Loading test data"
	with open('lexlatenttest_set.txt.ls') as f:
		data= f.readlines()
	test1= []
	test2= []
	
	for i in range(0, len(data), 2):
		test1.append(data[i].split())
		test2.append(data[i+1].split())

	test_label=[]
	for line in open('data/test_set.txt'):
		line = line.strip()
		pieces= line.split('\t')
		test_label.append(int(pieces[2]))


	train_q1= np.zeros((len(train1), len(train1[0])))
	train_q2= np.zeros((len(train2), len(train2[0])))
	# print train1[0], train1[0][0]
	for i in range(len(train1)):
		train_q1[i]= train1[i]
		train_q2[i]= train2[i]

	test_q1= np.zeros((len(test1), len(test1[0])))
	test_q2= np.zeros((len(test2), len(test2[0])))
	# print test1[0], test1[0][0]
	for i in range(len(test1)):
		test_q1[i]= test1[i]
		test_q2[i]= test2[i]

	# Make the input vector to be a 10 * 10 matrix, which serves as the input for CNN
	train_q1= np.array(train_q1).reshape(train_q1.shape[0], 10, 10, 1)
	train_q2= np.array(train_q2).reshape(train_q2.shape[0], 10, 10, 1)
	test_q1= np.array(test_q1).reshape(test_q1.shape[0], 10, 10, 1)
	test_q2= np.array(test_q2).reshape(test_q1.shape[0], 10, 10, 1)

	print "Train data size = ", train_q1.shape
	print "Test data size = ", test_q1.shape

	train_label= np.array(train_label).reshape(len(train_label), 1)
	test_label= np.array(test_label).reshape(len(test_label), 1)

	input_shape=(10, 10, 1)

	language_model = build_cnn(input_shape)
	language_model2= build_cnn(input_shape)
	
	
	model = Sequential()
	num_hidden_layers_mlp=3
	model.add(Merge([language_model, language_model2]))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	model.fit([train_q1, train_q2], train_label,
          batch_size=batch_size,
          epochs=num_epochs,
          verbose=1, validation_data= ([test_q1, test_q2], test_label))
	score = model.evaluate([test_q1, test_q2], test_label, verbose=1)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])





