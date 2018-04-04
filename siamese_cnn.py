import numpy as np
import scipy.io
import sys
import argparse
import random
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Reshape
from keras.layers.merge import Concatenate
from keras.layers import Merge, Flatten
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint, RemoteMonitor
from itertools import izip_longest
from keras.layers import Conv2D, MaxPooling2D

from sklearn.externals import joblib
from sklearn import preprocessing

from spacy.en import English

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

def build_cnn(input_shape):
	model = Sequential()
	model.add(Conv2D(64, (3, 3), activation='relu', input_shape = input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	return model

def read_data(file_name):
	label = []
	q1 = []
	q2 = []
	for line in open(file_name):
		line = line.strip()
		pieces = line.split('\t')
		q1.append(pieces[0].decode('utf-8'))
		q2.append(pieces[1].decode('utf-8'))
		label.append(int(pieces[2]))
	# Select part of the vectors
	select = np.random.permutation(len(q1)/100)
	q1 = [q1[i] for i in select]
	q2 = [q2[i] for i in select]
	label = [label[i] for i in select]
	return q1, q2, label

def generate_vector(q1, q2, max_len, word_vec_dim):
	input_1 = np.zeros((len(q1), max_len, word_vec_dim))
	input_2 = np.zeros((len(q2), max_len, word_vec_dim))
	for i, item in enumerate(q1):
		#Get tokens
		tokens = nlp(item)
		question_tensor = np.zeros((max_len, word_vec_dim))
		for j in xrange(len(tokens)):
			if j < max_len:
				question_tensor[j, :] = tokens[j].vector
		input_1[i] = question_tensor

	for i, item in enumerate(q2):
		tokens= nlp(item)
		question_tensor = np.zeros((max_len, word_vec_dim))
		for j in xrange(len(tokens)):
			if j < max_len:
				question_tensor[j, :] = tokens[j].vector
		input_2[i]=question_tensor
	input_1= input_1.reshape(input_1.shape[0], max_len, word_vec_dim, 1)
	input_2= input_2.reshape(input_2.shape[0], max_len, word_vec_dim, 1)
	return input_1, input_2



if __name__ == "__main__":
	
	max_len=30
	word_vec_dim=300
	num_hidden_units_lstm=150
	num_hidden_layers_lstm=1
	nlp = English()
	input_shape=(max_len, word_vec_dim, 1)
	language_model = build_cnn(input_shape)
	language_model2= build_cnn(input_shape)
	model = Sequential()
	num_hidden_layers_mlp=3
	model.add(Merge([language_model, language_model2]))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])
	
	num_epochs=20
	batch_size=200
	
	print "Read training data"
	q1_train, q2_train, label_train = read_data("train_set.txt")
	
	print "Read test data"
	q1_test, q2_test, label_test = read_data("test_set.txt")

	#Set input vector for the train data of the network
	input_1, input_2 = generate_vector(q1_train, q2_train, max_len, word_vec_dim)

	label_train= np.array(label_train).reshape(input_2.shape[0], 1)

	print "input1 = ", input_1.shape, "input2 = ", input_2.shape, "label_train = ", label_train.shape
	
	input_1_test, input_2_test = generate_vector(q1_test, q2_test, max_len, word_vec_dim)
	
	label_test= np.array(label_test).reshape(input_2_test.shape[0], 1)

	print "input1 = ", input_1_test.shape, "input2 = ", input_2_test.shape, "label_test = ", label_test.shape

	model.fit([input_1, input_2], label_train,
          batch_size=batch_size,
          epochs=num_epochs,
          verbose=1, validation_data= ([input_1_test, input_2_test], label_test))
	score = model.evaluate([input_1_test, input_2_test], label_test, verbose=1)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	