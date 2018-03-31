import sys
from feature_extract import *
import pickle
from sklearn import linear_model
import scipy as sp
import os


def logloss(act, pred):
	epsilon = 1e-15
	pred = sp.maximum(epsilon, pred)
	pred = sp.minimum(1-epsilon, pred)
	ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
	ll = ll * -1.0/len(act)
	return ll

def LR():
	tp = 0.0
	fp = 0.0
	fn = 0.0
	tn = 0.0

	# read in training/test data with labels and create features

	testfull = readInData(testfilename)
	trainfull = readInData(trainfilename)


	train = [(x[0], x[1]) for x in trainfull]
	test = [(x[0], x[1]) for x in testfull]

	if len(test) <= 0 or len(train) <= 0:
		sys.exit()

	logistic = linear_model.LogisticRegression()
	train_data = []
	train_data_label = []
	test_data = []
	test_data_label = []
	for item in train:
		temp = [item[0]['f3stem'], item[0]['recall3stem'], item[0]['precision3stem'],
				item[0]['f2stem'], item[0]['recall2stem'], item[0]['precision2stem'],
				item[0]['f1stem'], item[0]['recall1stem'], item[0]['precision1stem'],
				item[0]['f3gram'], item[0]['recall3gram'], item[0]['precision3gram'],
				item[0]['f2gram'], item[0]['recall2gram'], item[0]['precision2gram'],
				item[0]['f1gram'], item[0]['recall1gram'], item[0]['precision1gram']]
		train_data.append(temp)
		train_data_label.append(item[1])
	for item in test:
		temp = [item[0]['f3stem'], item[0]['recall3stem'], item[0]['precision3stem'],
				item[0]['f2stem'], item[0]['recall2stem'], item[0]['precision2stem'],
				item[0]['f1stem'], item[0]['recall1stem'], item[0]['precision1stem'],
				item[0]['f3gram'], item[0]['recall3gram'], item[0]['precision3gram'],
				item[0]['f2gram'], item[0]['recall2gram'], item[0]['precision2gram'],
				item[0]['f1gram'], item[0]['recall1gram'], item[0]['precision1gram']]
		test_data.append(temp)
		test_data_label.append(item[1])

	print "Read in", len(train), "valid training data ... "
	print "Read in", len(test), "valid test data ...  "
	classifier = logistic.fit(train_data, train_data_label)
	
	predict_result = classifier.predict(test_data)
	counter = 0
	real_value=[]
	predict_value=[]
	for i, t in enumerate(predict_result):

		sent1 = testfull[i][2]
		sent2 = testfull[i][3]

		guess = t
		label = test_data_label[i]
		if label=='True':
			real_value.append(1)
		else:
			real_value.append(0)
		if guess=='True':
			predict_value.append(1)
		else:
			predict_value.append(0)
		# print guess, label
		if guess == 'True' and label == 'False':
			fp += 1.0
		elif guess == 'False' and label == 'True':
			fn += 1.0
		elif guess == 'True' and label == 'True':
			tp += 1.0
		elif guess == 'False' and label == 'False':
			tn += 1.0
		if label == guess:
			counter += 1.0
			# if guess:
			# print "GOLD-" + str(label) + "\t" + "SYS-" + str(guess) + "\t" + sent1 + "\t" + sent2

	P = tp / (tp + fp)
	R = tp / (tp + fn)
	F = 2 * P * R / (P + R)

	print
	#print "PRECISION: %s, RECALL: %s, F1: %s" % (P, R, F)
	print "ACCURACY: %s" % (counter / len(predict_result))

	print "# true pos:", tp
	print "# false pos:", fp
	print "# false neg:", fn
	print "# true neg:", tn
	probs = classifier.predict_proba(test_data)[:, 1]
	print "Logloss(soft): %s" % logloss(real_value, probs)

def readInData(filename):
	my_list=[]
	data = []
	index=0

	for line in open(filename):
		line = line.strip()
		if len(line.split('\t')) == 3:
			(origsent, candsent, judge) = line.split('\t')
		else:
			print line
			print '!---!'*20
			continue
		features = paraphrase_Das_features(origsent.decode('utf-8'), candsent.decode('utf-8'))
		my_list.append(features)
		if judge=='1':
			amt_label = 'True'
		else:
			amt_label='False'
		data.append((my_list[index], amt_label, origsent.decode('utf-8'), candsent.decode('utf-8')))
		index+=1
		
	
	# if filename == 'test_set.txt':
	# 	with open('test.pkl','wb') as f:
	# 		pickle.dump(my_list,f,pickle.HIGHEST_PROTOCOL)
	# else:
	# 	with open('train.pkl', 'wb') as f:
	# 		pickle.dump(my_list, f, pickle.HIGHEST_PROTOCOL)

	return data


if __name__=='__main__':
	trainfilename ='train_set.txt'
	testfilename ='test_set.txt'
	LR()
	