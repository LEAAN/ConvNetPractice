from __future__ import print_function

import pandas
from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split
import numpy as np
import pickle
import sys
import set_para
import evalu
import plot

# img_rows, img_cols = 64, 64
ilist = set_para.lookfor(keyword = 'omni_1A', imagelist = None)
ilist = set_para.imagetodict(labelpath = None, labelname = None, imagelist = ilist,
    label_of_the_folder = 0, picpath = '/home/workstation/Documents/PIROPO/omni_1A/omni1A_training_additional_bg/')
ilist = set_para.imagetodict(labelpath = None, labelname = None, imagelist = ilist,
    label_of_the_folder = 0, picpath = '/home/workstation/Documents/PIROPO/omni_1A/omni1A_training_bg/')
ilist = set_para.imagetodict(labelpath = None, labelname = None, imagelist = ilist,
    label_of_the_folder = 0, picpath = '/home/workstation/Documents/PIROPO/omni_1A/omni1A_training_illum/')

train, test = train_test_split(pandas.DataFrame.from_dict(ilist), random_state = 1)

y_train = np.asarray(train.label.tolist())
y_test = np.asarray(test.label.tolist())
X_train = np.asarray(train.picarray.tolist())
X_test = np.asarray(test.picarray.tolist())

modelname = raw_input("Ask me == ")
modelpath = "/home/workstation/Documents/PIROPO/"
nb_classes = 2

X_train = X_train.reshape((len(X_train),-1))
X_test = X_test.reshape((len(X_test),-1))
y_train = y_train.ravel()
y_test = y_test.ravel()

clf = svm.SVC(verbose = True)
clf.fit(X_train, y_train)
y_predicted = clf.predict(X_test)

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(y_test, y_predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_predicted))