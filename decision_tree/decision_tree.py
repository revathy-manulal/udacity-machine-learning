#!/usr/bin/python



import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

from sklearn import tree
from sklearn.metrics import accuracy_score
clf1 = tree.DecisionTreeClassifier(min_samples_split=2)
clf1 = clf1.fit(features_train, labels_train)
labels_pred = clf1.predict(features_test)

acc_min_samples_split_2 = accuracy_score(labels_pred, labels_test)


clf2 = tree.DecisionTreeClassifier(min_samples_split=50)
clf2 = clf2.fit(features_train, labels_train)
labels_pred = clf2.predict(features_test)

acc_min_samples_split_50 = accuracy_score(labels_pred, labels_test)
