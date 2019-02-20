import numpy as np
import pandas
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

from sklearn.model_selection import KFold, cross_val_score
from pprint import pprint
# knn = sklearn.neighbors.KNeighborsClassifier()
# data = numpy.loadtxt('wine.data', )
# print((data))

scaler = StandardScaler()
data1 = pandas.read_csv('perceptron-train.csv', header=None)
data2 = pandas.read_csv('perceptron-test.csv', header=None)
X_train = data1.loc[:,1:]
X_test = data2.loc[:,1:]
Y_train = data1[0]
Y_test = data2[0]
clf = Perceptron(random_state=241)
clf.fit(X_train, Y_train)
predictions1 = clf.predict(X_test)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf = Perceptron(random_state=241)
clf.fit(X_train_scaled, Y_train)
predictions2 = clf.predict(X_test_scaled)
print(predictions2)
print(predictions1)

a = (accuracy_score(y_true=Y_test, y_pred=predictions1))

b = (accuracy_score(y_true=Y_test, y_pred=predictions2))
print(b-a)
print('Правильно!')
# scores = []
# for i in range(1,51):
#     model = KNeighborsClassifier(n_neighbors=i)
#     kfold = KFold(n_splits=5, random_state=42, shuffle=True)
#     a = cross_val_score(estimator=model, X=x, cv=kfold, y = y, scoring='accuracy')
#     scores.append(np.mean(a))
# print(max(scores))
# print(scores.index(max(scores)))
#
# x_1 = scale(x.astype(float))
# scores_1 = list()
# for i in range(1,51):
#     model = KNeighborsClassifier(n_neighbors=i)
#     kfold = KFold(n_splits=5, random_state=42, shuffle=True)
#     a = cross_val_score(estimator=model, X=x_1, cv=kfold, y = y, scoring='accuracy')
#     scores_1.append(np.mean(a))
#
# print(max(scores_1))
# print(scores_1.index(max(scores_1)))
