import numpy as np
import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

from sklearn.model_selection import KFold, cross_val_score
from pprint import pprint
# knn = sklearn.neighbors.KNeighborsClassifier()
# data = numpy.loadtxt('wine.data', )
# print((data))

data1 = pandas.read_csv('wine.data', header=None)
# indexes = np.arange(len(data1))
# data1.index = indexes
# print((data1[[1]]))
# print(data1)
y = data1[0]
x = data1.loc[:,1:]
# print(x)
scores = []
for i in range(1,51):
    model = KNeighborsClassifier(n_neighbors=i)
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    a = cross_val_score(estimator=model, X=x, cv=kfold, y = y, scoring='accuracy')
    scores.append(np.mean(a))
print(max(scores))
print(scores.index(max(scores)))

x_1 = scale(x.astype(float))
scores_1 = list()
for i in range(1,51):
    model = KNeighborsClassifier(n_neighbors=i)
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    a = cross_val_score(estimator=model, X=x_1, cv=kfold, y = y, scoring='accuracy')
    scores_1.append(np.mean(a))

print(max(scores_1))
print(scores_1.index(max(scores_1)))
