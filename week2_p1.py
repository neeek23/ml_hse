import sklearn
import numpy
import pandas
from pprint import pprint
# knn = sklearn.neighbors.KNeighborsClassifier()
data = numpy.fromfile('wine.data')
print((data))

data1 = pandas.read_csv('wine.data', index_col=0)
print(data1)