import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier
# X = np.array([[0.6, 2, 1], [0.5, 4, 0], [1.1, 6, 1.1]])
# y = np.array([0, 1, 1])
# clf = DecisionTreeClassifier()
# clf.fit(X, y)
# importances = clf.feature_importances_
# print(importances)

# def f(x: pandas.DataFrame):
#     for i in x['Age']:
#         if np.isnan(i):
#             x.drop()
#


data = pandas.read_csv('titanic.csv', index_col='PassengerId')

X = data[['Pclass', 'Fare', 'Age', 'Sex']]
print(X)
y= data['Survived']
X_nan = X['Age']
print(X_nan)

# X = X[not np.isnan(X.Age)]
X = X.dropna()
print(X)
# TODO: заменить пол
X['Sex']
print(X)

# clf = DecisionTreeClassifier()
# clf.fit(X, y)
# importances = clf.feature_importances_
# print(importances)