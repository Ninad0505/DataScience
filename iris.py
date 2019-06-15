from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
# from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
iris = load_iris()
#print(iris)
#print(iris.keys())
#print(iris['DESCR'])
#print(iris.data)
#print(iris.data.T)  # in this all four columns get separated.

features = iris.data.T

sepal_length = features[0]
sepal_width = features[1]
petal_length = features[2]
petal_width = features[3]

#plt.scatter(sepal_length, sepal_width, c = iris.target)
#plt.xlabel('sepal length')
#plt.ylabel('sepal width')
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

#X_new = np.array([[5.0, 2.9, 1, 0.2]])
#prediction = knn.predict(X_new)

#print(prediction)
y_pred = knn.predict(X_test)
print(knn.score(X_test, y_test))
#print(f1_score(y_test, y_pred))           # can only be used for binary predictions
#print(accuracy_score(y_test, y_pred))     # can only be used for binary predictions
print(y_pred)