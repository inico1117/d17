# d17
#KNN
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def createDataSet():
    groups = np.array([[1.0,1.1],
                       [1.0,1.0],
                       [0,0],
                      [0,0.1],
                       [1.1,0]])
    labels = ['A','A','B','B','B']
    return groups,labels
X,y = createDataSet()
X_train,X_test,y_train,y_test = train_test_split(X,y)
cla = KNeighborsClassifier()
cla.fit(X_train,y_train)
y_pred = cla.predict(X_test)
print(cla.predict(X_test).reshape(1,-1))
print(metrics.accuracy_score(y_test,y_pred))
color = np.arctan2(X[:,0],X[:,1])
plt.figure()
plt.scatter(X[:,0],X[:,1],c=color,marker='o',label='x')
plt.legend(loc='upper left')
plt.show()
