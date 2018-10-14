# d17

#KNN-test
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

#KNN-dating
import numpy as np
import matplotlib.pyplot as plt

def file2matrix(filename):              #文本转数组
    f = open(filename)
    Lines = f.readlines()
    nums = len(Lines)
    mat = np.zeros((nums,3))
    labels = []
    index = 0
    for line in Lines:
        line = line.strip()               #截掉回车字符
        list = line.split('\t')
        if list[-1] == 'largeDoses':
            list[-1] = 3
        elif list[-1] == 'smallDoses':
            list[-1] = 2
        else:
            list[-1] = 1
        mat[index,:] = list[0:3]
        labels.append(list[-1])
        index += 1
    return mat,labels
X_train,y_train = file2matrix('datingTestSet.txt')
X_test,y_test = file2matrix('datingTestSet2.txt')
X_train[:,0] = preprocessing.scale(X_train[:,0])
X_test[:,0] = preprocessing.scale(X_test[:,0])
print(X)
print(y)
plt.figure()
plt.subplot(221)
plt.scatter(X[:,0],X[:,1])
plt.subplot(222)
plt.scatter(X[:,0],X[:,2])
plt.subplot(223)
plt.scatter(X[:,1],X[:,2])
plt.show()
cla = KNeighborsClassifier()
cla.fit(X_train,y_train)
y_pred = cla.predict(X_test)
print(y_pred)
print(metrics.accuracy_score(y_test,y_pred))
