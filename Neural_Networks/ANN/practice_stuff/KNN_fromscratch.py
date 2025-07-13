import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = sns.load_dataset('iris')
y = LabelEncoder().fit_transform(iris["species"])
X = iris.drop('species',axis=1)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33 )
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def predict(test_data,train_data, k):
    
    diff = np.subtract(train_data, test_data)
    diff = np.square(diff)
    distance = np.sqrt(np.sum(diff, axis=1))
    classes = []
    for i in range(k):
        classes.append(y_train[np.argmin(distance)])
        distance[np.argmin(distance)] = np.max(distance)
    
    cls = max(set(classes), key=classes.count) #really good funtion
    return cls

error_rate=[]
my_error_rate=[]
for k in range(1,40,2):
    y_pred=[]
    for i in range(len(y_test)):
        y_pred.append(predict(X_test[i,:],X_train, k))
    knn= KNeighborsClassifier(k)
    knn.fit(X_train,y_train)
    y_pred_sklearn=knn.predict(X_test)
    my_error_rate.append(np.mean(y_pred!=y_test))
    error_rate.append(np.mean(y_pred_sklearn!=y_test))
    accuracy = knn.score(X_test, y_test)
    print( "Model Accuracy: %.4f" % accuracy )

print(error_rate, my_error_rate)

