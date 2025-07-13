import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
from sklearn.model_selection import train_test_split
# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#from sklearn import cross_validation  <<--- this was updated
from sklearn import model_selection
# value of K is 10.
splits = model_selection.KFold( n_splits=10, random_state=8, shuffle=True)

#the above code makes indexes for n kfolds
results = []
for train_index, val_index in splits.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    model = LogisticRegression(max_iter=1000000)
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    results.append(np.mean(y_val!=val_predictions))
    #aggregate all values after this 

print(results)





