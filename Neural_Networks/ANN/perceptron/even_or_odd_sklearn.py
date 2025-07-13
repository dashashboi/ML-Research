from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import Perceptron
from sklearn.datasets import fetch_openml
import numpy as np
#import matplotlib.pyplot as plt
mnist1 = fetch_openml('mnist_784')

def even_or_odd(x):
    return x%2==0

X = np.asarray(mnist1.data)
X = X / 256 
y = even_or_odd(np.array(mnist1.target).T.astype(np.int16))

model = Perceptron(max_iter=10000000)

cv = RepeatedStratifiedKFold(n_splits=8, n_repeats=3, random_state=1)
# Evaluate the model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# Summarize the results
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))






