# Importing essential libraries
import numpy as np
import matplotlib.pyplot as plt

# Importing Boston Dataset from SciKit Learn
from sklearn.datasets import load_boston

# Importing linear regression, preprocessing, and pipeline modules
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Importing model validation and ROC modules
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import roc_curve,plot_roc_curve, balanced_accuracy_score

# Identifying variables
X,y = load_boston(return_X_y=True)
y = (y > y.mean()).astype(int)
# Splitting training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)

# Creating a pipeline
analysis = make_pipeline(StandardScaler(),LogisticRegression())
# Fitting the data
analysis.fit(X_train,y_train)
# plotting the data
plot_roc_curve(analysis,X_train,y_train)




from sklearn.datasets import load_breast_cancer
# Importing ROC Module
from sklearn.metrics import roc_auc_score

# Defining test, train, dependant, and independent variables and fitting the model
X, y = load_breast_cancer(return_X_y=True)
clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)

# Printing ROC score
roc_auc_score(y, clf.decision_function(X))




