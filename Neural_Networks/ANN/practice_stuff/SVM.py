from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

#X[0]:
plot_X = np.array([1, 2, 5, 1.5, 8, 1, 9, 7, 8.7, 2.3, 5.5, 7.7, 6.1, 8, 8])
#X[1]:
plot_y = np.array([2, 4, 8, 1.8, 8, 0.6, 11, 10, 9.4, 4, 3, 8.8, 7.5, 8, 2]) 
plt.scatter(plot_X, plot_y)
plt.show()

# Load a sample dataset (e.g., diabetes dataset)
training_X = np.stack((plot_X, plot_y)).T
training_y = [0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1,1]

clf = svm.SVC(kernel='linear', C=1.0) #if kernel is not mentioned in the SVC parameters, RBF is used 
clf.fit(training_X, training_y)

print(clf.coef_, clf.intercept_)
# Extract the model parameter for the linear SVM from the trained SVM model
w = clf.coef_[0]

# Calculate the y-offset for the linear SVM
# we are finding the gradient of y=mx+c by using the vector w  
# for the unit vector w[1] is y and w[0] is, and the hyperplane is perpendicular, so we flip fraction and negate (m1*m2=-1)
a = -w[0] / w[1]

# make the x-axis space for the data points
XX = np.linspace(-5, 13)

# get the labels to plot the decision boundary
yy = a * XX - clf.intercept_[0] / w[1]

# plotting the decision boundary
plt.plot(XX, yy, 'k-')

# showing the plot visually
plt.scatter(training_X[:, 0], training_X[:, 1], c=yy)
plt.legend()
plt.show()


#When you fit an SVC (Support Vector Classification) model without using GridSearchCV, it does not automatically search for the best hyperparameters. Instead, it uses the default hyperparameters specified in the SVC constructor.
#By default, the SVC model uses the following hyperparameters:

#C: The regularization parameter (inverse of the regularization strength). Default value is 1.0.
#kernel: The kernel function (e.g., ‘linear’, ‘rbf’, etc.). Default value is ‘rbf’.
#gamma: The kernel coefficient for ‘rbf’, ‘poly’, and ‘sigmoid’. Default value is ‘scale’ (which depends on the data).

#to check about using gridsearchCV go to the SVC iris in practice stuff folder

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('income_evaluation.csv')
X_train, X_test, y_train, y_test = train_test_split(df.drop("income", axis=1),df["income"], test_size=0.2)

svc = svm.SVC(random_state=101)
svc.fit(X_train,y_train)

accuracies = cross_val_score(svc,X_train,y_train,cv=5)

print("Train Score:",np.mean(accuracies))
print("Test Score:",svc.score(X_test,y_test))




grid = {
    'C':[0.01,0.1,1,10],
    'kernel' : ["linear","poly","rbf","sigmoid"],
    'degree' : [1,3,5,7],
    'gamma' : [0.01,1]
}

svm  = svm.SVC ()

svm_cv = GridSearchCV(svm, grid, cv = 5)
svm_cv.fit(X_train,y_train)

print("Best Parameters:",svm_cv.best_params_)
print("Train Score:",svm_cv.best_score_)
print("Test Score:",svm_cv.score(X_test,y_test))

