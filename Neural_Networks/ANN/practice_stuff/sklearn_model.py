from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import Perceptron

# Define the dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# Define the model
model = Perceptron(max_iter=10000000)
#model.fit(X, y)  ...we arent doing this cross val score does it for us

# Define model evaluation method, cv = cross validation
#splits split the data(X) into that many portions with equal number of classes in each portion (stratified kfold,a better version of K-fold)
#repeated tells that it is repeated 3 times with different randomization each time
cv = RepeatedStratifiedKFold(n_splits=8, n_repeats=3, random_state=1)
# Evaluate the model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# Summarize the results
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

model.fit(X, y)
# Define new data
row = [0.12777556,-3.64400522,-2.23268854,-1.82114386,1.75466361,0.1243966,1.03397657,2.35822076,1.01001752,0.56768485]
# Make a prediction
y_hat = model.predict([row])
# Summarize the prediction
print('Predicted Class: %d' % y_hat)




#used to check best combination of hyperparamteres, ie tuning hyperparameters
#we are gonna provide it only the learning rate (eta0)
from sklearn.model_selection import GridSearchCV

# define grid
grid = dict()
grid['eta0'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
# define search
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('Best Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))





#number of epochs
grid['max_iter'] = [1, 10, 100, 1000, 10000]
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('Best Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))






## NOW WE LEARN REGULARISATION, first lasso (l1) regression
from sklearn.linear_model import LassoCV
from sklearn.datasets import load_diabetes

# Load a sample dataset (e.g., diabetes dataset)
X, y = load_diabetes(return_X_y=True)
# Create a LassoCV model
lasso_cv = LassoCV(cv=5)  # 5-fold cross-validation, the model is linear regression still
# Fit the model to the data
lasso_cv.fit(X, y)
# Get the optimal λ
best_lambda = lasso_cv.alpha_




## now ridge(l2) regression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
import numpy as np

# Generate some sample data
n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)

# Define the model, this is also still an extension of linear regression
ridge_model = Ridge(alpha=1.0)  # You can adjust the regularization strength (alpha)

# Fit the model to the data
ridge_model.fit(X, y)

# Get the coefficients
coefficients = ridge_model.coef_
intercept = ridge_model.intercept_

print("Coefficients:", coefficients)
print("Intercept:", intercept)

ridge_cv = RidgeCV(cv=5)  # 5-fold cross-validation, the model is linear regression still
# Fit the model to the data
ridge_cv.fit(X, y)
# Get the optimal λ
best_lambda = ridge_cv.alpha_

