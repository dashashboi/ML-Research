#In this file I keep the extra codes and snippets that i want to remember
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
from sklearn.model_selection import train_test_split
# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)#useless bruh
#from sklearn import cross_validation  <<--- this was updated
from sklearn import model_selection
# value of K is 10.
splits = model_selection.KFold( n_splits=10, random_state=8, shuffle=True)
print(splits)
#the above code makes indexes for n kfolds
results = []
for train_index, val_index in splits.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    model = LogisticRegression(max_iter=1000000)
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    #aggregate all values after this 
    results.append(np.mean(y_val!=val_predictions))

print(np.mean(results))



#LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, 
#intercept_scaling=1, l1_ratio=None, max_iter=100,  
#multi_class='warn', n_jobs=None, penalty='l2',  
#random_state=0, solver='warn', tol=0.0001, verbose=0, warm_start=False)  



## use L2 regularisation with LGR
model = LogisticRegression(penalty='l2', solver='liblinear', C=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)  # Replace X_val with your validation data

#Compute Metrics:
#Evaluate the model’s performance using appropriate metrics (e.g., accuracy, precision, recall, F1-score).
#You can also access the model’s coefficients (weights) using model.coef_.
#Hyperparameter Tuning (Optional):
#Adjust the regularization strength by modifying the C parameter.
#Smaller C values increase regularization (stronger penalty), while larger values reduce it.



from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler # use in SVM, KNN etc

#it standardizes all features to a unit vairance and removes the means(zero mean), ie all features lie on the same scale now
#many algorithms require data like this
scaler = StandardScaler()
scaled_features= scaler.fit_transform(X)
X_train = scaler.fit_transform(X_train) #it uses the same saved transformations again on the new test values
#it saves the meu(mean) and the sigma(standard deviation) for each feature which can be accessed/used using transform

#Normalizing data is a crucial preprocessing step in machine learning. It involves rescaling the features of your dataset so that they fall within a specific range.
#By doing this, you make the data more suitable for training machine learning models. 
#Machine learning algorithms often perform better when features are on a smaller scale(ie no one feature dominates others, all contribute equally)
#IT IS DIFFERENT FROM STANDARDSCALAR AS IT PUTS ALL DATA IN THE RANGE 0-1
#it effectively handles outliers
#it considers the entire sample as it works on rows
normalized_arr = normalize([np.array(X)])

# scales features(columnwise only, unlike normalize) to a specified range (by default between 0 and 1)
#formula daikhlo by hovering mouse over the function
#it does not handle outliers
#it does not consider the whole sample and only works on columns
#it is a regular scaledown
mms = MinMaxScaler()
scaled_data= mms.fit_transform(X)



#to turn values/classnames into unique integers, you can do both:
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder

iris = sns.load_dataset('iris')

#this is BS, it basically gives you a column each unique val and all the indexes for each and tells True if it was that value in the row
#if it wasn't it will tell false
# if you keep the drop_first= True it gives you one less column than all the unique vals
mylabl1 = pd.get_dummies(iris["species"])

#this is what we're looking for, each unique val is assigned a number
mylabl2 = LabelEncoder().fit_transform(iris["species"])
iris['species']= iris['species'].map({'setosa':0,'versicolor':1,'virginica':2})



#use these to time algo by putting at start and end of program 
import time
start_time = time.time()
end_time = time.time()
print( "Run Time: %.3f" % (end_time - start_time) )



#usage if cross_validate
from sklearn.model_selection import cross_validate
from sklearn.datasets import load_iris

# Load the Iris dataset
X, y = load_iris(return_X_y=True)

# Create a logistic regression model
model = LogisticRegression()
# Perform 5-fold cross-validation
cv_results = cross_validate(model, X, y, cv=5, scoring='accuracy')
# Extract relevant information
mean_accuracy = np.mean(cv_results['test_score'])
fit_times = cv_results['fit_time']
score_times = cv_results['score_time']

print(f"Mean accuracy: {mean_accuracy:.4f}")
print(f"Fit times (seconds): {fit_times}")
print(f"Score times (seconds): {score_times}")



