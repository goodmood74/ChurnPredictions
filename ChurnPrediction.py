# MACHINE LEARNING MODELS COMPARISON ON A CHURN DATASET
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

pd.options.mode.chained_assignment = None  # default='warn'

# Create dataframe
DIR = "C:/Users/podte/PycharmProjects/kaggle/churn"
train_data = pd.read_csv(DIR + "/train.csv")
test_data = pd.read_csv(DIR + "/test.csv")


# Data preparation and cleaning
def replace_with_boolean(dataset, column):
    # Function to replace 'yes' and 'no' values with 1 and 0
    new_column = dataset[column].replace({"no": 0, "yes": 1})
    dataset[column] = new_column
    return dataset


columns = ['churn', 'international_plan', 'voice_mail_plan']
for x in columns: train_data = replace_with_boolean(train_data, columns)


# It seems a little bit too complicated, but I've created this function to "automate" the task in the future.
def create_new_column(dataset, columns_to_drop, column_name):
    # Function to concatenate columns to one new column
    train_data[column_name] = dataset[columns_to_drop].sum(axis=1)
    train_data.drop(columns=columns_to_drop, inplace=True)
    return dataset


columns_to_drop = [['total_day_minutes', 'total_eve_minutes', 'total_night_minutes'],
                   ['total_day_charge', 'total_eve_charge', 'total_night_charge'],
                   ['total_day_calls', 'total_eve_calls', 'total_night_calls']]

new_columns = ['total_minutes', 'total_charge', 'total_calls']

for x, columns in enumerate(columns_to_drop): train_data = create_new_column(train_data, columns, new_columns[x])
# train_data = train_data.eval("total_minutes = {}".format("+".join(columns))).drop(columns) # another method

delete_columns = ['state', 'area_code']
train_data = train_data.drop(columns=delete_columns)

# Creating training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(train_data.drop(columns='churn'), train_data.churn, test_size=0.2,
                                                    random_state=42)

# Creating a LogisticRegression model using pipeline.
# Grid search using CV

"""grid = {"C":np.logspace(-3,3,7), "penalty":"l2"}# l1 lasso l2 ridge
logreg = LogisticRegression(max_iter=1000)
logreg_cv = GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(X_train,y_train)

print("Tuned hyperparameters :(best parameters) ", logreg_cv.best_params_)
print("Accuracy:", logreg_cv.best_score_)"""

lgr = make_pipeline(StandardScaler(),
                    LogisticRegression(max_iter=1000, C=1))
lgr.fit(X_train, y_train)

# Making predictions
lgr_pred = lgr.predict(X_test)

# Print metrics
print("Metrics for Logistic Regression model: \n")
print("Accuracy:", metrics.accuracy_score(y_test, lgr_pred))
print("Precision:", metrics.precision_score(y_test, lgr_pred))
print("Recall:", metrics.recall_score(y_test, lgr_pred))
print("MCC:", metrics.matthews_corrcoef(y_test, lgr_pred))

# SVM model
# First trying to see if the data is linearly separable.
lsvc = make_pipeline(StandardScaler(),
                     LinearSVC(max_iter=1500))

lsvc.fit(X_train, y_train)
svm_pred = lsvc.predict(X_test)
print("\nMetrics for Linear Support Vector Machine model: \n")
print("Accuracy:", metrics.accuracy_score(y_test, svm_pred))
print("Precision:", metrics.precision_score(y_test, svm_pred))
print("Recall:", metrics.recall_score(y_test, svm_pred))
print("MCC:", metrics.matthews_corrcoef(y_test, svm_pred))

# MCC score is very low, which shows us, that the model is not the best fit.

svc = make_pipeline(StandardScaler(),
                    SVC(max_iter=1500, kernel='rbf', gamma=0.17, C=1.5))

svc.fit(X_train, y_train)
nlsvm_pred = svc.predict(X_test)

print("\nMetrics for Nonlinear Support Vector Machine model: \n")
print("Accuracy:", metrics.accuracy_score(y_test, nlsvm_pred))
print("Precision:", metrics.precision_score(y_test, nlsvm_pred))
print("Recall:", metrics.recall_score(y_test, nlsvm_pred))
print("MCC:", metrics.matthews_corrcoef(y_test, nlsvm_pred))

# After trying RBF kernel I've received better results.
# Also, by changing C parameter we can change the recall/precision trade off.

# Random Forest model

rfc = make_pipeline(StandardScaler(),
                    RandomForestClassifier(n_estimators=33, max_depth=20))

rfc.fit(X_train, y_train)
rf_pred = rfc.predict(X_test)

print("\nMetrics for Random Forest model: \n")
print("Accuracy:", metrics.accuracy_score(y_test, rf_pred))
print("Precision:", metrics.precision_score(y_test, rf_pred))
print("Recall:", metrics.recall_score(y_test, rf_pred))
print("MCC:", metrics.matthews_corrcoef(y_test, rf_pred))

gnb = make_pipeline(StandardScaler(),
                    GaussianNB())
gnb.fit(X_train, y_train)
gnb_pred = gnb.predict(X_test)

print("\nMetrics for Naive Bayes model: \n")
print("Accuracy:", metrics.accuracy_score(y_test, gnb_pred))
print("Precision:", metrics.precision_score(y_test, gnb_pred))
print("Recall:", metrics.recall_score(y_test, gnb_pred))
print("MCC:", metrics.matthews_corrcoef(y_test, gnb_pred))

# After examining 5 models the best scores I get is 0.976 for accuracy and 0.906 for MCC with Random Forest Classifier.
