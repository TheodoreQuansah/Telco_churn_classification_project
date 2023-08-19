import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from scipy import stats
from env import get_connection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

#################################### cleaning data columns #######################################################################

def encode_columns(train, val, test):
    # One-hot encoding categorical columns with get dummies
    train = pd.get_dummies(train, columns=[
        'gender', 'senior_citizen', 'partner', 'dependents', 'phone_service',
        'multiple_lines', 'online_security', 'online_backup', 'device_protection',
        'tech_support', 'streaming_tv', 'streaming_movies', 'payment_type',
        'contract_type', 'internet_service_type', 'paperless_billing'
    ], drop_first=True)
    
    val = pd.get_dummies(val, columns=[
        'gender', 'senior_citizen', 'partner', 'dependents', 'phone_service',
        'multiple_lines', 'online_security', 'online_backup', 'device_protection',
        'tech_support', 'streaming_tv', 'streaming_movies', 'payment_type',
        'contract_type', 'internet_service_type', 'paperless_billing'
    ], drop_first=True)
    
    test = pd.get_dummies(test, columns=[
        'gender', 'senior_citizen', 'partner', 'dependents', 'phone_service',
        'multiple_lines', 'online_security', 'online_backup', 'device_protection',
        'tech_support', 'streaming_tv', 'streaming_movies', 'payment_type',
        'contract_type', 'internet_service_type', 'paperless_billing'
    ], drop_first=True)

    # Drop columns with 'No internet service' and 'internet_service_type_None'
    cols_to_drop = train.columns[train.columns.str.contains('No internet service')]
    train = train.drop(columns=cols_to_drop)
    train = train.drop(columns=['internet_service_type_None'])
    train = train.rename(columns={'gender_male': 'gender'})

    cols_to_drop = val.columns[val.columns.str.contains('No internet service')]
    val = val.drop(columns=cols_to_drop)
    val = val.drop(columns=['internet_service_type_None'])
    val = val.rename(columns={'gender_male': 'gender'})

    cols_to_drop = test.columns[test.columns.str.contains('No internet service')]
    test = test.drop(columns=cols_to_drop)
    test = test.drop(columns=['internet_service_type_None'])
    test = test.rename(columns={'gender_male': 'gender'})

    # Rename columns by removing "_Yes"
    columns_to_rename = [
        "partner_Yes", "dependents_Yes", "phone_service_Yes", 
        "multiple_lines_Yes", "online_security_Yes", "online_backup_Yes", 
        "device_protection_Yes", "tech_support_Yes", "streaming_tv_Yes", 
        "streaming_movies_Yes", "paperless_billing_Yes"
    ]
    for col in columns_to_rename:
        new_name = col.replace("_Yes", "")
        train.rename(columns={col: new_name}, inplace=True)
        val.rename(columns={col: new_name}, inplace=True)
        test.rename(columns={col: new_name}, inplace=True)

    # Rename columns based on provided replacements
    columns_to_rename = [
        "senior_citizen_1", "payment_type_Manual Payment", 
        "contract_type_one/two-years", "internet_service_type_Fiber optic"
    ]
    for col in columns_to_rename:
        new_name = col.replace("senior_citizen_1", "senior_citizen").replace("payment_type_Manual Payment",
                                                                             "payment_type").replace("contract_type_one/two-years", 
                                                                             "contract_type").replace("internet_service_type_Fiber optic", 
                                                                             "internet_service_type")
        train.rename(columns={col: new_name}, inplace=True)
        val.rename(columns={col: new_name}, inplace=True)
        test.rename(columns={col: new_name}, inplace=True)
    
    return train, val, test







#################################### creating x, y subsets #######################################################################

def x_y_split(train, val, test):

    # Split each set into features (X) and target (y)
    X_train, y_train = train.drop(columns='churn'), train['churn']
    X_val, y_val = val.drop(columns='churn'), val['churn']
    X_test, y_test = test.drop(columns='churn'), test['churn']

    return X_train, y_train, X_val, y_val, X_test, y_test






#################################### decision tree #######################################################################

def train_decision_tree(X_train, y_train, X_val, y_val, max_depth=3, random_state=42):
    # Create a Decision Tree model
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    # Fit the model on the training data
    clf.fit(X_train, y_train)
    
    # Calculate feature importances
    fi = pd.DataFrame({'Feature': X_train.columns, 'Importance': clf.feature_importances_})
    fi = fi.sort_values(by='Importance', ascending=False).head(3)
    
    # Calculate accuracy on training and validation data
    train_accuracy = clf.score(X_train, y_train)
    val_accuracy = clf.score(X_val, y_val)
    
    # Print accuracy statements
    print(f"Accuracy of Decision Tree on train data is {round(train_accuracy, 4)}")
    print(f"Accuracy of Decision Tree on validate data is {round(val_accuracy, 4)}")
    
    # Return only the feature importance DataFrame
    return fi



#################################### random forest #######################################################################

def train_random_forest(X_train, y_train, X_val, y_val):
    # Create a Random Forest model
    rf = RandomForestClassifier(min_samples_leaf = 7, max_depth=8, random_state=42)
    # Fit the model on the training data
    rf.fit(X_train, y_train)
    
    # Calculate feature importances
    fi = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf.feature_importances_})
    fi = fi.sort_values(by='Importance', ascending=False).head(3)
    
    # Calculate accuracy on training and validation data
    train_accuracy = rf.score(X_train, y_train)
    val_accuracy = rf.score(X_val, y_val)
    
    # Print accuracy statements
    print(f"Accuracy of Random Forest on train data is {round(train_accuracy, 4)}")
    print(f"Accuracy of Random Forest on validate data is {round(val_accuracy, 4)}")
    
    # Return only the feature importance DataFrame
    return fi



#################################### KNeighbors #######################################################################

def train_knn(train, val, n_neighbors=30):

    columns = ['tenure', 'monthly_charges', 'total_charges', 'contract_type', 'internet_service_type']
    # Select the specified columns
    X_train = train[columns].copy()
    y_train = train['churn']
    
    X_val = val[columns].copy()
    y_val = val['churn']
    
    # Apply Min-Max scaling to the selected columns
    mms = MinMaxScaler()
    X_train[['tenure', 'monthly_charges', 'total_charges']] = mms.fit_transform(X_train[['tenure', 'monthly_charges', 'total_charges']])
    X_val[['tenure', 'monthly_charges', 'total_charges']] = mms.transform(X_val[['tenure', 'monthly_charges', 'total_charges']])
    
    # Create a KNN model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Fit the model on the training data
    knn.fit(X_train, y_train)
    
    # Calculate accuracy on training and validation data
    train_accuracy = knn.score(X_train, y_train)
    val_accuracy = knn.score(X_val, y_val)
    
    # Print accuracy statements
    print(f"Accuracy of KNN on train data is {round(train_accuracy, 4)}")
    print(f"Accuracy of KNN on validate data is {round(val_accuracy, 4)}")




#################################### LogisticRegression #######################################################################

def train_logistic_regression(train, val):
    # Separate features and target
    X_train = train.drop(columns=['churn'])
    y_train = train['churn']

    X_val = val.drop(columns=['churn'])
    y_val = val['churn']

    # Apply Min-Max scaling to the selected columns
    mms = MinMaxScaler()
    X_train[['tenure', 'monthly_charges', 'total_charges']] = mms.fit_transform(X_train[['tenure', 'monthly_charges', 'total_charges']])
    X_val[['tenure', 'monthly_charges', 'total_charges']] = mms.transform(X_val[['tenure', 'monthly_charges', 'total_charges']])

    # Create a Logistic Regression model
    log_reg = LogisticRegression(random_state=42)

    # Fit the model on the training data
    log_reg.fit(X_train, y_train)

    # Calculate accuracy on training and validation data
    train_accuracy = log_reg.score(X_train, y_train)
    val_accuracy = log_reg.score(X_val, y_val)

    # Print accuracy statements
    print(f"Accuracy of Logistic Regression on train data is {round(train_accuracy, 4)}")
    print(f"Accuracy of Logistic Regression on validate data is {round(val_accuracy, 4)}")





#################################### KNeighbors model evaluation on test #######################################################################

def test_knn(train, test, n_neighbors=30):

    columns = ['tenure', 'monthly_charges', 'total_charges', 'contract_type', 'internet_service_type']
    # Select the specified columns
    X_train = train[columns].copy()
    y_train = train['churn']
    
    X_test = test[columns].copy()
    y_test = test['churn']
    
    # Apply Min-Max scaling to the selected columns
    mms = MinMaxScaler()
    X_train[['tenure', 'monthly_charges', 'total_charges']] = mms.fit_transform(X_train[['tenure', 'monthly_charges', 'total_charges']])
    X_test[['tenure', 'monthly_charges', 'total_charges']] = mms.transform(X_test[['tenure', 'monthly_charges', 'total_charges']])
    
    # Create a KNN model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Fit the model on the training data
    knn.fit(X_train, y_train)
    
    # Calculate accuracy on training and validation data
    train_accuracy = knn.score(X_train, y_train)
    test_accuracy = knn.score(X_test, y_test)
    
    # Print accuracy statements
    print(f"Accuracy of KNN on test data is {round(test_accuracy, 4)}")