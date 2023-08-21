import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


#################################### creating x, y subsets #######################################################################

def x_y_split(train, val, test):

    # Split each set into features (X) and target (y)
    X_train, y_train = train.drop(columns='churn'), train['churn']
    X_val, y_val = val.drop(columns='churn'), val['churn']
    X_test, y_test = test.drop(columns='churn'), test['churn']

    return X_train, y_train, X_val, y_val, X_test, y_test


#################################### decision tree #######################################################################

def train_decision_tree(X_train, y_train, X_val, y_val, max_depth=3, random_state=42):
    # Exclude non-numeric columns from the training and validation sets
    numeric_columns = X_train.select_dtypes(include=['number']).columns.tolist()
    X_train = X_train[numeric_columns]
    X_val = X_val[numeric_columns]

    # Create a Decision Tree model
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    
    # Fit the model on the training data
    clf.fit(X_train, y_train)
    
    # Calculate feature importances
    fi = pd.DataFrame({'Feature': X_train.columns, 'Importance': clf.feature_importances_})

    # Get the top 3 important features
    mfi = fi.sort_values(by='Importance', ascending=False).head(3)

    # Calculate accuracy on training and validation data
    train_accuracy = clf.score(X_train, y_train)
    val_accuracy = clf.score(X_val, y_val)
    
    # Print accuracy statements
    print(f"Accuracy of Decision Tree on train data is {round(train_accuracy, 4)}")
    print(f"Accuracy of Decision Tree on validate data is {round(val_accuracy, 4)}")

    # Return the trained model
    return mfi




#################################### random forest #######################################################################

def train_random_forest(X_train, y_train, X_val, y_val):
    # Exclude non-numeric columns from the training and validation sets
    numeric_columns = X_train.select_dtypes(include=['number']).columns.tolist()
    X_train = X_train[numeric_columns]
    X_val = X_val[numeric_columns]
    
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

def train_logistic_regression(X_train, y_train, X_val, y_val):
    # Exclude non-numeric columns from the training and validation sets
    numeric_columns = X_train.select_dtypes(include=['number']).columns.tolist()
    X_train = X_train[numeric_columns].copy()
    X_val = X_val[numeric_columns].copy()
    
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
    
    # Calculate accuracy on training and test data
    train_accuracy = knn.score(X_train, y_train)
    test_accuracy = knn.score(X_test, y_test)
   
    
    # Print accuracy statement
    print(f"Accuracy of KNN on test data is {round(test_accuracy, 4)}")


#################################### knn_predictions #######################################################################

def generate_knn_predictions(train, test, n_neighbors, output_csv_filename):
    # Select the specified columns
    columns = ['customer_id', 'tenure', 'monthly_charges', 'total_charges', 'contract_type', 'internet_service_type']
    X_train = train[columns].copy()
    X_train.drop(columns=['customer_id'], inplace=True)
    y_train = train['churn']

    X_test = test[columns].copy()
    X_test.drop(columns=['customer_id'], inplace=True)

    # Apply Min-Max scaling to the selected columns
    mms = MinMaxScaler()
    X_train[['tenure', 'monthly_charges', 'total_charges']] = mms.fit_transform(X_train[['tenure', 'monthly_charges', 'total_charges']])
    X_test[['tenure', 'monthly_charges', 'total_charges']] = mms.transform(X_test[['tenure', 'monthly_charges', 'total_charges']])

    # Create a KNN model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit the model on the training data
    knn.fit(X_train, y_train)

    # Predict probabilities and labels on the test set
    y_proba = knn.predict_proba(X_test)[:, 1]  # Probability of churn
    y_pred = knn.predict(X_test)  # Predicted labels

    # Assuming you have the customer_id column in your X_test DataFrame
    customer_ids = test['customer_id'].reset_index(drop=True)

    # Create a DataFrame for predictions
    predictions_df = pd.DataFrame({
        'customer_id': customer_ids,
        'probability_of_churn': y_proba,
        'prediction_of_churn': y_pred
    })

    # Map the binary predictions (1=churn, 0=not_churn)
    predictions_df['prediction_of_churn'] = predictions_df['prediction_of_churn'].astype(int)

    # Save the predictions to a CSV file
    predictions_df.to_csv(output_csv_filename, index=False)




