import os
import pandas as pd

from sklearn.model_selection import train_test_split
from env import get_connection
from sklearn.preprocessing import LabelEncoder

def encode_categorical_columns(data_frame, categorical_columns):
    
    categorical_columns = ["gender", "partner", "dependents", "phone_service", "multiple_lines", "online_security",
                           
                       "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies",
                           
                       "contract_type", "internet_service_type", "payment_type", "churn"]
    
    label_encoder = LabelEncoder()
    
    for col in categorical_columns:
        
        data_frame[col] = label_encoder.fit_transform(data_frame[col])
        
    return data_frame


def train_val_test(df, strat, seed=42):
    
    train, val_test = train_test_split(df, train_size=0.7, random_state=seed, stratify=df[strat])
    
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed, stratify=val_test[strat])
    
    return train, val, test


def dummies(df):

    df = pd.get_dummies(df, columns = ['sex'], drop_first = True)
    
    df = pd.get_dummies(df, columns = ['class', 'embark_town'])
    
    df = pd.get_dummies(df)
    
    return df


def theometrics(TP, TN, FP, FN):
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    recall = TP / (TP + FN)
    
    true_positive_rate = TP / (TP + FN)
    
    false_positive_rate = FP / (FP + TN)
    
    true_negative_rate = TN / (TN + FP)
    
    false_negative_rate = FN / (FN + TP)
    
    precision = TP / (TP + FP)
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    support = TP + FN
    
    data = {
        'metric': ['Accuracy', 'Recall', 'True Positive Rate', 'False Positive Rate', 'True Negative Rate', 'False Negative Rate', 'Precision', 'F1-Score', 'Support'],
        'value': [accuracy, recall, true_positive_rate, false_positive_rate, true_negative_rate, false_negative_rate, precision, f1_score, support]
    }
    
    metrics_df = pd.DataFrame(data)
    
    return metrics_df


