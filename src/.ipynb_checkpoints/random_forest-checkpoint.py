import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score

def default_random_forest(df):
    y = df.pop('attrition').values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    y_preds = rf.predict(X_test)
    
    accuracy_score_ = accuracy_score(y_test, y_preds)
    precision_score_ = precision_score(y_test, y_preds)
    recall_score_ = recall_score(y_test, y_preds)
    
    return f'Accuracy Score: {accuracy_score_}', f'Precision Score: {precision_score_}', f'Recall Score: {recall_score_}'


def smote_default_random_forest(df):
    y = df.pop('attrition').values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
    smt = SMOTE(random_state=42)
    X_train_SMOTE, y_train_SMOTE = smt.fit_sample(X_train, y_train)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_SMOTE, y_train_SMOTE)
    y_preds = rf.predict(X_test)
    
    accuracy_score_ = accuracy_score(y_test, y_preds)
    precision_score_ = precision_score(y_test, y_preds)
    recall_score_ = recall_score(y_test, y_preds)
    
    return f'Accuracy Score: {accuracy_score_}', f'Precision Score: {precision_score_}', f'Recall Score: {recall_score_}'

def smote_1000_tree_random_forest(df):
    y = df.pop('attrition').values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
    smt = SMOTE(random_state=42)
    X_train_SMOTE, y_train_SMOTE = smt.fit_sample(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=1000,random_state=42)
    rf.fit(X_train_SMOTE, y_train_SMOTE)
    y_preds = rf.predict(X_test)
    
    accuracy_score_ = accuracy_score(y_test, y_preds)
    precision_score_ = precision_score(y_test, y_preds)
    recall_score_ = recall_score(y_test, y_preds)
    
    return f'Accuracy Score: {accuracy_score_}', f'Precision Score: {precision_score_}', f'Recall Score: {recall_score_}'
    
if __name__ == '__main__':
    # default_random_forest(df)
    