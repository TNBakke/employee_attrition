import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.preprocessing import StandardScaler 
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

def logistic_regression_class_weight(class_weight, lasso_features):
    """Performs logistic regression on attrition DataFrame with list of
    desired features (i.e. from lasso regression analysis)

    Args:
        class_weight ([dict]): Dictionary for the desired class
        weight (i.e. w = {0:10, 1:90})
        lasso_features ([list]): List of features you want included

    Returns:
        [Tuple]: Confusion Matrix results from logistic regression
    """    
    df = pd.read_csv("./data/clean_one_hot_data.csv")
    df.drop(df.columns[0], axis=1,inplace=True)
    
    scaler = StandardScaler()
    scaler.fit_transform(df)
    
    y = df['attrition']
    X = df.drop(columns=['attrition'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    logreg = LogisticRegression(class_weight=w)
    logreg.fit(X_train, y_train)
    
    y_pred = logreg.predict(X_test)

    con_matrix = confusion_matrix(y_test, y_pred).ravel()
    
    return con_matrix

if __name__ == '__main__':
    # logistic_regression_class_weight(class_weight, lasso_features)