import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
plt.rc("font", size=16)

from sklearn.preprocessing import StandardScaler 
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

def logistic_regression(file_path, class_weight=w):
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