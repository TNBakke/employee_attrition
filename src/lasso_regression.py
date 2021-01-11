import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler 

def lasso_reg(alpha):
    """Computes R^2 metrics for training and test set on Attrition df
    Args:
        alpha ([float]): Select desired alpha for lasso regression

    Returns:
        [Tuple]: Tuple of R^2 scores for training and test set 
    """   
    df = pd.read_csv("./data/clean_one_hot_data.csv")
    df.drop(df.columns[0], axis=1,inplace=True)
    
    scaler = StandardScaler()
    scaler.fit_transform(df)
    
    y = df['attrition']
    X = df.drop(columns=['attrition'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    reg = Lasso(alpha=alpha, random_state=42)
    reg.fit(X_train, y_train)
    
    return f'Lasso Regression: R^2 score on training set', reg.score(X_train, y_train)*100, 
    f'Lasso Regression: R^2 score on test set', reg.score(X_test, y_test)*100
    

    
if __name__ == '__main__':
    lasso_reg(0.5):