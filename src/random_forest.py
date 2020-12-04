import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score

def default_random_forest(df):
    """Generate a defual Random Forest for DataFrame

    Args:
        df ([DataFrame]): Attrition DataFrame 

    Returns:
        [Tuple]: Tuple with accuracy, precision and recall scores
    """    
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


def random_forest_grid(df):
    """Using CV to find the most optimal hyperparameters for random forest model

    Args:
        df ([DataFrame]): Attrition DataFrame

    Returns:
        [list]: List of best hyperparameters
    """   
    rf = RandomForestClassifier(random_state=42, class_weight = 'balanced')
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions = grid_param, n_iter = 500,
                                cv=5, verbose=2, random_state=42, n_jobs=-1)
    y = df.attrition
    X = df.drop('attrition',axis=1)
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    rf_random.fit(X_train,y_train)
    
    return rf_random.best_params_


if __name__ == '__main__':
    # default_random_forest(df)
    # random_forest_grid(df)