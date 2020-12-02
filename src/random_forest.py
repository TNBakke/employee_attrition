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


def grid_search_wrapper(refit_score='precision_score'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(rf, param_grid, scoring=scorers, refit=refit_score,
                           cv=skf, return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train.values, y_train.values)

    # make the predictions
    y_pred = grid_search.predict(X_test.values)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    return grid_search


if __name__ == '__main__':
    # default_random_forest(df)
    