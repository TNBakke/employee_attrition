import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

def knn_model_std_scaler(df):
    scaler = StandardScaler()
    
    # Compiling all features to have standard scale
    
    features = [['age', 'daily_rate', 'distance_from_home', 'hourly_rate', 'monthly_income', 'monthly_rate', 'percent_salary_hike', 
             'total_working_years', 'training_times_last_year', 'years_at_company', 'years_in_current_role', 
             'years_since_last_promotion', 'years_with_curr_manager']]
    
    for feature in features:
        df[feature] = scaler.fit_transform(df[feature])
    
    # Rule of Thumb: k = square root of df rows of 1470 which equals ~38
    knn = KNeighborsClassifier(n_neighbors=38)
    
    y = df['attrition']
    X = df.drop(columns=['attrition'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    accuracy_score_ = accuracy_score(y_test, y_pred))
    roc_auc_score_ = roc_auc_score(y_test, y_pred)
    
    return f'Accuracy Score: {accuracy_score_}', f'ROC_AUC_Score: {roc_auc_score_}
    
def knn_model_tuning(leaf_size_list, n_neighbors_list, p=[1,2] ):
    # convert to dictionary
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

    # Making model
    clf = GridSearchCV(knn, hyperparameters, cv=10)
    best_model = clf.fit(X_train,y_train)

    #Best Hyperparameters Value
    print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
    print('Best p:', best_model.best_estimator_.get_params()['p'])
    print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

    #Predict testing set
    y_pred = best_model.predict(X_test)

    #Check performance using accuracy
    print(accuracy_score(y_test, y_pred))

    #Check performance using ROC
    roc_auc_score(y_test, y_pred)
    
    
if __name__ == '__main__':
    # knn_model_std_scaler(df)