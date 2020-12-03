import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression

plt.rc("font", size=16)

def heat_map(df):
    """Plots heat map for your data/features

    Args:
        df ([DataFrame]): DataFrame you would like plotted
    """    
    
    plt.figure(figsize=(30, 30))
    sns.heatmap(df.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":15})
    
def feature_hist(df):
    """Plots histogram for all non categorical features in DataFrame

    Args:
        df ([DataFrame]): DataFrame you would like plotted
    """    
    df.hist(linewidth=1.0,figsize=(20,20))
    
def attrition_by_feature_plot(df, feature):
    """Plots quick plot to show target variable vs. feature

    Args:
        df ([DataFrame]): DataFrame you want plotted
        feature ([str]): Column name you want plotted against target
    """    
    pd.crosstab(df.feature,df.Attrition).plot(kind='bar', figsize=(15,10))
    plt.title(f'Attrition by {feature}')
    plt.xlabel(f'{feature}')
    plt.ylabel('Employee Count')
    plt.xticks(rotation=45)
    plt.show()

def select_feature_hist_plot(features):
    """Plots histograms of the list of features you want plotted

    Args:
        features ([list]): List of features that you want plotted
    """
    plt.figure(figsize=(20, 10))

    for i, column in enumerate(features, 1):
        plt.subplot(2, 4, i)
        df[df["Attrition"] == 'No'][column].hist(bins=35, color='blue', label='Attrition = NO', alpha=0.6)
        df[df["Attrition"] == 'Yes'][column].hist(bins=35, color='red', label='Attrition = YES', alpha=0.6)
        plt.legend()
        plt.xlabel(column)
        plt.show()
    
def roc_plot(X_test, y_test):
    """[summary]

    Args:
        X_test ([type]): [description]
        y_test ([type]): [description]
    """    
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()

def lasso_eval_lambda_plot(): 
    """[summary]
    """    
    lambdas = (0.001, 0.01, 0.1, 0.5, 1, 2, 10)
    l_num = 7
    pred_num = X.shape[1]

    # prepare data for enumerate
    coeff_a = np.zeros((l_num, pred_num))
    train_r_squared = np.zeros(l_num)
    test_r_squared = np.zeros(l_num) 

    
    for ind, i in enumerate(lambdas):    
        reg = Lasso(alpha = i)
        reg.fit(X_train, y_train)

        coeff_a[ind,:] = reg.coef_
        train_r_squared[ind] = reg.score(X_train, y_train)
        test_r_squared[ind] = reg.score(X_test, y_test)
    
    plt.figure(figsize=(18, 8))
    plt.plot(train_r_squared, 'bo-', label=r'$R^2$ Training set', color="darkblue", alpha=0.6, linewidth=3)
    plt.plot(test_r_squared, 'bo-', label=r'$R^2$ Test set', color="darkred", alpha=0.6, linewidth=3)
    plt.xlabel('Lamda index'); plt.ylabel(r'$R^2$')
    plt.xlim(0, 6)
    plt.title(r'Evaluate lasso regression with lamdas: 0 = 0.001, 1= 0.01, 2 = 0.1, 3 = 0.5, 4= 1, 5= 2, 6 = 10')
    plt.legend(loc='best')
    plt.grid()    

if __name__ == '__main__':
    # heat_map(df)
    # feature_hist(df)
    # count_plot(df)
    # attrition_by_feature_plot(df,feature)
    # roc_plot(X_test, y_test)
    