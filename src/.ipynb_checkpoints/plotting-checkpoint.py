import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
plt.rc("font", size=16)

def heat_map(df):
    plt.figure(figsize=(30, 30))
    sns.heatmap(df.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":15})
    
def feature_hist(df):
    df.hist(linewidth=1.0,figsize=(20,20))
    
def count_plot(df):
    sns.countplot(x='Attrition', data=df)
    plt.show()
    
def attrition_by_feature_plot(df, feature):
    pd.crosstab(df.feature,df.Attrition).plot(kind='bar', figsize=(15,10))
    plt.title(f'Attrition by {feature}')
    plt.xlabel(f'{feature}')
    plt.ylabel('Employee Count')
    plt.xticks(rotation=45)
    plt.show()
    
def roc_plot(X_test, y_test):
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
    
    
if __name__ == '__main__':
    # heat_map(df)
    # feature_hist(df)
    # count_plot(df)
    # attrition_by_feature_plot(df,feature)
    # roc_plot(X_test, y_test)
    