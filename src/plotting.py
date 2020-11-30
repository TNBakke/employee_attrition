def heat_map(df):
    plt.figure(figsize=(30, 30))
    sns.heatmap(df.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":15})
    
def feature_hist(df):
    df.hist(linewidth=1.0,figsize=(20,20))