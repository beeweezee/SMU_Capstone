import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, classification_report
import numpy as np
import seaborn as sns

def generate_report(test_df):
    report = dict()
    report['totalSamples'] = len(test_df)
    report['overallAccuracy'] = test_df['match'].sum() / report['totalSamples']
    
    cat_results = dict()
    for cat in test_df['category'].unique():
        cat_results[cat] = {}
        df_cat = test_df[test_df['category'] == cat]
        cat_total_samples = len(df_cat)
        cat_results[cat]['totalSamples'] = cat_total_samples
        cat_results[cat]['accuracy'] = df_cat['match'].sum() / cat_total_samples
    cat_results = dict(sorted(cat_results.items(), key=lambda item: item[1]['accuracy'], reverse=True))
    report['byCategory'] = cat_results
    
    return report

def plot_metrics_charts(report):
    x = 'Category'
    y = []
    for cat in report['category']:
        y.append(report[cat]['accuracy'])
    plt.title('Accuracy by Category')
    plt.xlabel('category')
    plt.ylabel('accuracy')
    plt.bar(x,y)
    #plt.show()
    return plt.show()
 
def plot_roc_curve(labels, predictions, x_lim=[0.0, 1.0], y_lim=[0.0, 1.0], target_names=['clean', 'dirty']):
    
    """
    Function to plot Calculate AUC & Plot ROC
    Args:
    labels: Labels for Test Set
    predictions: Predictions for Test Set
    Returns:
    Saves plots to disk
    """
    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # caculate n_classes
    n_classes = len(np.unique(labels))
    # calculate dummies once
    y_test_dummies = pd.get_dummies(labels, drop_first=False).values
    for i, k in enumerate(target_names):
        fpr[k], tpr[k], _ = roc_curve(y_test_dummies[:, i], predictions[:, i])
        roc_auc[k] = auc(fpr[k], tpr[k])
    
    # roc for each class
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC')
    for i in target_names:
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %s' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    #plt.savefig('roc.png')
    return plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return plt.show()

