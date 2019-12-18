import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from copy import deepcopy
#import xgboost as xgb
from sklearn import model_selection
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, f1_score, recall_score, precision_score, precision_recall_curve
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.utils.fixes import signature
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import class_weight
from sklearn import metrics

from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels



params = {
    'n_estimators':480,
    'max_depth':6, 
    'num_leaves': 150,
    'learning_rate':0.15,
    'min_child_weight':20,
    'lambda_l2':0.05,
    'subsample':0.6,
    'colsample_bytree':0.1,
    'reg_alpha':0.05,
    #'scale_pos_weight':1,
    'is_unbalance':True,
    'silent':1,
    'objective':'multiclass'
}



#Prepare the dataset
train_path = r'Z:\Nika Yanuka\Nika\Data Science Projects\20_newsgroups\Doc2Vec_200_features\Num_data_No_Freq_remove_to_TEST\Train_data_vecs.csv'
test_path = r'Z:\Nika Yanuka\Nika\Data Science Projects\20_newsgroups\Doc2Vec_200_features\Num_data_No_Freq_remove_to_TEST\Test_data_vecs.csv'

train_dataset = pd.read_csv(train_path)
test_dataset= pd.read_csv(test_path)



# Plot Confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           #xticklabels=[], yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax





def convert_to_true_lables(Y):
    #Convert the name labels to numeric lables
    Values_dict = {'1': ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x'],
            '2': ['rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey'],
            '3': ['sci.crypt','sci.electronics','sci.med','sci.space'],
            '4': ['misc.forsale'],
            '5': ['talk.politics.misc','talk.politics.guns','talk.politics.mideast'],
            '6': ['talk.religion.misc','alt.atheism','soc.religion.christian']}
    
    
    new_labels = []
    for label in Y:
        for key in Values_dict:
            if label in Values_dict[key]:
                new_labels.append(int(key))
                
    return new_labels

def recreate_labels(y):
    #Sort labels
    a = np.unique(y)
    
    new_labels = []
    for label in y:
        new_labels.append(a.index(label))
        
    return new_labels
        
def split_dataset(train_dataset,test_dataset):
    X_train = train_dataset.drop(columns = ['File name','Label'])
    Y_train = train_dataset['Label']
    
    X_test = test_dataset.drop(columns = ['File name','Label'])
    Y_test = test_dataset['Label']
    
    X_test,X_train1,Y_test,Y_train1 = train_test_split(X_test,Y_test,test_size = 0.25,random_state = 1)
    X_train.append(X_train1) 
    Y_train.append(Y_train1)
    
    
    X_test,X_Val,Y_test,Y_Val = train_test_split(X_test,Y_test,test_size = 0.5,random_state = 1)

    return X_train,X_Val,X_test, Y_train,Y_Val,Y_test



def plot_roc_curve(y_test,y_score):
    
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
    
            
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

def main():
    
    
    #Train and test set
    X_train,X_Val,X_test, Y_train,Y_Val,Y_test = split_dataset(train_dataset,test_dataset)
    
    
    #Building and training the model
    model = lgb.LGBMClassifier(**params)
    print('Start training...')
    model = model.fit(X_train, Y_train, eval_set = [(X_Val, Y_Val)],early_stopping_rounds=None, verbose=20)
    model.booster_.save_model('W:\Yuval\mode.txt')
    
    
    #Predictions and eveluations
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_Val)
    pred_test = model.predict(X_test)
    Yscore = model.predict(X_test,raw_score=True)
    

    #Output - model results
    print('Confusion MATRIX train')
    plot_confusion_matrix(Y_train, pred_train, ['1','2','3','4','5','6'])
    
    print('Confusion MATRIX test')
    plot_confusion_matrix(Y_test, pred_test, ['1','2','3','4','5','6'])
    
    
    print('Accuracy Train : ')
    print(accuracy_score(Y_train,pred_train))
    print('Accuracy Test : ')
    print(accuracy_score(Y_test,pred_test))
    
    print('Done.')
    
    
    
    
if __name__ == "__main__":
    main()