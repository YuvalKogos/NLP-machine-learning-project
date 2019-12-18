import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, f1_score, recall_score, precision_score
import pdb
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc



########################################
##This script developed by Yuval Kogos##
########################################

SMOTE_FOR_OVER_SAMPLING = False
DROP_FAILED_WLS = False
NORMALIZE_INPUT_FEATURES = False
WL_INDEX_AS_FEATURE = False
PLOT_PRECISION_RECALL_CURVE = True
LOAD_PRETRAINED_MODEL = False
SAVE_MODEL = True


#Prepare the dataset
train_path = r'W:\Nika\NN_text_class\Doc2Vec_200_features\Num_data_No_Freq_remove_to_TEST\Train_data_vecs.csv'
test_path = r'W:\Nika\NN_text_class\Doc2Vec_200_features\Num_data_No_Freq_remove_to_TEST\Test_data_vecs.csv'

train_dataset = pd.read_csv(train_path)
test_dataset= pd.read_csv(test_path)

#Model's parameters
input_size = 200
hidden_dim_1 = 500
hidden_dim_2 = 50
num_classes = 7
num_epochs = 30
batch_size = 200
learning_rate = 0.001
validation_eval = 2

#trying to apply class weights

criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, df, labels):
        'Initialization'
        self.labels = labels        
        self.df = df

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.df.iloc[index].values, self.labels.iloc[index]
        
    
    
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

    

class OneLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(OneLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        x = F.relu(self.fc1(x))        
        x = self.fc2(x)
        
        return x


class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)  
        self.fc3 = nn.Linear(hidden_size_2, num_classes)  
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


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


def main():
 
    #Train and test set
    X_train,X_val,X_test, Y_train,Y_val,Y_test = split_dataset(train_dataset,test_dataset)
   
    #    training_generator = data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)
    training_set = Dataset(X_train, Y_train)
    training_generator = data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_set = Dataset(X_val, Y_val)
    validation_generator = data.DataLoader(validation_set, batch_size=len(X_val))
    test_set = Dataset(X_test, Y_test)
    test_generator = data.DataLoader(test_set, batch_size=len(X_test))

    if LOAD_PRETRAINED_MODEL:
        net = torch.load('checkpoints/model.pt')
    else:
        input_dim = X_train.shape[1]
        net = TwoLayerNet(input_dim, hidden_dim_1, hidden_dim_2 , num_classes)

    if torch.cuda.is_available():
        net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  



    print('Start trainning...')
    for epoch in range(num_epochs):
        step = 0
        step_start_time = time.time()
        for local_batch, local_labels in training_generator:  
            # Convert torch tensor to Variable
            local_batch, local_labels = local_batch.to(device).float(), local_labels.to(device).long()

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net(local_batch)
            loss = criterion(outputs, local_labels)
            loss.backward()
            optimizer.step()

            if step % 10 == 9:
#                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
#                       %(epoch+1, num_epochs, step + 1, len(training_set)//batch_size, loss.data[0]))
                 print('Epoch [%d/%d], Step [%d/%d], Loss: %.7f' 
                        %(epoch+1, num_epochs, step + 1, len(training_set)//batch_size, loss.item()))
                 print('Time is {}'.format(time.time() - step_start_time))

            step += 1
            step_start_time = time.time()

        if epoch % validation_eval == validation_eval - 1:
            for local_batch, local_labels in validation_generator:
                # Convert torch tensor to Variable
                local_batch, local_labels = local_batch.to(device).float(), local_labels.to(device).long()
                outputs = net(local_batch)
                _, predicted = torch.max(outputs.data, 1)

                ground_truth = local_labels.numpy()
                pred = predicted.numpy()

#            print('Validation classification report:')
#            print(classification_report(ground_truth, pred))

    if SAVE_MODEL:
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(net, 'checkpoints/model.pt')

    
    for local_batch, local_labels in test_generator:
        # Convert torch tensor to Variable
        local_batch, local_labels = local_batch.to(device).float(), local_labels.to(device).long()
        outputs = net(local_batch)
        _, predicted = torch.max(outputs.data, 1)
        
        ground_truth_test = local_labels.numpy()
        pred_test = predicted.numpy()
        
    train_generator_1 = data.DataLoader(training_set, batch_size=len(X_train))
    for local_batch, local_labels in train_generator_1:
        # Convert torch tensor to Variable
        local_batch, local_labels = local_batch.to(device).float(), local_labels.to(device).long()
        outputs = net(local_batch)
        _, predicted = torch.max(outputs.data, 1)
        
        ground_truth_train = local_labels.numpy()
        pred_train = predicted.numpy()
        
    


    #Model accuracy and test results printing
    print('Model accuracy and testing results:')
    print('Test value counts: ', Y_test.value_counts())

    
    print("Test classification report:")
    print(classification_report(ground_truth_test, pred_test))

    
    print('Confusion matrix test : ')
    print(confusion_matrix(Y_test,pred_test))
    
    print('pred train value count : ', len(pred_train))
    print('y train value count : ', len(Y_train))
    
    print('Train Accuracy : ',accuracy_score(Y_train,pred_train))
    print('Confusion matrix train : ',confusion_matrix(Y_train,pred_train))
    
    print('Test Accuracy : ')
    print(accuracy_score(Y_test,pred_test))
    plot_confusion_matrix(Y_test, pred_test, None)
    
    



if __name__ == '__main__':
    main()
