import pandas as pd
import numpy as np 
import random 
import time
import csv
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, f1_score, recall_score, precision_score, precision_recall_curve


########################################
##This script developed by Yuval Kogos##
########################################

filePath = 'C:/Users/device_lab/Desktop/After_Parser/Phase_4_PC3_cyc_1_70.csv'
NUM_ITERATION = 4


#Prepare the dataset
train_path = r'Z:\Nika Yanuka\Nika\Data Science Projects\20_newsgroups\Doc2Vec_100_features\Num_data_No_Freq_remove_to_TEST\Train_data_vecs.csv'
test_path = r'Z:\Nika Yanuka\Nika\Data Science Projects\20_newsgroups\Doc2Vec_100_features\Num_data_No_Freq_remove_to_TEST\Test_data_vecs.csv'

train_dataset = pd.read_csv(train_path)
test_dataset= pd.read_csv(test_path)

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



def train_and_get_accuracy_score(X_train,X_Val,X_test, Y_train,Y_Val,Y_test,params):
    model = lgb.LGBMClassifier(**params)
    print('Start training...')
    model = model.fit(X_train, Y_train, eval_set = [(X_Val, Y_Val)],early_stopping_rounds=None, verbose=20)
    pred_val = model.predict(X_Val)
    pred_test = model.predict(X_test)
    pred_train = model.predict(X_train)
    
    accuracy_test = accuracy_score(Y_test,pred_test)
    accuracy_val = accuracy_score(Y_Val,pred_val)
    accuracy_train = accuracy_score(Y_train,pred_train)
    
    
    return accuracy_test,accuracy_val,accuracy_train

def main():     
    X_train,X_Val,X_test, Y_train,Y_Val,Y_test = split_dataset(train_dataset,test_dataset)
    
    #the params list with all the posabilities to check
    params_for_tuning = {
        'n_estimators': [400,440,480,520,560],
        'learning_rate': [0.01,0.05,0.1,0.15,0.2],
        'max_depth': [5,6,7,8,9],
        'min_child_weight': [1,20,40,60,80,100],
        'colsample_bytree' : [0.01,0.2,0.4,0.6,0.8,0.99],
        'gamma':[0.01,0.2,0.4,0.6,0.8,0.99]
 
    } 
     
     
    params_for_running = {
    'n_estimators':500,
    'max_depth':6, 
    'learning_rate':0.05,
    'min_child_weight':20,
    'gamma':0,
    'subsample':0.6,
    'colsample_bytree':0.9,
    'reg_alpha':0.05,
    #'scale_pos_weight':1,
    'is_unbalance':True,
    'silent':1,
    'objective':'multiclass'
}
    with open('Z:/Nika Yanuka/Nika/Data Science Projects/20_newsgroups/GS_tuning_results_17_oct.csv',mode = 'w') as output_file:
         
         
         csv_writer = csv.writer(output_file)
         csv_writer.writerow(['Iteration','Parameter','best value','accuracy test','accuracy val','accuracy train','params with best value','line status'])
         
         for iteration in range(NUM_ITERATION):
             for parameter in params_for_tuning:     #iterate over every parameter at params_for_tuning
                 
                 max_score = 0
                 max_score_idx = 0
                 
                 print('Checking ', parameter)
                 
                 for i in range(len(params_for_tuning[parameter])): #iterate over every value of the parameter                        
                     #chagne the value of the parameter at the running dict 
                     params_for_running[parameter] = params_for_tuning[parameter][i]
                     print(params_for_running)
                     accuracy_test,accuracy_val,accuracy_train = train_and_get_accuracy_score(X_train,X_Val,X_test, Y_train,Y_Val,Y_test,params_for_running)
                    
                     if accuracy_test > max_score:
                         max_score = accuracy_test
                         max_score_idx = i
    
                     line_tmp = [iteration,parameter,params_for_tuning[parameter][i],accuracy_test,accuracy_val,accuracy_train,params_for_running, 'Decription']
                     csv_writer.writerow(line_tmp)
                     
                 #Change the parameter at the running list to the best one         
                 params_for_running[parameter] = params_for_tuning[parameter][max_score_idx]
                 
                 #recall = (TP) / (TP + FN)
                 #precision = (TP) / (TP+FP)
                 
                 write_line = [iteration,parameter,params_for_tuning[parameter][max_score_idx],max_score,params_for_running,'SUMMERY']
                 csv_writer.writerow(write_line)
                    
                
            
    print('done.')
        
    
    
    
    
if __name__ == "__main__":
    main()