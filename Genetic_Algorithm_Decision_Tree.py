import pandas as pd
import numpy as np 
import random 
import time
import csv
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, f1_score, recall_score, precision_score, precision_recall_curve


########################################
##This script developed by Yuval Kogos##
########################################

LOAD_DATASET = False
MUTATION_RATE = 0.2
NUMBER_OF_SETS_IN_POPULATION = 10
NUM_GENERATIONS = 100


#Prepare the dataset
train_path = r'W:\Nika\NN_text_class\Doc2Vec_100_features\Num_data_No_Freq_remove_to_TEST\Train_data_vecs.csv'
test_path = r'W:\Nika\NN_text_class\Doc2Vec_100_features\Num_data_No_Freq_remove_to_TEST\Test_data_vecs.csv'

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
    

X_train,X_Val,X_test, Y_train,Y_Val,Y_test = split_dataset(train_dataset,test_dataset)

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

class Population:
    def __init__(self,params,mutation_rate):
        #initialize the population which will be represented as a list of ParamSets
        self.population = []
        self.passed_90 = []
        self.fittest_indx_generation = 0
        self.max_fitness_per_generation = 0
        self.confusion_list_per_generation = []
        self.mutation_rate = mutation_rate
        self.lest_gen_fit_scores = []
        self.last_gen_parent_one = 1
        self.last_gen_parent_two = 1

        for i in range(NUMBER_OF_SETS_IN_POPULATION):
            param = ParamSet(params,generate_random=True)
            self.population.append(param)


    
    def get_fittest(self):
    
        fittest_value = 0
        fittest_indx = 0
        
        for i in range(len(self.population)):
            accuracy_test,accuracy_val,accuracy_train = self.population[i].get_fitness()
            if accuracy_test > fittest_value:
                fittest_value = accuracy_test
                fittest_indx = i
        
        
        return self.population[fittest_indx],fittest_value


    def generate_new_population(self,params):
        self.max_fitness_per_generation = 0
        choose_list = self.create_parents_array()
   
        indx_one = random.choice(choose_list)
        indx_two = random.choice(choose_list)
        
        self.last_gen_parent_one= indx_one
        self.last_gen_parent_two = indx_two
        
        parent_one = self.population[indx_one]
        parent_two = self.population[indx_two]
        
        for i in range(len(self.population)):
            child = self.crossover(parent_one, parent_two)
            self.mutation(child,params)
            self.population[i] = child



    def mutation(self,child,params):
        for param_name in child.gene:
            percent = random.uniform(0,1)
            if percent < self.mutation_rate:
                child.gene[param_name] = random.choice(params[param_name])



    def crossover(self,parent_a,parent_b):

        #choose randomly from which parent to take each gene (at the dictionary)
        dict = {}
        for param_name in parent_a.gene:
            chosen_parent = random.randint(0,2)
            if chosen_parent == 0:
                dict[param_name] = parent_a.gene[param_name]
            else:
                dict[param_name] = parent_b.gene[param_name]

        child = ParamSet(dict)
        
        return child
    
    def create_parents_array(self):
        #Create a list that each cell will present the chances of it's match paramset to be chosen
        total_fitness = 0
        fitness_list = [None] * len(self.population)
        for i in range(len(self.population)):
            accuracy_test,accuracy_val,accuracy_train = self.population[i].get_fitness()
            if accuracy_test > self.max_fitness_per_generation:
                self.max_fitness_per_generation = accuracy_test
                self.fittest_indx_generation = i
                self.confusion_list_per_generation = [accuracy_val,accuracy_train]
            if accuracy_test > 0.9:
                self.passed_90.append(self.population[i])
            total_fitness += accuracy_test
            fitness_list[i] = accuracy_test
        
        self.lest_gen_fit_scores = fitness_list
        #Change the values at fitness_list to their probabilty to be chosen
        for i in range(len(self.population)):
            fitness_list[i] = (fitness_list[i] / total_fitness) * 100
        

        choose_list = []
        for i in range(len(fitness_list)):
            times = int(round(fitness_list[i]))
            for k in range(times):
                choose_list.append(i)

        #choose list is a list that each index (of item in population) is appeard in a count of it's chance to be picked
        return choose_list



class ParamSet:
    def __init__(self,genes,generate_random = False):

        if generate_random == True:
            self.gene = self.initialize_random(genes)
        else:
            self.gene = genes


    def initialize_random(self,params):
        #create a dictionary of the parameters and random values
        #params vlaues is a list of lists with the same indices as the params_names
        dict = {}
        for x,y, in params.items():
            value = random.choice(y)
            dict[x] = value

        return dict

    def get_fitness(self):
        accuracy_test,accuracy_val,accuracy_train = train_and_get_accuracy_score(X_train,X_Val,X_test, Y_train,Y_Val,Y_test,self.gene)
        if accuracy_test > 0.9:
            print('accuracy_test is greater then 80%')
            print(self.gene)
        
        return accuracy_test,accuracy_val,accuracy_train




def get_current_generation_description(population,generation,last_gen_pops):
    result_list = [generation,population.fittest_indx_generation,population.confusion_list_per_generation[0],population.confusion_list_per_generation[1],population.max_fitness_per_generation]
    for pop in last_gen_pops:        
        result_list.append(pop)
            
    result_list.extend([population.lest_gen_fit_scores,population.last_gen_parent_one,population.last_gen_parent_two])
    return result_list
        
    
    

def main():

    
    if LOAD_DATASET:
        print('Loading file...')
        filePath = 'W:/Yuval/Phase_4_full_fixed.csv'
        dataset = pd.read_csv(filePath,skiprows = [10])
        dataset = dataset.loc[dataset['Bank'] == 2]
        
        a = ((dataset['BLK'] >= 1330) & (dataset['BLK'] <= 1610))
        a = a.astype('int32')
        dataset.insert(25,'is_Edge_BLK',a)


    params = {
        'n_estimators': [300,400,450,500,550,600,650],
        'learning_rate': [0.001,0.003,0.005,0.01,0.05,0.1,0.15,0.2,0.5],
        'max_depth': [2,3,4,5,6,7,8,9],
        'min_child_weight': [1,20,40,60,80,100],
        'colsample_bytree' : [0.01,0.2,0.4,0.6,0.8,0.99],
        'gamma':[0.01,0.2,0.4,0.6,0.8,0.99],
        'subsample':[0.6],
        'reg_alpha':[0.05],
        'is_unbalance':[True],
        'silent':[1],
        'objective':['multiclass']
 
    } 

    population = Population(params,MUTATION_RATE)
    input('Press enter to continue...')
    with open('W:/Yuval/GA_tuning_results.csv',mode = 'w') as output_file:
        csv_writer = csv.writer(output_file)
        output_headlines = ['Generation','Best_child_index','Accurcay_train','Accuracy_val','Accuracy_test']
    
        for i in range(NUMBER_OF_SETS_IN_POPULATION):
            output_headlines.extend(['Eta','max_depth','min_child_weight','gamma','subsample','alpha','scale_weight','n_estimators'])
    
        output_headlines.extend(['Fitness_Scores_list','Parent one', 'Parent_two'])
        csv_writer.writerow(output_headlines)
    
    
        print('initial population:')
    
        for pop in population.population:
            print(pop.gene)

        for i in range(NUM_GENERATIONS):
            print('Generation: ',i)
        
            last_gen_pop_data = []
        
            #Record the data from the previous generation beofre creating a new one
            for child in population.population:
                for x,y, in child.gene.items():
                    last_gen_pop_data.append(str(y))

                    #create a new population
            population.generate_new_population(params)
        
            #get and print the data from the previous generation 
            #(because the estimation happend at new gen creation process)
            output_list = get_current_generation_description(population,i,last_gen_pop_data)
            csv_writer.writerow(output_list)
        
    

        
    fittest_gene,f1_value = population.get_fittest()
        
    print('The best params set : ')
    print(fittest_gene.gene)
    
    print('At score of : ')
    print(f1_value)
    
    
    if len(population.passed_90) > 0:
        print('Passed 0.9 f1:')
        for child in population.passed_90:
            print(child.gene)
    
    
    
    print("done.")


if __name__ == "__main__":
    main()