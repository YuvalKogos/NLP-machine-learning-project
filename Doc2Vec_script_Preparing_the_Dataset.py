import gensim
import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join
import csv
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from nltk.stem import PorterStemmer
import numpy as np


########################################
#####This script made by Yuval Kogos####
########################################



TRAIN_INPUT_PATH = 'Z:/Nika Yanuka/Nika/Data Science Projects/20_newsgroups/20news_bydate/20news_bydate_train'
TEST_INPUT_PATH =  'Z:/Nika Yanuka/Nika/Data Science Projects/20_newsgroups/20news_bydate/20news_bydate_test'

TRAIN_OUTPUT_PATH = 'Z:/Nika Yanuka/Nika/Data Science Projects/20_newsgroups/Train_data_vecs.csv'
TEST_OUTPUT_PATH =  'Z:/Nika Yanuka/Nika/Data Science Projects/20_newsgroups/Test_data_vecs.csv'

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.LabeledSentence(doc,    [self.labels_list[idx]])




def convert_to_true_lables(Y):
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



def eliminate_low_frequencies(data,labels):  
    #new_data = data
    #loop over the labels vector
    for label in list(np.unique(labels)):
        print('Removing freq from label ',label)
        indices = [i for i, value in enumerate(labels) if value == label] 
        words_vector = []
        for i in indices:
            for word in data[i]:
                words_vector.append(word)
        
        f_dist = FreqDist(words_vector)
        
        words_to_remove = []
        for tup in f_dist.items():
            if tup[1] <= 2:
                words_to_remove.append(tup[0])
        
        for i in indices:
            for word in data[i]:
                if word in words_to_remove:
                    data[i].remove(word)
            
        
    return data


def eliminate_low_frequencies_test(data):
    new_data = []
    for file in data:
        f_dist = FreqDist(file)
        
        for tup in f_dist.items():
            if tup[1] <= 2:
                file.remove(tup[0])
        new_data.append(file)
        
    return new_data
                
        

def implement_nltk(data):
    nltk.download('stopwords')
    tokenizer = RegexpTokenizer(r'\w+')
    stopword_set = set(stopwords.words('english'))

    new_data = []
    for file in data:
        new_str = file.lower()
        dlist = tokenizer.tokenize(new_str)
        dlist = list(set(dlist).difference(stopword_set))
        new_data.append(dlist)
        
    return new_data
                


def lemmatization(data):
    nltk.download('wordnet')
    ps=PorterStemmer()
    #define the lemmatiozation object
    lemmatizer = WordNetLemmatizer()
    
    new_data = []
    for file in data:
        new_file = []
        for word in file:
            stemmed_word = ps.stem(word)
            lemetized_word = lemmatizer.lemmatize(stemmed_word)
            new_file.append(lemetized_word)
        new_data.append(new_file)
        
    return new_data


def prepare_files_list(path):
    data = []
    labels= [] 
    file_names = []

    for directory in listdir(path):
        doc_labels = []
        doc_labels = [f for f in listdir(path +'/'+ directory)] #if f.endswith('.txt')]
        
        
        for file in doc_labels:
            labels.append(directory)
            data.append(open(path +'/'+ directory +'/'+ file).read())
            file_names.append(file)


    return data,labels,file_names
        



def create_final_data_file(path,data,labels,model,filenames):
    writer = csv.writer(open(path,'w',newline = ''))
    
    #write headers 
    headers = ['File name']
    for i in range(300):
        head = 'f' + str(i)
        headers.append(head)
        
    headers.append('Label')
    writer.writerow(headers)
    
    for i in range(len(data)):
        line = [filenames[i]]
        result_vector = model.infer_vector(data[i])
        for k in range(len(result_vector)):
            line.append(result_vector[k])
        line.append(labels[i])
        
        writer.writerow(line)
        
    
        
    


def test_model(data):
    #loading the model
    d2v_model = gensim.models.doc2vec.Doc2Vec.load('doc2vec.model')
    #start testing
    #printing the vector of document at index 1 in docLabels
    docvec = d2v_model.docvecs[1]
    print(docvec)
    #printing the vector of the file using its name
    docvec = d2v_model.docvecs['1.txt'] #if string tag used in training
    print(docvec)
    #to get most similar document with similarity scores using document-index
    similar_doc = d2v_model.docvecs.most_similar(14) 
    print(similar_doc)
    #to get most similar document with similarity scores using document- name
    sims = d2v_model.docvecs.most_similar('1.txt')
    print(sims)
    #to get vector of document that are not present in corpus 
    docvec = d2v_model.docvecs.infer_vector('war.txt')
    print(docvec)


def main():
    # Data processing 
    data,labels,file_names = prepare_files_list(TRAIN_INPUT_PATH)
    print('Cleaning data...')
    cleaned_data = implement_nltk(data)
    print('lematizing data...')
    lematized_data = lemmatization(cleaned_data)
    print('Generating true labels')
    true_labels = convert_to_true_lables(labels)
    print('Removing frequencies...')
    Freq_data = eliminate_low_frequencies(lematized_data,true_labels)
    #Freq_data = eliminate_low_frequencies_test(lematized_data)
    it = LabeledLineSentence(Freq_data, true_labels)
    
    
    # Doc2Vec model decleration
    model = gensim.models.Doc2Vec(vector_size=100, min_count=1, alpha=0.025, min_alpha=0.025)
    model.build_vocab(it)

    #training the model
    print('Start training...')
    model.train(it, epochs=model.iter, total_examples=model.corpus_count)
       
         
    #saving the created model
    model.save('doc2vec.model')
    print('model saved')
#    print(model.infer_vector(cleaned_data[11000]))
    
    # Create train data-base
    create_final_data_file(TRAIN_OUTPUT_PATH,Freq_data,true_labels,model,file_names)
    
    # Test data processing
    print('Start test')
    Test_data,Test_labels,Test_file_names = prepare_files_list(TEST_INPUT_PATH)
    Test_cleaned_data = implement_nltk(Test_data)
    Test_lematized_data = lemmatization(Test_cleaned_data)
    true_test_labels = convert_to_true_lables(Test_labels)
    print('Removing frequencies TEST...')
    Freq_test_data=Test_lematized_data
    it = LabeledLineSentence(Freq_test_data, true_test_labels)

    #Create test data base
    create_final_data_file(TEST_OUTPUT_PATH,Freq_test_data,true_test_labels,model,Test_file_names)
    
    
    print('Doc2Vec DONE !!!')  
    
    
    
   
        
    
if __name__ == "__main__":
    main()

