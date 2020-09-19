# Modular GNB script

# Import required modules
import json
from datetime import datetime
import random
import pickle
import sys
import statistics as stat
import pandas as pd
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.base import TransformerMixin
from sklearn.model_selection import cross_val_score,train_test_split,GridSearchCV
from sklearn.metrics import roc_auc_score,confusion_matrix,accuracy_score,f1_score,precision_score,recall_score,precision_recall_fscore_support,classification_report
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import seaborn as sns
stopwords = open("../data/stopwords.txt",'r').read().split('\n\n')

def create_dataframe(file):

    """
    This function will return the given file in needed format.
    """

    if 'json' in file:
        list_of_articles = []
        with open(file,'rb') as f:
            for line in f:
                article = json.loads(line)
                list_of_articles.append(article)

        data = [[' '.join(article['text'].split('\n')),article['label']] for article in list_of_articles]
        return pd.DataFrame(data,columns=['text','label'])
    if 'csv' in file:
        return pd.DataFrame(pd.read_csv(file),columns=['text','label'])

def preprocess(news):

    """
    This function will preprocess the text.

    split an article into sentences
     go to each sentence and split it to words
      if this word  is not in stopwords or other common words I've decided
       AND
      if its alphabetic (getting rid of puctuation and numbers)
       AND
      if len of the word is greater than 2
        lemmatize and lowercase the the word
      return the cleaned article
    """
    l = WordNetLemmatizer()
    sentences = news.split(".")
    return " ".join([l.lemmatize(word.lower()) for sentence in sentences for word in sentence.split() if word not in stopwords if word.isalpha() if len(word)> 2 if word.lower() not in ["said","the","first","also","would","one","two","they"]])

def SVM(train,dev,test,iters):

    """
    this function will do the 'job' for mixed sampling. it will split the dataset into incremental steps
    then apply grid search etc. and finally return the results.
    """

    steps = list(range(round(train.shape[0]/10),train.shape[0],round(train.shape[0]/10)))
    steps.append(train.shape[0])
    
    svm_mixed = [[] for i in range(len(steps))]

    for step in steps:
        start = datetime.now()
        for _ in range(iters):
            
            random_state = random.sample(range(100),1)[0] # Changing random_state for each iteration to get different sets.
            
            T = train.sample(step,random_state=random_state)
            
            #GRIDSEARCHCV
            svc_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', SVC()),
            ])

            hyperparameters = dict(
                tfidf__min_df      = (4, 10, 16),
                tfidf__ngram_range = ((1, 1), (1, 2), (1, 3)),
                clf__kernel        = ["linear", "poly","sigmoid"],
                clf__gamma         = ('scale', 'auto'),
                clf__C             = np.logspace(1,3,3)

            )

            svc_grid_search = GridSearchCV(svc_pipeline, hyperparameters,cv=3,scoring='f1_macro',n_jobs=2)

            svc_grid_search.fit(T.text, list(T.label))            
            #END
            
            #CREATE THE MODEL
            svc_model = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', SVC()),
            ]).set_params(**svc_grid_search.best_params_)
            
            svc_model.fit(T.text, list(T.label))
            
            #TEST ON THE SETS
            
            d_res = classification_report(list(dev.label),svc_model.predict(dev.text)).split('\n')[6].split('      ')[1:-1]
            t_res = classification_report(list(test.label),svc_model.predict(test.text)).split('\n')[6].split('      ')[1:-1]
            
            svm_mixed[steps.index(step)].append([[float(i) for i in d_res],[float (i) for i in t_res]])

        end = datetime.now()
        print('['+str(steps.index(step)+1)+'/'+str(len(steps))+']',(end-start).total_seconds(),'secs')
    svm_mixed.insert(0,[[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]])
    steps.insert(0,0)
    return steps,svm_mixed

def SVM_CLASS(train,dev,test,which_class,iters):
    """
    this function will do the 'job' for class samplings. it will split the dataset into incremental steps
    then apply grid search etc. and finally return the results.
    """
          
    if which_class == 0.0:
        steps = list(range(round(train[train.label == which_class].shape[0]/10),
                   train[train.label == which_class].shape[0],
                   round(train[train.label == which_class].shape[0]/10)))
        
    elif which_class == 1.0:
        steps = list(range(round(train[train.label == which_class].shape[0]/8),
                   train[train.label == which_class].shape[0],
                   round(train[train.label == which_class].shape[0]/8)))
        steps.append(train[train.label == which_class].shape[0])
    
    svm_class = [[] for i in range(len(steps))]

    for step in steps:
        start = datetime.now()        
        for _ in range(iters):
            
            random_state = random.sample(range(100),1)[0] # Changing random_state for each iteration to get different sets.
            
            if which_class == 0.0:
                
                T = pd.concat([train[train.label == 0.0].sample(step,random_state=random_state),
                               train[train.label == 1.0]]).sample(step+train[train.label == 1.0].shape[0])
                
            elif which_class == 1.0:
                
                T = pd.concat([train[train.label == 1.0].sample(step,random_state=random_state),
                               train[train.label == 0.0]]).sample(step+train[train.label == 0.0].shape[0])
            
            #GRIDSEARCHCV
            svc_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', SVC()),
            ])

            hyperparameters = dict(
                tfidf__min_df      = (4, 10, 16),
                tfidf__ngram_range = ((1, 1), (1, 2), (1, 3)),
                clf__kernel        = ["linear", "poly","sigmoid"],
                clf__gamma         = ('scale', 'auto'),
                clf__C             = np.logspace(1,3,3)

            )

            svc_grid_search = GridSearchCV(svc_pipeline, hyperparameters,cv=3,scoring='f1_macro',n_jobs=-1)

            svc_grid_search.fit(T.text, list(T.label))            
            #END
            
            #CREATE THE MODEL
            svc_model = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', SVC()),
            ]).set_params(**svc_grid_search.best_params_)
            
            svc_model.fit(T.text, list(T.label))
            
            #TEST ON THE SETS
            
            d_res = classification_report(list(dev.label),svc_model.predict(dev.text)).split('\n')[6].split('      ')[1:-1]
            t_res = classification_report(list(test.label),svc_model.predict(test.text)).split('\n')[6].split('      ')[1:-1]

            svm_class[steps.index(step)].append([[float(i) for i in d_res],[float (i) for i in t_res]])
        end = datetime.now()
        print('['+str(steps.index(step)+1)+'/'+str(len(steps))+']',(end-start).total_seconds(),'secs')
    svm_class.insert(0,[[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]])
    steps.insert(0,0)
    return steps,svm_class
        
 
  


if __name__ == '__main__':

    """
    when script is called, at least one data file has to be given in a json or csv format. 
    """

    SEED = 44
    print('\nSTARTING...\n\n')

    if len(sys.argv) == 1:
        print('THIS SCRIPT TAKES AT LEAST ONE ARGUMENT. PLEASE PROVIDE DATA...')
    
    elif len(sys.argv) == 2:

        DF = create_dataframe(sys.argv[1])

        TRAIN_DF, _  = train_test_split(DF, test_size = 0.25, random_state=SEED)
        DEV_DF, TEST_DF = train_test_split(_ , test_size = 0.50, random_state=SEED)


    elif len(sys.argv) == 3:

        TRAIN_DF = create_dataframe(sys.argv[1])

        DEV_DF, TEST_DF = train_test_split(create_dataframe(sys.argv[2]),test_size = 0.50, random_state=SEED)

    elif len(sys.argv) == 4:

        TRAIN_DF = create_dataframe(sys.argv[1])
        DEV_DF = create_dataframe(sys.argv[2])
        TEST_DF = create_dataframe(sys.argv[3])

    print('\n\nPREPROCESSING...')
    train = TRAIN_DF.iloc[:,-2:]
    train['text'] = train['text'].map(preprocess)

    dev = DEV_DF.iloc[:,-2:]
    dev['text'] = dev['text'].map(preprocess)

    test = TEST_DF.iloc[:,-2:]
    test['text'] = test['text'].map(preprocess)
    print('\nPREPROCESSING IS DONE!')


    svm_question = input('\nWould you like to continue with mixed sampling or class sampling? [mixed/class]:')

    if svm_question == 'mixed':
        iters = int(input('Number of iterations:'))
        steps,svm_results = SVM(train,dev,test,iters)

        SVM_FILE = {'steps':steps,'results':svm_results}

        filename = 'SVM_RESULTS.pickle'
        pickle.dump(SVM_FILE, open("../outputs/pickled_results/"+filename, 'wb'))

    elif svm_question == 'class':
        which_class = input('Which class would you like to sample? [0,1,both]:')
        iters = int(input('Number of iterations:'))
        
        if which_class == 'both':
            class1_steps,svm_class1_results = SVM_CLASS(train,dev,test,1.0,iters)
            SVM_CLASS1_FILE = {'steps':class1_steps,'results':svm_class1_results}
            filename = 'SVM_CLASS1_RESULTS.pickle'
            pickle.dump(SVM_CLASS1_FILE, open("../outputs/pickled_results/"+filename, 'wb'))
            
            print ('\n\nCLASS 1 IS DONE. CLASS 2 IS STARTING!\n\n')
            
            class0_steps,svm_class0_results = SVM_CLASS(train,dev,test,0.0,iters)
            SVM_CLASS0_FILE = {'steps':class0_steps,'results':svm_class0_results}
            filename = 'SVM_CLASS0_RESULTS.pickle'
            pickle.dump(SVM_CLASS0_FILE, open("../outputs/pickled_results/"+filename, 'wb'))

