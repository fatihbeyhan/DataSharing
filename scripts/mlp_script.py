# Modular MLP script

# Import required modules
import json
import pandas as pd
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import copy
import random
import pickle
import itertools
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



import torch
torch.manual_seed(44)
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import timeit

import sys

import statistics as stats



#Creating MLP
class MLP(nn.Module):
    def __init__(self,feature_size,hidden_unit):
        super(MLP,self).__init__()
        self.hidden_unit = hidden_unit
        
        self.layer1 = nn.Linear(feature_size,self.hidden_unit[0])
        self.layer2 = nn.Linear(self.hidden_unit[0],self.hidden_unit[1])
        self.layer3 = nn.Linear(self.hidden_unit[1],1)

        self.drop = nn.Dropout(p=0.2)

        
    #This must be implemented
    def forward(self,x):
    
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.drop(x)
        
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.drop(x)

        x = self.layer3(x)
        x = torch.sigmoid(x)

        return x
        
    #This function takes an input and predicts the class, (0 or 1)        
    def predict(self,x):
        #Apply softmax to output. 
        pred = self.forward(x)
        ans = []
        #Pick the class with maximum weight
        for t in pred:
            if t[0]>0.500001:
                ans.append(1)
            else:
                ans.append(0)
        return ans


def model_train(m,l,o,training_batches,val_x,val_y):
  epochs = 20
  last_eval = 100
  for i in range(0,epochs):
    best_model = copy.deepcopy(m)
    for batch in training_batches:
      x = batch[0]
      y = batch[1]

      y_pred = m.forward(x)
      loss = l(y_pred,y)    

      o.zero_grad()
      loss.backward()
      o.step()
    if i % 1 == 0:
      evalloss = l(m.forward(val_x),val_y)
      if last_eval>evalloss:
        last_eval = evalloss
      if last_eval < evalloss:
        break
  return best_model


def gscv_data(min_df,ngram_range,dataset,random_state):
  #if val == True:
  tr,te =train_test_split(dataset,
                          test_size=0.2,
                          random_state= random_state)

  vectorizer = TfidfVectorizer(ngram_range=ngram_range,min_df=min_df)
  
  trt =   torch.from_numpy(vectorizer.fit_transform(tr.text).toarray().astype('float64')).float()
  trl =   torch.from_numpy(tr.iloc[:,-1:].to_numpy().astype('float64')).float()

  tet = torch.from_numpy(vectorizer.transform(te.text).toarray().astype('float64')).float()
  tel = torch.from_numpy(te.iloc[:,-1:].to_numpy().astype('float64')).float()

  train_batches = DataLoader(TensorDataset(trt,trl), batch_size=40, shuffle=False)
  return train_batches, tet, tel, vectorizer
  #elif val==False:

   # vectorizer = TfidfVectorizer(ngram_range=ngram_range,min_df=min_df)
    
    #trt =   torch.from_numpy(vectorizer.fit_transform(dataset.text).toarray().astype('float64')).float()
    #trl =   torch.from_numpy(dataset.iloc[:,-1:].to_numpy().astype('float64')).float()

    #train_batches = DataLoader(TensorDataset(trt,trl), batch_size=trt.shape[0], shuffle=False)
    #return train_batches


def GridSearchCV_MLP(dataset,params,cv=3):
  param_combinations = list(itertools.product(*[params['tfidf']['min_df'],params['tfidf']['ngram_range'],
                                                params['mlp']['lr'],params['mlp']['hidden_unit']]))
  results = {}
  for p in param_combinations:

    param_result = []

    for _ in range(cv):

      min_df,ngram_range,lr,hidden_unit = p[0],p[1],p[2],p[3]        
      
      t_batch,val_x,val_y,n = gscv_data(min_df,ngram_range,dataset=dataset,random_state=_)
      
      torch.manual_seed(44)
      m = MLP(feature_size=val_x.shape[1],hidden_unit=hidden_unit)
      l = nn.BCELoss()
      o = torch.optim.Adam(m.parameters(), lr=lr) 
      model = model_train(m,l,o,t_batch,val_x,val_y) 

      param_result.append(f1_score(val_y,model.predict(val_x),average='macro'))       
    
    results[p] = stats.mean(param_result)
    p = list(results.keys())[list(results.values()).index(max(list(results.values())))]
  return p[0],p[1],p[2],p[3]

def scores(b_m,dx,dy,tx,ty):
  df,dp,dr = [float(i.strip()) for i in classification_report(b_m.predict(dx),dy).split('\n')[-3].split('      ')[1:-1]]
  tf,tp,tr = [float(i.strip()) for i in classification_report(b_m.predict(tx),ty).split('\n')[-3].split('      ')[1:-1]]
  return df,dp,dr,tf,tp,tr

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





if __name__ == "__main__":


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
        DEV_DF   = create_dataframe(sys.argv[2])
        TEST_DF  = create_dataframe(sys.argv[3])


    print('\n\nPREPROCESSING...')
    train = TRAIN_DF.iloc[:,-2:]
    train['text'] = train['text'].map(preprocess)

    dev = DEV_DF.iloc[:,-2:]
    dev['text'] = dev['text'].map(preprocess)

    test = TEST_DF.iloc[:,-2:]
    test['text'] = test['text'].map(preprocess)
    print('\nPREPROCESSING IS DONE!')
    
    mlp_question = input('\nWould you like to continue with mixed sampling or class sampling? [mixed/class]:')
    
    params = {'tfidf':{'min_df':[10,20],'ngram_range':[(1,1),(1,2)]},'mlp':{'lr':[0.001,0.0001],'hidden_unit':[(512,256),(256,128)]}}
    
    if mlp_question == 'mixed':
      iters = int(input('Number of iterations:'))

      steps = [1250,1750,2250,2750,3250]
      
      mlp_mixed = [[] for i in range(len(steps))]
      for size in steps: # sample size is chosen
        for _ in range(iters): # sampling same size for multiple time due to realiabity of this experiment

          y = train.sample(size, random_state=random.sample(range(6760),1)[0]) # sample with random random_state to make sure you get different sets each time
        
          min_df,ngram_range,lr,hidden_unit = GridSearchCV_MLP(dataset=y,params=params,cv=3) # find the best parameters for this sample
          train_batches, val_x, val_y , vec = gscv_data(min_df,ngram_range,y,random_state=0) # make your data ready for training a model with the best params
        
          m = MLP(feature_size=(next(iter(train_batches)))[0].shape[1],hidden_unit=hidden_unit) # create your model with parameters
          l = nn.BCELoss()
          o = torch.optim.Adam(m.parameters(), lr=lr) 
        
          model = model_train(m,l,o,train_batches,val_x,val_y) # train the model

          # test on the dev and test set
          dev_x =   torch.from_numpy(vec.transform(dev.text).toarray().astype('float64')).float()
          dev_y =   torch.from_numpy(dev.iloc[:,-1:].to_numpy().astype('float64')).float()

          test_x = torch.from_numpy(vec.transform(test.text).toarray().astype('float64')).float()
          test_y = torch.from_numpy(test.iloc[:,-1:].to_numpy().astype('float64')).float()

          df,dp,dr,tf,tp,tr = scores(model,dev_x,dev_y,test_x,test_y) # get your scores
          
          r = ((df,dp,dr),(tf,tp,tr))
          mlp_mixed[steps.index(size)].append(r) # save the scores

      mlp_mixed.insert(0,[[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]])
      steps.insert(0,0)

      MLP_FILE = {'steps': steps, 'results': mlp_mixed}

      filename = 'MLP_RESULTS.pickle'
      pickle.dump(MLP_FILE, open("../outputs/pickled_results/"+filename, 'wb'))



    elif mlp_question == 'class':

      which_class = input('Which class would you like to sample? [0,1,both]:')
      iters = int(input('Number of iterations:'))

      if which_class == 'both':
        
        #class 0
        steps = [1250,1750,2250,2723]

        mlp_class0 = [[] for i in range(len(steps))]

        for size in steps: # sample size is chosen

          for _ in range(iters): # sampling same size for multiple time due to realiabity of this experiment

            y = pd.concat([train[train.label == 0].sample(size, random_state= random.sample(range(6760),1)[0] ),
                              train[train.label == 1]]).sample(850+size) # sample with random random_state to make sure you get different sets each time
          
            min_df,ngram_range,lr,hidden_unit=GridSearchCV_MLP(dataset=y,params=params,cv=3) # find the best parameters for this sample
            train_batches, val_x, val_y , vec = gscv_data(min_df,ngram_range,y,random_state=0) # make your data ready for training a model with the best params
          
            m = MLP(feature_size=(next(iter(train_batches)))[0].shape[1],hidden_unit=hidden_unit) # create your model with parameters
            l = nn.BCELoss()
            o = torch.optim.Adam(m.parameters(), lr=lr) 
          
            model = model_train(m,l,o,train_batches,val_x,val_y) # train the model

            # test on the dev and test set
            dev_x =   torch.from_numpy(vec.transform(dev.text).toarray().astype('float64')).float()
            dev_y =   torch.from_numpy(dev.iloc[:,-1:].to_numpy().astype('float64')).float()

            test_x = torch.from_numpy(vec.transform(test.text).toarray().astype('float64')).float()
            test_y = torch.from_numpy(test.iloc[:,-1:].to_numpy().astype('float64')).float()

            df,dp,dr,tf,tp,tr = scores(model,dev_x,dev_y,test_x,test_y) # get your scores
            
            
            r = ((df,dp,dr),(tf,tp,tr))
            mlp_class0[steps.index(size)].append(r) # save the scores
            
        mlp_class0.insert(0,[[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]])
        steps.insert(0,0)

        MLP_CLASS0_FILE = {'steps': steps, 'results': mlp_class0}

        filename = 'MLP_CLASS0_RESULTS.pickle'
        pickle.dump(MLP_CLASS0_FILE, open("../outputs/pickled_results/"+filename, 'wb'))


        #class 1

        steps = [200,400,600,850]

        mlp_class1 = [[] for i in range(len(steps))]

        for size in steps: # sample size is chosen

          for _ in range(iters): # sampling same size for multiple time due to realiabity of this experiment

            y = pd.concat([train[train.label == 1].sample(size, random_state=random.sample(range(6760),1)[0]),
                       train[train.label == 0]]).sample(2723+size) # sample with random random_state to make sure you get different sets each time
          
            min_df,ngram_range,lr,hidden_unit=GridSearchCV_MLP(dataset=y,params=params,cv=3) # find the best parameters for this sample
            train_batches, val_x, val_y , vec = gscv_data(min_df,ngram_range,y,random_state=0) # make your data ready for training a model with the best params
          
            m = MLP(feature_size=(next(iter(train_batches)))[0].shape[1],hidden_unit=hidden_unit) # create your model with parameters
            l = nn.BCELoss()
            o = torch.optim.Adam(m.parameters(), lr=lr) 
          
            model = model_train(m,l,o,train_batches,val_x,val_y) # train the model

            # test on the dev and test set
            dev_x =   torch.from_numpy(vec.transform(dev.text).toarray().astype('float64')).float()
            dev_y =   torch.from_numpy(dev.iloc[:,-1:].to_numpy().astype('float64')).float()

            test_x = torch.from_numpy(vec.transform(test.text).toarray().astype('float64')).float()
            test_y = torch.from_numpy(test.iloc[:,-1:].to_numpy().astype('float64')).float()

            df,dp,dr,tf,tp,tr = scores(model,dev_x,dev_y,test_x,test_y) # get your scores
            
            
            r = ((df,dp,dr),(tf,tp,tr))
            mlp_class0[steps.index(size)].append(r) # save the scores
           
        mlp_class1.insert(0,[[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]])
        steps.insert(0,0)

        MLP_CLASS1_FILE = {'steps': steps, 'results': mlp_class1}

        filename = 'MLP_CLASS1_RESULTS.pickle'
        pickle.dump(MLP_CLASS1_FILE, open("../outputs/pickled_results/"+filename, 'wb'))



