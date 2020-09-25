# Modular Bi-LSTM script

# Import required modules


import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import torch
from torchtext import data
import torch.nn as nn
import pandas as pd
import itertools
import sys
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
import random
from sklearn.metrics import classification_report
nltk.download('wordnet')
stopwords = open("stopwords.txt",'r').read().split('\n\n')


SEED = 44

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class DataFrameDataset(data.Dataset):

    def __init__(self, df, fields, is_test=False, **kwargs):
        examples = []
        for i, row in df.iterrows():
            label = row.label if not is_test else None
            text = row.text
            examples.append(data.Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)
        data_field = fields

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), data_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_field, True, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)

class LSTM_net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        
        # text = [sent len, batch size]
        embedded = self.embedding(text)
        # embedded = [sent len, batch size, emb dim]
          
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        output = self.fc1(hidden)
        output = self.dropout(self.fc2(output))
            
        return output

def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def fpr(preds,y):
    rounded_preds = torch.round(torch.sigmoid(preds)).cpu().numpy()
    y = y.cpu().numpy()

    return np.array([float(i) for i in classification_report(rounded_preds,y).split('\n')[6].split('      ')[1:-1]])

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


def GridSearchBiLSTM(params,train_iterator,valid_iterator,test_iterator):
  param_combinations = list(itertools.product(*[params['lr'],
                                              params['n_layers'],
                                              params['dropout']]))
  best_score = 0
  best_params = [0,0,0]
  for learning_rate,N_LAYERS,DROPOUT in param_combinations:
    num_epochs = 8
    

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    BIDIRECTIONAL = True
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] 

    model = LSTM_net(INPUT_DIM, 
            EMBEDDING_DIM, 
            HIDDEN_DIM, 
            OUTPUT_DIM, 
            N_LAYERS, 
            BIDIRECTIONAL, 
            DROPOUT, 
            PAD_IDX)

    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    model.to(device) #to GPU
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(num_epochs):
        
        train_loss, train_acc = model_train(model, train_iterator,optimizer,criterion)
        valid_acc, _ = evaluate(model, valid_iterator)
        test_acc, _  = evaluate(model, test_iterator )
        
      
    score = valid_acc


    if score > best_score:
      best_score == score
      best_params = [learning_rate,N_LAYERS,DROPOUT]

  return best_params[0],best_params[1],best_params[2]



# training function 
def model_train(model, iterator,optimizer,criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        text, text_lengths = batch.text
        
        optimizer.zero_grad()
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator):
    epoch_acc = 0
    model.eval()
    FPR = np.array([0.0,0.0,0.0])
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            acc = binary_accuracy(predictions, batch.label)
            FPR += fpr(predictions,batch.label)
            epoch_acc += acc.item()

    return epoch_acc / len(iterator), FPR/len(iterator) 

if __name__ == "__main__":

    """
    Here Dataset is being fed and preprocessing is being done.
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
        DEV_DF   = create_dataframe(sys.argv[2])
        TEST_DF  = create_dataframe(sys.argv[3])

    print('\n\nPREPROCESSING...')
    train = TRAIN_DF.iloc[:,-2:]
    train['text'] = train['text'].map(preprocess)

    dev = DEV_DF.iloc[:,-2:]
    dev['text'] = dev['text'].map(preprocess)

    test = TEST_DF.iloc[:,-2:]
    test['text'] = test['text'].map(preprocess)
    print('\n\nPREPROCESSING IS DONE!')
    

    bilstm_question = input('\nWould you like to continue with mixed sampling or class sampling? [mixed/class]:')

    params = {'lr' : [0.01 , 0.001],
              'n_layers' : [2 , 3],
              'dropout' : [0.2,0.5]}

    """
    This is the point.
    """
    if bilstm_question == 'mixed':
        iters = int(input('Number of iterations:'))

        steps = [1750,2250,2750,3250] # Step size is relative. Might change it to get reliable scores or for less time consuming execution.

        bilstm_mixed = [[] for i in range(len(steps))]

        for size in steps: # sample size is chosen
            for _ in range(iters): # sampling same size for multiple time due to realiabity of this experiment

            train = TRAIN_DF.sample(size,random_state= random.sample(range(6760),1)[0])
            dev = DEV_DF
            test = TEST_DF

            TEXT  = data.Field(tokenize = 'spacy', include_lengths = True)
            LABEL = data.LabelField(dtype = torch.float)

            fields = [('text',TEXT), ('label',LABEL)]
            train_ds, val_ds, test_ds = DataFrameDataset.splits(fields, train_df = train, val_df = dev, test_df=test)

            MAX_VOCAB_SIZE = 25000
            
            TEXT.build_vocab(train_ds, 
                            max_size = MAX_VOCAB_SIZE, 
                            vectors = 'glove.6B.200d', # this will take some time to download but what can i do?
                            unk_init = torch.Tensor.zero_)

            LABEL.build_vocab(train_ds)

            BATCH_SIZE = 64

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                (train_ds, val_ds, test_ds), 
                batch_size = BATCH_SIZE,
                sort_within_batch = True,
                    device = device)
            
            learning_rate,N_LAYERS,DROPOUT = GridSearchBiLSTM(params,train_iterator,valid_iterator,test_iterator)

            num_epochs = 10
            INPUT_DIM = len(TEXT.vocab)
            EMBEDDING_DIM = 200
            HIDDEN_DIM = 256
            OUTPUT_DIM = 1
            BIDIRECTIONAL = True
            PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] 

            model = LSTM_net(INPUT_DIM, 
                    EMBEDDING_DIM, 
                    HIDDEN_DIM, 
                    OUTPUT_DIM, 
                    N_LAYERS, 
                    BIDIRECTIONAL, 
                    DROPOUT, 
                    PAD_IDX)

            pretrained_embeddings = TEXT.vocab.vectors
            model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

            model.to(device) #to GPU
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


            for epoch in range(num_epochs):
                train_loss, train_acc = model_train(model, train_iterator,optimizer,criterion)
            
            _,DFPR = evaluate(model,valid_iterator)
            _,TFPR = evaluate(model,test_iterator)
            
            r = (DFPR,TFPR)
            
            bilstm_mixed[steps.index(size)].append(r) # save the scores
           
        bilstm_mixed.insert(0,[[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]])
        steps.insert(0,0)

        BILSTM_MIXED_FILE = {'steps': steps, 'results': mlp_class0}

        filename = 'BILSTM_MIXED_RESULTS.pickle'
        pickle.dump(BILSTM_MIXED_FILE, open("../outputs/pickled_results/"+filename, 'wb'))

    elif bilstm_question == 'class':
        
        which_class = input('Which class would you like to sample? [0,1,both]:')
        iters = int(input('Number of iterations:'))

        steps = [1750,2250,2750,3250] # Step size is relative. Might change it to get reliable scores or for less time consuming execution.

        bilstm_mixed = [[] for i in range(len(steps))]

        for size in steps: # sample size is chosen
            for _ in range(iters): # sampling same size for multiple time due to realiabity of this experiment

            train = TRAIN_DF.sample(size,random_state= random.sample(range(6760),1)[0])
            dev = DEV_DF
            test = TEST_DF

            TEXT  = data.Field(tokenize = 'spacy', include_lengths = True)
            LABEL = data.LabelField(dtype = torch.float)

            fields = [('text',TEXT), ('label',LABEL)]
            train_ds, val_ds, test_ds = DataFrameDataset.splits(fields, train_df = train, val_df = dev, test_df=test)

            MAX_VOCAB_SIZE = 25000
            
            TEXT.build_vocab(train_ds, 
                            max_size = MAX_VOCAB_SIZE, 
                            vectors = 'glove.6B.200d', # this will take some time to download but what can i do?
                            unk_init = torch.Tensor.zero_)

            LABEL.build_vocab(train_ds)

            BATCH_SIZE = 64

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                (train_ds, val_ds, test_ds), 
                batch_size = BATCH_SIZE,
                sort_within_batch = True,
                    device = device)
            
            learning_rate,N_LAYERS,DROPOUT = GridSearchBiLSTM(params,train_iterator,valid_iterator,test_iterator)

            num_epochs = 10
            INPUT_DIM = len(TEXT.vocab)
            EMBEDDING_DIM = 200
            HIDDEN_DIM = 256
            OUTPUT_DIM = 1
            BIDIRECTIONAL = True
            PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] 

            model = LSTM_net(INPUT_DIM, 
                    EMBEDDING_DIM, 
                    HIDDEN_DIM, 
                    OUTPUT_DIM, 
                    N_LAYERS, 
                    BIDIRECTIONAL, 
                    DROPOUT, 
                    PAD_IDX)

            pretrained_embeddings = TEXT.vocab.vectors
            model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

            model.to(device) #to GPU
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


            for epoch in range(num_epochs):
                train_loss, train_acc = model_train(model, train_iterator,optimizer,criterion)
            
            _,DFPR = evaluate(model,valid_iterator)
            _,TFPR = evaluate(model,test_iterator)
            
            r = (DFPR,TFPR)
            
            bilstm_mixed[steps.index(size)].append(r) # save the scores
           
        bilstm_mixed.insert(0,[[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]])
        steps.insert(0,0)

        BILSTM_MIXED_FILE = {'steps': steps, 'results': mlp_class0}

        filename = 'BILSTM_MIXED_RESULTS.pickle'
        pickle.dump(BILSTM_MIXED_FILE, open("../outputs/pickled_results/"+filename, 'wb'))
              

    
# Create same for classes.