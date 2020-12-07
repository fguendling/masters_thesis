# -*- coding: utf-8 -*-
# examples
# https://colab.research.google.com/drive/1ixOZTKLz4aAa-MtC6dy_sAvc9HujQmHN 
#https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/ 
#https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04 

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import transformers
from transformers import AutoModel, BertTokenizerFast, BertForSequenceClassification, BertConfig
import chardet
from nltk import sent_tokenize

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW

from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      
      # dropout layer
      self.dropout = nn.Dropout(0.1)
      
      # relu activation function
      self.relu =  nn.ReLU()

      if True:
          # dense layer 1
          self.fc1 = nn.Linear(768,400)
          
          # dense layer 2 (Output layer)
          self.fc2 = nn.Linear(400,2)
      else:
          # dense layer 1
          self.fc1 = nn.Linear(768,500)
          
          # dense layer 2 (Output layer)
          self.fc2 = nn.Linear(500,200)

          # dense layer 2 (Output layer)
          self.fc3 = nn.Linear(200,2)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

      #pass the inputs to the model  
      _, cls_hs = self.bert(sent_id, attention_mask=mask)

      if True:
          x = self.fc1(cls_hs)
    
          x = self.relu(x)
    
          x = self.dropout(x)
    
          # output layer
          x = self.fc2(x)
          
      else:
          x = self.fc1(cls_hs)
    
          x = self.relu(x)
    
          x = self.dropout(x)
    
          # second layer
          x = self.fc2(x)
  
          x = self.relu(x)
    
          x = self.dropout(x)
    
          # output layer
          x = self.fc3(x)

      # apply softmax activation
      x = self.softmax(x)

      return x


class DoTrainBert():

    train_text = None
    train_labels = None
    temp_text = None
    temp_labels = None
    val_text = None
    test_text = None
    val_labels = None
    test_labels = None
    bert = None
    tokenizer = None

    train_seq = None
    train_mask = None
    train_y = None
    
    val_seq = None
    val_mask = None
    val_y = None
    
    test_seq = None
    test_mask = None
    test_y = None
    
    def load_data(self):
        
        df = pd.read_csv("/Users/felix/FRA_UAS/master/DHL_Document_Gap_Detector/scrapeweb/scripts/_20202310_Pretraining_test.csv", sep=',', error_bad_lines=False)

        (self.train_text, self.temp_text, 
         self.train_labels, self.temp_labels) = train_test_split(df['text'], df['label'], 
                                                       random_state=2018, 
                                                       test_size=0.3, 
                                                       stratify=df['label'])
        
        (self.val_text, self.test_text, 
         self.val_labels, self.test_labels) = train_test_split(self.temp_text, self.temp_labels, 
                                                     random_state=2018, 
                                                     test_size=0.5, 
                                                     stratify=self.temp_labels)

    def load_realdata(self):                                    
        # realdata would be one document, that was neither in train or test datasets
        # it has to be split into sentences (eval_sentences = sent_tokenize(doc))
        #df = pd.DataFrame()
        #df["text"] = eval_sentences

        #df = pd.read_csv(fn, encoding=result['encoding'])
        
        #self.test_text = df["text"]
        #MAXTOK = 30
        None

    def tokenize_realdata(self):
        
        MAXTOK = 30
        tokens_test = self.tokenizer.batch_encode_plus(
            self.test_text.tolist(),
            max_length = MAXTOK,
            pad_to_max_length=True,
            truncation=True
        )
        self.test_seq = torch.tensor(tokens_test['input_ids'])
        self.test_mask = torch.tensor(tokens_test['attention_mask'])
       
    def load_model(self):
        # import BERT-base pretrained model 

        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        #config = BertConfig.from_pretrained( 'bert-base-uncased', output_hidden_states=True)    
        #self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)                                                                  # Use the 12-layer BERT model, with an uncased vocab.
        # ^another option, its not working at the moment (from https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
        # Load the BERT tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    def tokenize_and_tensors(self):
    
        MAXTOK = 30
    
        tokens_train = self.tokenizer.batch_encode_plus(
            self.train_text.tolist(),
            max_length = MAXTOK,
            pad_to_max_length=True,
            truncation=True
        )
        
        # tokenize and encode sequences in the validation set
        tokens_val = self.tokenizer.batch_encode_plus(
            self.val_text.tolist(),
            max_length = MAXTOK,
            pad_to_max_length=True,
            truncation=True
        )
        
        # tokenize and encode sequences in the test set
        tokens_test = self.tokenizer.batch_encode_plus(
            self.test_text.tolist(),
            max_length = MAXTOK,
            pad_to_max_length=True,
            truncation=True
        )
        
        self.train_seq = torch.tensor(tokens_train['input_ids'])
        self.train_mask = torch.tensor(tokens_train['attention_mask'])
        self.train_y = torch.tensor(self.train_labels.tolist())
        
        self.val_seq = torch.tensor(tokens_val['input_ids'])
        self.val_mask = torch.tensor(tokens_val['attention_mask'])
        self.val_y = torch.tensor(self.val_labels.tolist())
        
        self.test_seq = torch.tensor(tokens_test['input_ids'])
        self.test_mask = torch.tensor(tokens_test['attention_mask'])
        self.test_y = torch.tensor(self.test_labels.tolist())

        #seq_len = [len(i.split()) for i in self.train_text]
        #pd.Series(seq_len).hist(bins = 30)

    def build_tensordatasets(self):
        #define a batch size
        batch_size = 20
        
        # wrap tensors
        train_data = TensorDataset(self.train_seq, self.train_mask, self.train_y)
        
        # sampler for sampling the data during training
        train_sampler = RandomSampler(train_data)
        
        # dataLoader for train set
        self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        
        # wrap tensors
        val_data = TensorDataset(self.val_seq, self.val_mask, self.val_y)
        
        # sampler for sampling the data during training
        val_sampler = SequentialSampler(val_data)
        
        # dataLoader for validation set
        self.val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

    def prep_bert(self, pWithWeights = True):
        
        for param in self.bert.parameters():
            param.requires_grad = False

        # specify GPU (changed from cuda to cpu because MacBooks don't have Nvidia GPU, they have Intel GPU only)
        self.device = torch.device("cpu")
    
        self.model = BERT_Arch(self.bert)
        
        # push the model to GPU
        self.gpumodel = self.model.to(self.device)
        
        print("build tensor datasets")

        # define the optimizer
        self.optimizer = AdamW(self.model.parameters(), lr = 2e-5)          # learning rate

        if pWithWeights:
            #compute the class weights
            self.class_weights = compute_class_weight('balanced', np.unique(self.train_labels), self.train_labels)
            
            print("Class Weights:", self.class_weights)
    
            # converting list of class weights to a tensor
            weights= torch.tensor(self.class_weights,dtype=torch.float)
            
            # push to GPU
            weights = weights.to(self.device)
            
            # define the loss function
            self.cross_entropy = nn.NLLLoss(weight=weights) 
        
        # number of training epochs
        self.epochs = 20

    def train(self):
      
      self.gpumodel.train()
    
      total_loss, total_accuracy = 0, 0
      
      # empty list to save model predictions
      total_preds=[]
      
      # iterate over batches
      for step,batch in enumerate(self.train_dataloader):
        
        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
          print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.train_dataloader)))
    
        # push the batch to gpu
        batch = [r.to(self.device) for r in batch]
     
        sent_id, mask, labels = batch
    
        # clear previously calculated gradients 
        self.gpumodel.zero_grad()        
    
        # get model predictions for the current batch
        preds = self.gpumodel(sent_id, mask)
    
        # compute the loss between actual and predicted values
        loss = self.cross_entropy(preds, labels)
    
        # add on to the total loss
        total_loss = total_loss + loss.item()
    
        # backward pass to calculate the gradients
        loss.backward()
    
        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(self.gpumodel.parameters(), 1.0)
    
        # update parameters
        self.optimizer.step()
    
        # model predictions are stored on GPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()
    
        # append the model predictions
        total_preds.append(preds)
    
      # compute the training loss of the epoch
      avg_loss = total_loss / len(self.train_dataloader)
      
      # predictions are in the form of (no. of batches, size of batch, no. of classes).
      # reshape the predictions in form of (number of samples, no. of classes)
      total_preds  = np.concatenate(total_preds, axis=0)
    
      #returns the loss and predictions
      return avg_loss, total_preds

    def evaluate(self):
      
      print("\nEvaluating...")
      
      # deactivate dropout layers
      self.gpumodel.eval()
    
      total_loss, total_accuracy = 0, 0
      
      # empty list to save the model predictions
      total_preds = []
    
      # iterate over batches
      for step,batch in enumerate(self.val_dataloader):
        
        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
          
          # Calculate elapsed time in minutes.
          #elapsed = format_time(time.time() - t0)
                
          # Report progress.
          print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.val_dataloader)))
    
        # push the batch to gpu
        batch = [t.to(self.device) for t in batch]
    
        sent_id, mask, labels = batch
    
        # deactivate autograd
        with torch.no_grad():
          
          # model predictions
          preds = self.gpumodel(sent_id, mask)
    
          # compute the validation loss between actual and predicted values
          loss = self.cross_entropy(preds,labels)
    
          total_loss = total_loss + loss.item()
    
          preds = preds.detach().cpu().numpy()
    
          total_preds.append(preds)
    
      # compute the validation loss of the epoch
      avg_loss = total_loss / len(self.val_dataloader) 
    
      # reshape the predictions in form of (number of samples, no. of classes)
      total_preds  = np.concatenate(total_preds, axis=0)
    
      return avg_loss, total_preds    

    def final(self):
        # set initial loss to infinite
        best_valid_loss = float('inf')
        
        # empty lists to store training and validation loss of each epoch
        train_losses=[]
        valid_losses=[]
        
        #for each epoch
        for epoch in range(self.epochs):
             
            print('\n Epoch {:} / {:}'.format(epoch + 1, self.epochs))
            
            #train model
            train_loss, _ = self.train()
            
            #evaluate model
            valid_loss, _ = self.evaluate()
            
            #save the best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.gpumodel.state_dict(), 'saved_weights.pt')
            
            # append training and validation loss
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            
            print(f'\nTraining Loss: {train_loss:.3f}')
            print(f'Validation Loss: {valid_loss:.3f}')

    def finaleval(self):
        print("prepping bert")
        #load weights of best model
        
        path = 'saved_weights.pt'
        self.gpumodel.load_state_dict(torch.load(path))
        
        # get predictions for test data
        with torch.no_grad():
          preds = self.gpumodel(self.test_seq.to(self.device), self.test_mask.to(self.device))
          preds = preds.detach().cpu().numpy()
          
        preds = np.argmax(preds, axis = 1)
        print(classification_report(self.test_y, preds))

        cm = confusion_matrix(self.test_y, preds, labels=[1,0])
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")
    
        ax.set_title('Confusion Matrix')
    
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
    
        ax.xaxis.set_ticklabels(['IF', 'Non-IF'])
        ax.yaxis.set_ticklabels(['IF', 'Non-IF'])

    def finaltest(self):
        print("prepping bert")
        #load weights of best model
        
        path = 'saved_weights.pt'
        self.gpumodel.load_state_dict(torch.load(path))
        
        # get predictions for test data
        with torch.no_grad():
          preds = self.gpumodel(self.test_seq.to(self.device), self.test_mask.to(self.device))
          preds = preds.detach().cpu().numpy()
          
        print(np.argmax(preds, axis = 1))
        #fout = open(r"C:\Users\User\Documents\!Local\NLP\rawdata\20201023_IFDetect_out.csv","w+")
        fout = open(r"/users/felix/FRA_UAS/master/DHL_Document_Gap_Detector/scrapeweb/scripts/20201106_out.csv","w+")
        for i in range(0, len(self.test_text)):
            try:
                fout.write('"%s";%s;%s\n' % (self.test_text[i], 
                                             '{0:.2f}'.format(preds[i][0]).replace(".",","),
                                             '{0:.2f}'.format(preds[i][1]).replace(".",",")))
            except:
                pass
        fout.close()

def completetraining():
    print("create class")
    x = DoTrainBert()
    print("load data")
    x.load_data() # training data
    print("load model")
    x.load_model() # load pretrained model 'bert-base-uncased'
    print("tokenize")
    x.tokenize_and_tensors()
    print("build tensor datasets")
    x.build_tensordatasets()
    print("prepping bert")
    x.prep_bert()
    print("training")
    x.final()

def testing():
    print("create class")
    x = DoTrainBert()
    print("load data")
    x.load_data()
    print("load model and build custom bert")
    x.load_model()
    x.prep_bert()
    print("load model and build custom bert")
    x.tokenize_and_tensors()
    print("finaleval")
    x.finaleval()

def realdata():
    print("create class")
    x = DoTrainBert()
    print("load data")
    x.load_realdata()
    print("load model and build custom bert")
    x.load_model()
    x.prep_bert(False)
    print("load model and build custom bert")
    x.tokenize_realdata()
    print("finaleval")
    x.finaltest()

#completetraining()    
#testing()
realdata() # Evaluation mit einem Evaluations-Dokument aus dem Internet 

# Die geeigneten Daten (Training und Test) stammen aus 
# 20201002_Interface_extraction_v2.xlsx und
# 20201005_if_ont.xlsx, zusammengefasst wurden die Dateien in
# 20202310_Pretraining_test
