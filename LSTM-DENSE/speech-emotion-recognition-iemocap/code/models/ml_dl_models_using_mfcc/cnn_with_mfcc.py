from __future__ import print_function
import numpy as np 
import pandas as pd 
import argparse
import pickle
import random
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import warnings
import os
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix
import wandb

import torch 
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import librosa
import librosa.display
import regex as re
from torchvision.io import read_image
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print('Device being used is {}'.format(device))

classes = {'angry':0, 'happiness':1, 'excited':2, 'neutral':3} 

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.mfccs = dataset['mfccs']
        self.labels = dataset['label']
        self.filename = dataset['wav_file']

    def __len__(self):
        return len(self.mfccs)

    def __getitem__(self, idx):
        return torch.tensor(self.mfccs[idx]), torch.tensor(self.labels[idx]), self.filename[idx]
    
def collate_fn(batch):
    # Determine the maximum length of the spectrograms in the batch
    max_len = max(mfcc.shape[1] for mfcc, label, filename in batch)

    # Pad the spectrograms with zeros to the maximum length
    padded_mfcc = []
    for mfcc, label, filename in batch:
        num_cols = mfcc.shape[1]
        padding = torch.zeros((13, max_len - num_cols))
        padded_mfcc.append(torch.cat([mfcc, padding], dim=1))

     # Concatenate the padded spectrograms into a tensor
    mfcc_tensor = torch.stack(padded_mfcc, dim=0)

     # Convert the labels to PyTorch tensor
    labels_tensor = torch.tensor([label for mfcc, label, filename in batch])

     # Create a list of filenames
    filenames_list = [filename for spec, label, filename in batch]

    return mfcc_tensor, labels_tensor, filenames_list



def load_data(args, batch_size=64):
    dataset = pickle.load(open('LSTM-DENSE/speech-emotion-recognition-iemocap/preprocess_info/feature_vectors.pkl','rb'))
    
    spectograms = []
    labels = []
    filenames = []
    mfccs = []
    print("Loading Dataset . . .")
    for i in range(len(dataset['label'])):
        if dataset['label'][i] in [0,1,3,7]:
            label = 0
            if dataset['label'][i]==3:
                label = 2
            elif dataset['label'][i]==7:
                label =3
            elif dataset['label'][i]==1:
                label=1
            spectograms.append(dataset['spec_db'][i])
            labels.append(label)
            filenames.append(dataset['wav_file'][i])
            mfccs.append(dataset['mfccs'][i])
            
    dataset['label'] = labels
    dataset['spec_db'] = spectograms
    dataset['mfccs'] = mfccs
    dataset['wav_file'] = filenames

    emotion_full_dict = {0:'angry', 1:'happiness', 2:'excited', 3:'neutral'}
    
    # print('Number of data points: {}'.format(len(dataset['label'])))
    #print('Number of data points after filtering: {}'.format(len(filtered_dataset['label'])))
    #breakpoint()
    train_ratio = 0.8

    # Calculate the number of data points in the train and test sets
    num_train = int(len(dataset['label']) * train_ratio)
    num_test = len(dataset['label']) - num_train

    # Create a list of indices for the train and test sets
    indices = list(range(len(dataset['label'])))
    random.shuffle(indices)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    # Split the data into train and test sets
    train_dataset = {}
    test_dataset = {}
    for key in dataset:
        train_dataset[key] = [dataset[key][i] for i in train_indices]
        test_dataset[key] = [dataset[key][i] for i in test_indices]

    train_dataset = CustomDataset(train_dataset)
    test_dataset = CustomDataset(test_dataset)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    
    for i, (inputs, labels, filenames) in enumerate(trainloader):
        # print(inputs.shape)
        # print(labels)
        # print('dataloader is working')
        break
    #breakpoint()
    #print('Training batches are {} and examples are {}'.format(len(trainloader), len(trainloader.dataset)))
    #print('Validation batches are {} and examples are {}'.format(len(valloader), len(valloader.dataset)))
    
    print("Dataset Loaded . . .")
    
    return trainloader, testloader

def train(epoch, model, trainloader, criterion, optimizer):
    model.train()
    correct_train = 0
    train_loss = 0
    train_acc = 0
    
    for batch_idx, (data, target,filename) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        # print(data.shape)
        
        # zero the gradient, forward, backward and running pytorch rhythm
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        # get the label of prediction
        pred = torch.max(output.data, 1)[1]
        correct_train += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()))
            # 
    train_loss /= len(trainloader.dataset)
    train_acc = 100. * correct_train / len(trainloader.dataset)
    return train_loss, int(train_acc.numpy())


def test(model, valloader, criterion):
    model.eval()
    test_loss = 0
    test_acc = 0
    correct = 0
    history_test = []

    pred_model = []
    actual = []

    for data, target,filename in valloader:
        data, target = data.to(device), target.to(device)

        # output from model
        output = model(data)

        # sum total loss
        test_loss += criterion(output, target).item()

        # get the label of prediction
        pred = torch.max(output.data, 1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        pred_model.append(pred.cpu().numpy())
        actual.append(target.data.cpu().numpy())


    test_loss /= len(valloader.dataset)
    test_acc = 100. * correct / len(valloader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(valloader.dataset),
    #     100. * correct / len(valloader.dataset)))


    #pred_with_label = [label_to_class[label] for label in list(np.concatenate(pred_model))]
    #actual_with_label = [label_to_class[label] for label in list(np.concatenate(actual))]

    #confusion_matrix(actual_with_label, pred_with_label, labels=final_labels)

    #print('\n Classification Report \n {} \n'.format(classification_report(actual_with_label, pred_with_label)))

    return test_loss, int(test_acc.numpy())
            





keep_prob =0.5
class Crude_Diag(nn.Module):
    def __init__(self, in_features):
        super(Crude_Diag, self).__init__()
        
        # Set a fixed seed for reproducibility
        torch.manual_seed(0)
        
        # Initialize scaling factors and weight matrix
        scaling_factors = torch.rand(in_features)
        weight = torch.diag(scaling_factors)
        
        # Modify weight matrix to set non-diagonal elements to zero
        with torch.no_grad():
            for i in range(in_features):
                for j in range(in_features):
                    if i != j:
                        weight[i, j] = 0.0
                        weight[i, j].requires_grad = False
        
        # Define linear layer with diagonal weight matrix
        self.linear = nn.Linear(in_features, in_features, bias=False)
        self.linear.weight = nn.Parameter(weight)
        
    def forward(self, x):
        x = self.linear(x)
        return x

class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=(1,3), stride=(1,2)),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=(1,3), stride=(1,2)),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=(1,3), stride=(1,2)),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=(1,2), stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        self.fc1 = torch.nn.Linear(768, 384, bias=True)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.pool =  torch.nn.AdaptiveAvgPool2d((1,1))

        self.lstm = nn.LSTM(input_size=384, hidden_size=256, num_layers=1)
        
        # Affine Layer
        self.fc2 = torch.nn.Linear(256, 50, bias=True)  #384 earlier
        self.Diag_Affine = Crude_Diag(50)
        self.fc3 = torch.nn.Linear(50, 4, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x= x.reshape(x.shape[0],1,x.shape[1],x.shape[2])
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.mean(out, axis=-1).reshape(out.shape[0],128*6)
        out = self.fc1(out)
        out = self.dropout(out)

        # Pass input through LSTM layer
        out, _ = self.lstm(out)
        
        out = self.fc2(out)

        out = self.Diag_Affine(out)
        out = self.fc3(out)
        return out
    
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 
    

def main(args):
    
    config = {
        "learning_rate": 0.001,
        "batch_size": 69,
        "epochs": 500,
        "architecture": "CNN",
    }

    trainloader, valloader = load_data(args, config['batch_size'])
    #print('Validation session is {}'.format(val_ses))
    # print('Training batches are {} and examples are {}'.format(len(trainloader), len(trainloader.dataset)))
    # print('Validation batches are {} and examples are {}'.format(len(valloader), len(valloader.dataset)))
    # sys.exit()

    model = CNN()
    model.to(device)



    # print("network before turning off sparse layer", count_parameters(model))

    for param in model.Diag_Affine.parameters():
        param.requires_grad = False     
    # print("network after turning off sparse layer", count_parameters(model))




    #print(trainloader.dataset.class_to_idx)
    anger = 0
    happiness = 0
    neutral = 0
    sadnes = 0
    max_length  = float('-inf')
    min_length  = float('inf')
    avg_length =0
    for i, (inputs, labels,filename) in enumerate(trainloader):
        labels = list(labels.numpy())
        anger += labels.count(0)
        happiness += labels.count(1)
        neutral += labels.count(3)
        sadnes += labels.count(2)
        if inputs.shape[-1]>max_length:
            max_length = inputs.shape[-1]
        if inputs.shape[-1]<min_length: 
            min_length = inputs.shape[-1]
        avg_length +=inputs.shape[-1]
    for i, (inputs, labels,filename) in enumerate(valloader):
        labels = list(labels.numpy())
        anger += labels.count(0)
        happiness += labels.count(1)
        neutral += labels.count(3)
        sadnes += labels.count(2)
        
    # avg_length/=len(trainloader)
    print('Anger: {}, Happiness: {}, Neutral: {}, Sadness: {}'.format(anger, happiness, neutral, sadnes))
    sample_weights = [1/anger, 1/happiness, 1/neutral, 1/sadnes]
    class_weights = torch.FloatTensor(sample_weights).cuda()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # wandb.login(key="a94b61c6268e685bc180a0634fae8dc030cd8ed4") #API Key is in your wandb account, under settings (wandb.ai/settings)

    # Create your wandb run
    wandb.login(key="207bf381f197203155b01dd8b56293d02aca5365") 
    run = wandb.init(
    name    = "IEMOCAP baseline-1", 
    reinit  = True, 
    project = "Domain Adaption",  
    config  = config ### Wandb Config for your run

)

    history, n_epoch = [], config["epochs"]
    best_val_acc = 0

    for epoch in range(1, n_epoch):    
        # exp_lr_scheduler.step(epoch)
        # import pdb
        # pdb.set_trace()
        train_loss, train_acc = train(epoch, model, trainloader, criterion, optimizer)
        test_loss, test_acc = test(model, valloader, criterion)
        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc})
        if test_acc > best_val_acc:
            best_val_acc = test_acc
            torch.save(model.state_dict(), 'best_model_session.pt')



        # plateau_scheduler.step(test_loss)
        history.append([train_loss, train_acc, test_loss, test_acc])

    run.finish()

    history_df = pd.DataFrame(history, columns=["train_loss", "train_acc", "test_loss", "test_acc"])
    history_df["epoch"] = [x for x in range(1, n_epoch)]
    # print(history_df)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='LSTM-DENSE/speech-emotion-recognition-iemocap/preprocess_info', help='path to dataset')
    args = parser.parse_args()
    
    main(args)
    


