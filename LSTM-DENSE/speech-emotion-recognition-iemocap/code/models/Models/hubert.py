import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, DataCollatorWithPadding, AutoProcessor, HubertForSequenceClassification
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import torchaudio
from numpy import save
import os
from transformers import AutoProcessor, Wav2Vec2Model
import gc
import subprocess


# Load the pre-trained Hubert model and its tokenizer
model =  HubertForSequenceClassification.from_pretrained("facebook/hubert-large-ls960-ft", num_labels=4)
processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(device)
# Load the RAVDESS dataset
class Hubert(nn.Module):
    def __init__(self,config):
        super(Hubert, self).__init__()
        self.hubert =  Wav2Vec2Model.from_pretrained("facebook/hubert-large-ls960-ft")

        self.fc1 = nn.Linear(768, 384, bias=True)
        nn.init.xavier_uniform(self.fc1.weight)
        self.pool = nn.AdaptiveMaxPool2d((1,1))
        #self.lstm = nn.LSTM(input_size=384, hidden_size=256, num_layers=1)
        
        ## Affine Layer
        self.fc2 = nn.Linear(384, 50, bias=True)  #384 earlier
        #self.Diag_Affine = Crude_Diag(config['Transformation_Matrix'])
        #self.Full_Affine = torch.nn.Linear(config['Transformation_Matrix'], config['Transformation_Matrix'])
        self.fc3 = nn.Linear(50, 4, bias=True)
        # nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters
        # nn.init.xavier_uniform_(self.fc3.weight) # initialize parameters
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        x = self.hubert(x)
        x = x.last_hidden_state
        x = self.fc1(x)
        #out = self.dropout(out)
        # Pass input through LSTM layer
        #out, _ = self.lstm(out)
        x = self.fc2(x)
        #out = self.Diag_Affine(out)
        #out = self.Full_Affine(out)
        x = self.fc3(x)
        return x
    
    
dataset = load_dataset("superb", "er", data_dir="/home/sgowrira/domain_adaptation/IEMOCAP_full_release")
#breakpoint()


session1 = load_dataset("superb", "er", split='session1', data_dir="/home/sgowrira/domain_adaptation/IEMOCAP_full_release")
session2 = load_dataset("superb", "er", split='session2', data_dir="/home/sgowrira/domain_adaptation/IEMOCAP_full_release")
session3 = load_dataset("superb", "er", split='session3', data_dir="/home/sgowrira/domain_adaptation/IEMOCAP_full_release")
session4 = load_dataset("superb", "er", split='session4', data_dir="/home/sgowrira/domain_adaptation/IEMOCAP_full_release")
session5 = load_dataset("superb", "er", split='session5', data_dir="/home/sgowrira/domain_adaptation/IEMOCAP_full_release")

train_dataset = concatenate_datasets([session1,session2,session3,session4])
#train_dataset = session1
val_dataset = session5

def create_array_col(dataset):
    for i in range(0,len(dataset)):
        dataset[i]["array"] = dataset[i]['audio']['array'].numpy()
    return dataset

def tokenize_function(example):
    input_feature = processor(example['audio'][0]['array'], sampling_rate=16000, return_tensors="pt", padding="longest") 
    return input_feature

def calculate_memory():
    command = "nvidia-smi | grep MiB | awk '{print $9 $10 $11}'"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    output = result.stdout.decode().strip()

    print(output)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=1, remove_columns=["audio", "file"])
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, batch_size=1, remove_columns=["audio", "file"])
data_collator = DataCollatorWithPadding(tokenizer=processor)

train_dataloader = DataLoader(
    tokenized_train_dataset, shuffle=True, batch_size=4, collate_fn=data_collator
)

val_dataloader = DataLoader(
    tokenized_val_dataset, shuffle=True, batch_size=4, collate_fn=data_collator
)
print("train_datalaoder: ", len(train_dataloader))
for i, batch in enumerate(train_dataloader):
    print(batch['input_values'].shape)
    print(batch['attention_mask'].shape)
    print(batch['labels'].shape)
    #breakpoint()
    break

#breakpoint()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# print('check 1')
# !nvidia-smi | grep MiB | awk '{print $9 $10 $11}'
# Fine-tune the model on the dataset
#print('memory before training: ')
#calculate_memory()
gc.collect()
num_epochs = 20
for epoch in range(num_epochs):
    # Training loop
    model.train()
    train_loss = 0.0
    for batch_idx, (inputs) in tqdm(enumerate(train_dataloader)):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(input_values=inputs.input_values, attention_mask=inputs.attention_mask)
        #breakpoint()
        loss = criterion(outputs.logits, inputs.labels)
        if batch_idx%50==0:
            print("Train_loss: ",loss)
        #breakpoint()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        del inputs, loss, outputs
        torch.cuda.empty_cache()
        #print("memory after each batch")
        #calculate_memory()
        # print(torch.cuda.memory_summary(device=device, abbreviated=False))

        # del(inputs)
        
    # Validation loop
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for batch_idx, inputs in tqdm(enumerate(val_dataloader)):
            inputs = inputs.to(device)
            outputs = model(input_values=inputs.input_values, attention_mask=inputs.attention_mask)
            loss = criterion(outputs.logits, inputs.labels)
            val_loss += loss.item()
            preds = np.argmax(outputs.logits.detach().cpu().numpy(), axis=1)
            val_acc += np.sum(preds == inputs.labels.cpu().numpy())
            del inputs, loss,outputs
            torch.cuda.empty_cache()
            # print(torch.cuda.memory_summary(device=device, abbreviated=False))
            
    
    # Print the training and validation loss and accuracy for each epoch
    train_loss /= len(train_dataloader)
    val_loss /= len(val_dataloader)
    val_acc /= len(val_dataset)
    print("Epoch {}: Train Loss = {:.3f}, Val Loss = {:.3f}, Val Acc = {:.3f}".format(epoch+1, train_loss, val_loss, val_acc))

