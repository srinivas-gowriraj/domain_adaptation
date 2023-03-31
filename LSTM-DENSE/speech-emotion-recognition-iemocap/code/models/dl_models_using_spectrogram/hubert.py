import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, DataCollatorWithPadding
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import torchaudio
from numpy import save
import os

# Load the pre-trained Hubert model and its tokenizer
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/hubert-large-ls960-ft", num_labels=4)
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

# Load the RAVDESS dataset

dataset = load_dataset("superb", "er", data_dir="/home/sgowrira/domain_adaptation/IEMOCAP_full_release")
#breakpoint()


session1 = load_dataset("superb", "er", split='session1', data_dir="/home/sgowrira/domain_adaptation/IEMOCAP_full_release")
session2 = load_dataset("superb", "er", split='session2', data_dir="/home/sgowrira/domain_adaptation/IEMOCAP_full_release")
session3 = load_dataset("superb", "er", split='session3', data_dir="/home/sgowrira/domain_adaptation/IEMOCAP_full_release")
session4 = load_dataset("superb", "er", split='session4', data_dir="/home/sgowrira/domain_adaptation/IEMOCAP_full_release")
session5 = load_dataset("superb", "er", split='session5', data_dir="/home/sgowrira/domain_adaptation/IEMOCAP_full_release")

train_dataset = concatenate_datasets([session1,session2,session3,session4])
val_dataset = session5

def create_array_col(dataset):
    for i in range(0,len(dataset)):
        dataset[i]["array"] = dataset[i]['audio']['array'].numpy()
    return dataset

def tokenize_function(example):
    input_feature = processor(example['audio'][0]['array'], sampling_rate=16000, return_tensors="pt", padding="longest") 
    return input_feature

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=1, remove_columns=["audio", "file"])
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, batch_size=1, remove_columns=["audio", "file"])
data_collator = DataCollatorWithPadding(tokenizer=processor)

train_dataloader = DataLoader(
    tokenized_train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator
)

val_dataloader = DataLoader(
    tokenized_val_dataset, shuffle=True, batch_size=8, collate_fn=data_collator
)

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

# Fine-tune the model on the dataset
num_epochs = 1
for epoch in range(num_epochs):
    # Training loop
    model.train()
    train_loss = 0.0
    for batch_idx, (inputs) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(input_values=inputs.input_values, attention_mask=inputs.attention_mask)
        #breakpoint()
        loss = criterion(outputs.logits, inputs.labels)
        #breakpoint()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    # Validation loop
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for batch_idx, inputs in enumerate(val_dataloader):
            outputs = model(input_values=inputs.input_values, attention_mask=inputs.attention_mask)
            loss = criterion(outputs.logits, inputs.labels)
            val_loss += loss.item()
            preds = np.argmax(outputs.logits.detach().numpy(), axis=1)
            val_acc += np.sum(preds == inputs.labels.numpy())
    
    # Print the training and validation loss and accuracy for each epoch
    train_loss /= len(train_dataloader)
    val_loss /= len(val_dataloader)
    val_acc /= len(val_dataset)
    print("Epoch {}: Train Loss = {:.3f}, Val Loss = {:.3f}, Val Acc = {:.3f}".format(epoch+1, train_loss, val_loss, val_acc))
