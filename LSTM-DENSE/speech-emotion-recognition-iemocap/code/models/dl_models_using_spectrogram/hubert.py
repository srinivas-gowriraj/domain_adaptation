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
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/hubert-large-ls960-ft")
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

# Load the RAVDESS dataset

dataset = load_dataset("superb", "er", data_dir="IEMOCAP_full_release")
#breakpoint()


session1 = load_dataset("superb", "er", split='session1', data_dir="IEMOCAP_full_release")
session2 = load_dataset("superb", "er", split='session2', data_dir="IEMOCAP_full_release")
session3 = load_dataset("superb", "er", split='session3', data_dir="IEMOCAP_full_release")
session4 = load_dataset("superb", "er", split='session4', data_dir="IEMOCAP_full_release")
session5 = load_dataset("superb", "er", split='session5', data_dir="IEMOCAP_full_release")

train_dataset = concatenate_datasets([session1,session2,session3,session4])
val_dataset = session5

def tokenize_function(example):
    input, sr = torchaudio.load(example['audio']["path"])
    input_feature = processor(input, sampling_rate=sr, return_tensors="pt", truncation=True) 
    return input_feature

tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=processor)
breakpoint()





    


# Iterate over all audio files in the directory and find the longest one
# for filename in train_dataset["audio"]:
#     filename = filename["path"]
    
#     if filename.endswith(".wav"):
#         file_path = filename
#         info = torchaudio.info(file_path)
#         length = info[0].length
#         if length > max_length:
#             max_length = length

# print("Max length:", max_length)    



def hubert_collate_fn(examples):
    # Sort the examples by descending input length
    examples = sorted(examples, key=lambda x: x["input_values"].shape[0], reverse=True)
    # Get the input lengths
    input_lengths = [x["input_values"].shape[0] for x in examples]
    # Pad the inputs with zeros
    inputs = torch.nn.utils.rnn.pad_sequence([x["input_values"] for x in examples], batch_first=True)
    # Stack the labels
    labels = torch.tensor([x["label"] for x in examples])
    # Return the batch as a dictionary
    return {
        "input_values": inputs,
        "attention_mask": (inputs != 0),
        "labels": labels,
        "input_lengths": input_lengths,
    }

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.audio = []
        audio = dataset["audio"]
        self.labels = dataset['label']
        for audio_file in tqdm(audio):
            input, sr = torchaudio.load(audio_file["path"])
            input_feature = processor(input, sampling_rate=sr, return_tensors="pt", padding=True)
            self.audio.append(input_feature)

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        
        return self.audio[idx], torch.tensor(self.labels[idx])
    
    # def collate_fn(batch):
    #     # Determine the maximum length of the spectrograms in the batch
    #     max_len = max(audio.shape[1] for audio , label in batch)

    #     # Pad the spectrograms with zeros to the maximum length
    #     padded_mfcc = []
    #     for audio , label  in batch:
    #         num_cols = audio.shape[1]
    #         padding = torch.zeros((13, max_len - num_cols))
    #         padded_mfcc.append(torch.cat([audio, padding], dim=1))

    #     # Concatenate the padded spectrograms into a tensor
    #     mfcc_tensor = torch.stack(padded_mfcc, dim=0)

    #     # Convert the labels to PyTorch tensor
    #     labels_tensor = torch.tensor([label for audio,label in batch])

    #     # Create a list of filenames
    #     filenames_list = [filename for spec, label, filename in batch]

    #     return mfcc_tensor, labels_tensor, filenames_list

train_dataset = CustomDataset(train_dataset)

# Define the function to preprocess the audio inputs
# def preprocess(inputs, outputs,max_length,val=False):
#     input_features = []
#     for audio_file in tqdm(inputs):
#         input, sr = torchaudio.load(audio_file["path"])
#         input_feature = processor(input, sampling_rate=sr, return_tensors="pt", padding=True) #, truncation=True,max_length  = max_length)
#         input_features.append(input_feature.input_values[0]) #save it as npz file here, time x frequency  
#     # input_features = torch.stack(input_features)
#     if val:
#         save('LSTM-DENSE/speech-emotion-recognition-iemocap/preprocess_info/hubert/input_features_val.npy', input_features,allow_pickle=True)
#     else:   
#         save('LSTM-DENSE/speech-emotion-recognition-iemocap/preprocess_info/hubert/input_features.npy', input_features,allow_pickle=True)
#     return  torch.tensor(outputs)

# Apply the preprocessing function to the dataset
# train_outputs = preprocess(train_dataset["audio"], train_dataset["label"],max_length)
# train_inputs =  np.load('LSTM-DENSE/speech-emotion-recognition-iemocap/preprocess_info/hubert/input_features.npy',allow_pickle=True)
# # train_inputs= train_inputs.astype(np.float32)
# train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_inputs), train_outputs)

# val_outputs = preprocess(val_dataset["audio"], val_dataset["label"],max_length,val=True)
# val_inputs =  np.load('LSTM-DENSE/speech-emotion-recognition-iemocap/preprocess_info/hubert/input_features_val.npy',allow_pickle=True)

# val_dataset = torch.utils.data.TensorDataset(val_inputs, val_outputs)

    

# Define the data loaders for the training and validation sets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,collate_fn=hubert_collate_fn)

for audio,label in train_loader:
    print(audio.shape,label.shape)
    break


val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,collate_fn=collate_fn)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Fine-tune the model on the dataset
num_epochs = 1
for epoch in range(num_epochs):
    # Training loop
    model.train()
    train_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    # Validation loop
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item()
            preds = np.argmax(outputs.logits.detach().numpy(), axis=1)
            val_acc += np.sum(preds == labels.numpy())
    
    # Print the training and validation loss and accuracy for each epoch
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_acc /= len(val_dataset)
    print("Epoch {}: Train Loss = {:.3f}, Val Loss = {:.3f}, Val Acc = {:.3f}".format(epoch+1, train_loss, val_loss, val_acc))
