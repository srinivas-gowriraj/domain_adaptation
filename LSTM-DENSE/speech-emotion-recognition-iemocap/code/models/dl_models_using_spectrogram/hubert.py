import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from datasets import load_dataset, concatenate_datasets
import numpy as np
from tqdm import tqdm


# Load the pre-trained Hubert model and its tokenizer
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/hubert-large-ls960-ft")
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
    

# Define the function to preprocess the audio inputs
def preprocess(inputs, outputs):
    input_features = []
    for audio_file in tqdm(inputs):
        input, sr = torchaudio.load(audio_file)
        input_feature = processor(input, sampling_rate=sr, return_tensors="pt", padding=True, truncation=True)
        input_features.append(input_feature.input_values[0])
    input_features = torch.stack(input_features)
    return input_features, torch.tensor(outputs)

# Apply the preprocessing function to the dataset
train_inputs, train_outputs = preprocess(train_dataset["audio"], train_dataset["label"])
train_dataset = torch.utils.data.TensorDataset(train_inputs, train_outputs)

val_inputs, val_outputs = preprocess(val_dataset["audio"], val_dataset["label"])
val_dataset = torch.utils.data.TensorDataset(val_inputs, val_outputs)

# Define the data loaders for the training and validation sets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

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
