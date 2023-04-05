import torch.nn as nn
import torch
class ConvNet(nn.Module):
    def __init__(self,config):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(13, 64, 5, padding=2)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 5, padding=2)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 384, bias=True)
        nn.init.xavier_uniform(self.fc1.weight)
        self.pool = nn.AdaptiveMaxPool2d((1,1))
        #self.lstm = nn.LSTM(input_size=384, hidden_size=256, num_layers=1)
        
        ## Affine Layer
        self.fc2 = nn.Linear(384, config['Transformation_Matrix'], bias=True)  #384 earlier
        #self.Diag_Affine = Crude_Diag(config['Transformation_Matrix'])
        #self.Full_Affine = torch.nn.Linear(config['Transformation_Matrix'], config['Transformation_Matrix'])
        self.fc3 = nn.Linear(config['Transformation_Matrix'], 4, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters
        nn.init.xavier_uniform_(self.fc3.weight) # initialize parameters
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        # x = self.dropout1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        # x = self.dropout2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        # x = self.dropout3(x)
        x = torch.mean(x, axis=-1).reshape(x.shape[0], 256)
        x = self.fc1(x)
        #out = self.dropout(out)

        # Pass input through LSTM layer
        #out, _ = self.lstm(out)
        
        x = self.fc2(x)
        
        #out = self.Diag_Affine(out)
        #out = self.Full_Affine(out)
        x = self.fc3(x)

        return x