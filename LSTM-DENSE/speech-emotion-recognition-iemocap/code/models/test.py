# class DiagonalLinear(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(DiagonalLinear, self).__init__()
#         self.weight = nn.Parameter(torch.ones(output_size, input_size))
#         # Set non-diagonal elements to be constant
#         with torch.no_grad():
#             self.weight.fill_diagonal_(1)
#             for i in range(input_size):
#                 for j in range(output_size):
#                     if i != j:
#                         self.weight[j, i] = 0.0
#                         self.weight[j, i].requires_grad = False

#     def forward(self, x):
#         return torch.matmul(x,self.weight) 



# Define the scaling network
import torch
import torch.nn as nn
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
        # L1 ImgIn shape=(?, 224, 224, 3)
        #    Conv     -> (?, 224, 224, 16)
        #    Pool     -> (?, 112, 112, 16)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L2 ImgIn shape=(?, 112, 112, 16)
        #    Conv      ->(?, 112, 112, 32)
        #    Pool      ->(?, 56, 56, 32)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L3 ImgIn shape=(?, 56, 56, 32)
        #    Conv      ->(?, 56, 56, 64)
        #    Pool      ->(?, 28, 28, 64)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        
        # L4 ImgIn shape=(?, 28, 28, 64)
        #    Conv      ->(?, 28, 28, 16)
        #    Pool      ->(?, 14, 14, 16)
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 16, kernel_size=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))

        # L4 FC 14x14x16 inputs -> 512 outputs
        self.fc1 = torch.nn.Linear(512, 512, bias=True)
        torch.nn.init.xavier_uniform(self.fc1.weight)
#         self.layer4 = torch.nn.Sequential(
#             self.fc1,
#             torch.nn.ReLU(),
#             torch.nn.Dropout(p=1 - keep_prob))
        # L5 Final FC 1024 inputs -> 512 outputs
        self.pool =  torch.nn.AdaptiveAvgPool2d((1,1))
        # Affine Layer
        self.Diag_Affine = Crude_Diag(512)
        # self.test_linear = torch.nn.Linear(512,512)
        self.fc2 = torch.nn.Linear(512, 4, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters
        # L6 Final FC 512 inputs -> 4 outputs
#         self.fc3 = torch.nn.Linear(512, 4, bias=True)
#         torch.nn.init.xavier_uniform_(self.fc3.weight) # initialize parameters

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x= x.reshape(x.shape[0],1,x.shape[1],x.shape[2])
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
#         print(out.size())
        # out = out.view(out.size(0), -1)
        # out = out.mean(axis = -1)
        out = torch.mean(out, axis=-1).reshape(-1,512)
#         print(out.size())# Flatten them for FC
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.Diag_Affine(out)
        # out = self.test_linear(out)
        out = self.fc2(out)
#         out = self.fc3(out)
        return out
    
if __name__=="__main__":
    model = CNN()
    input = torch.randn(64,13,33)
    out = model(input)
    print(out)