import torch
import torch.nn as nn
import torch.optim as optim

class BClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BClassifier, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.layer6 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer4(x)
        return x
    

class PanDataset(torch.utils.data.Dataset):
    def __init__(self, days, pids, data, labels):
        self.days = days
        self.pids = pids
        self.data = data
        self.labels = labels
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx,:]), self.labels[self.pids[idx]][self.days[idx]]
    

class PanDatasetTest(torch.utils.data.Dataset):
    def __init__(, days, pids, data, labels):
        
        self.labels = labels
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx,:]), self.labels[]
