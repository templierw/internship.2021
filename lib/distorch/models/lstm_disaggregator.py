import torch
from torch.cuda import is_available
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

class LSTMDisaggregator(nn.Module):

    def __init__(
        self,
        n_features,
        hidden_dim,
        n_outputs
    ):
        super(LSTMDisaggregator, self).__init__()
        
        self.device = 'cuda' if is_available() else 'cpu'

        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_outputs = n_outputs
        self.hidden = None
        self.cell = None
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.lstm = nn.LSTM(
            self.n_features, 
            self.hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.fc1 = nn.Linear(
            self.hidden_dim, self.hidden_dim * 4
            )
        xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(
            self.hidden_dim * 4, self.hidden_dim * 2
            )
        xavier_uniform_(self.fc2.weight)
        
        self.fc3 = nn.Linear(
            self.hidden_dim, self.n_outputs
            )
        xavier_uniform_(self.fc3.weight)
    
    def forward(self, X):
        #X is batch first (NLF)
        #output is (NLH)
        #final hidden is (2NH)
        X = X.reshape(-1, len(X), self.n_features)#.to(self.device)
        bfo, (self.hidden, self.cell) = self.lstm(X)

        #out = self.sigmoid(self.fc1(bfo[-1]))
        #out = self.sigmoid(self.fc2(out))

        out = self.fc3(bfo[-1])#out)

        return out#.view(-1, self.n_outputs)
    
    def predict(self, x):
        self.eval() 
        x_tensor = torch.as_tensor(x).float()
        y_hat_tensor = self(x_tensor.to(self.device))
        self.train()
        pv_pr = y_hat_tensor[:,1].detach().cpu().numpy()
        load_pr = y_hat_tensor[:,0].detach().cpu().numpy()
        return pv_pr, load_pr