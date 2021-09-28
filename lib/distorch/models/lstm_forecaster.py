import torch
from torch.cuda import is_available
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.autograd import Variable

class LSTMForecaster(nn.Module):

    def __init__(
        self,
        pred_len,
        n_features,
        hidden_dim,
        n_layers,
        n_outputs
    ):
        super(LSTMForecaster, self).__init__()
        
        self.device = 'cuda' if is_available() else 'cpu'

        self.pred_len = pred_len

        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.activation = nn.Sigmoid()

        self.lstm = nn.LSTM(
            self.n_features, 
            self.hidden_dim,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=0.5
        )
        self.fc1 = nn.Linear(
            self.hidden_dim, self.hidden_dim
            )
        xavier_uniform_(self.fc1.weight)
        
        self.fc2 = nn.Linear(
            self.hidden_dim, self.hidden_dim
            )
        xavier_uniform_(self.fc2.weight)
        
        self.fc3 = nn.Linear(
            self.hidden_dim, self.hidden_dim
            )
        xavier_uniform_(self.fc3.weight)
        
        self.fc4 = nn.Linear(
            self.hidden_dim, self.n_outputs
            )
        xavier_uniform_(self.fc3.weight)
        
    def reset_states(self, bs):
        hidden_state = torch.zeros(self.n_layers,bs,self.hidden_dim)
        cell_state = torch.zeros(self.n_layers,bs,self.hidden_dim)
        self.hidden = (hidden_state, cell_state)
    
    def forward(self, x):
        bs=x.size(0)
        self.reset_states(bs=bs)
        # X is batch first (N, L, F)
        # output is (N, L, H)
        # final hidden state is (1, N, H)
        # final cell state is (1, N, H)
        x = x.reshape(1, bs, self.n_features).to(self.device)
        bfo, self.hidden = self.lstm(x)


        out = self.fc1(bfo[-1])
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        out = self.activation(out)
        out = self.fc4(out)
        #print(out.shape)

        return out#.view(-1, self.n_outputs)
    
    def predict(self, x):
        self.eval() 
        x_tensor = torch.as_tensor(x).float()
        y_hat_tensor = self.forward(x_tensor.to(self.device))
        self.train()
        return y_hat_tensor.detach().cpu().numpy()