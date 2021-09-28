import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import is_available
from torch.nn.init import xavier_uniform_

class MLPDisaggregator(nn.Module):

    def __init__(self, hidden_layers, output_size=2):
        super(MLPDisaggregator, self).__init__()
        
        self.device = 'cuda' if is_available() else 'cpu'
        
        self.n_layers = len(hidden_layers)
        hidden_layers.append(output_size)

        self.fcl = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()
        
        for l in range(self.n_layers):
            self.fcl.append(
                nn.Linear(
                    in_features=hidden_layers[l],
                    out_features=hidden_layers[l + 1]
                    )
            )
            xavier_uniform_(self.fcl[l].weight)
    
    def forward(self, X):

        for l in range(self.n_layers - 1):
            X = self.fcl[l](X)
            X = self.sigmoid(X)

        return self.fcl[self.n_layers - 1](X)
    
    def predict(self, x):
        self.eval() 
        x_tensor = torch.as_tensor(x).float()
        y_hat_tensor = self(x_tensor.to(self.device))
        self.train()
        pv_pr = y_hat_tensor[:,1].detach().cpu().numpy()
        
        for i in range(len(pv_pr)):
            if pv_pr[i] < .005:
                pv_pr[i] = 0.0
        
        load_pr = y_hat_tensor[:,0].detach().cpu().numpy()
        return pv_pr, load_pr
    
    