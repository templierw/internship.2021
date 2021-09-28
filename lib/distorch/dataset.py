import torch
from torch.cuda import is_available
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split

import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler

class TaskDataset(Dataset):
    def __init__(self):

        self.device = 'cuda' if is_available() else 'cpu'
        self.dataset_created = False

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

    def __len__(self):
        return len(self.X)

        return train_loader, val_loader

    def load_dataset(self, filename):
        return pd.read_csv(
            f"{os.environ['DATA_DIR']}/{filename}.csv", index_col=0, parse_dates=True, dtype=float
        )

    def cyclic_transform(self, frame, scale):
        temp = ((2 * np.pi / scale) * frame)
        return np.sin(temp), np.cos(temp)

    def preprocess_datetime(self):
        self.df['month_x'] = 0.0
        self.df['month_y'] = 0.0
        self.df['day_x'] = 0.0
        self.df['day_y'] = 0.0
        self.df['hour_x'] = 0.0
        self.df['hour_y'] = 0.0

        for i in range(0, len(self.df)):
            self.df['month_x'][i], self.df['month_y'][i] = \
                 self.cyclic_transform(self.df.index[i].month - 1,12)
            self.df['day_x'][i], self.df['day_y'][i] = \
                 self.cyclic_transform(int(self.df.index[i].strftime('%w')), 7)
            self.df['hour_x'][i], self.df['hour_y'][i] = \
                 self.cyclic_transform(self.df.index[i].hour, 24)
            
    def get_train_val_set(self,sequence=False):

        if sequence:
            X_train, y_train = self.create_sequences()

        self.train_set = TensorDataset(self.X_t, self.y_t)
        self.val_set = TensorDataset(self.X_v, self.y_v)

        self.dataset_created = True

        return self.train_set, self.val_set
    
    def get_dataloader(self, batch_size):

        if not self.dataset_created:
            self.get_train_val_set()

        train_loader = DataLoader(
            dataset=self.train_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )

        val_loader = DataLoader(
            dataset=self.val_set, batch_size=batch_size
        )
        
        return train_loader, val_loader


class DisaggregationDataset(TaskDataset):

    def __init__(self, months, dataset='disaggregation', preprocess=False):
        super().__init__()

        if not os.path.isfile(f"{os.environ['DATA_DIR']}/{dataset}_preprocset.csv") and not preprocess:
            print("Preprocessed data unavailable locally - preprocessing...")
            preprocess = True

        if preprocess:
            self.df = self.load_dataset(f'{dataset}_set')
            self.preprocess_datetime()
            self.df.to_csv(f"{os.environ['DATA_DIR']}/{dataset}_preprocset.csv")
        
        else:
            self.df = self.load_dataset(f'{dataset}_preprocset')

        mask = self.df.index.month.isin(months)
            
        train_df = self.df[mask]
        val_df = self.df[~mask]

        self.train_index = train_df.index
        self.val_index = val_df.index

        self.index = self.df.index
        
        X_t = train_df.drop(columns=['load', 'pv']).values
        X_v = val_df.drop(columns=['load', 'pv']).values
        y_t = train_df[['load', 'pv']].values
        y_v = val_df[['load', 'pv']].values

        self.scaler = StandardScaler(with_mean=True, with_std=True)
        X_ts = self.scaler.fit_transform(X_t)
        X_vs = self.scaler.transform(X_v)

        self.X_t, self.y_t = torch.as_tensor(X_ts).float().to(self.device), \
                             torch.as_tensor(y_t).float().to(self.device)
        self.X_v, self.y_v = torch.as_tensor(X_vs).float().to(self.device), \
                             torch.as_tensor(y_v).float().to(self.device)


class ForecastingDataset(TaskDataset):

    def __init__(self, prediction_length, months, dataset='forecasting', preprocess=False):
        super().__init__()

        self.pred_len = prediction_length

        if not os.path.isfile(f"{os.environ['DATA_DIR']}/{dataset}_preprocset.csv") and not preprocess:
            
            print("Preprocessed data unavailable locally - preprocessing...")
            preprocess = True

        if preprocess:
            self.df = self.load_dataset(f'{dataset}_set')
            self.preprocess_datetime()
            self.compute_lag(lag=12)
            self.df.to_csv(f"{os.environ['DATA_DIR']}/{dataset}_preprocset.csv")
        
        else:
            self.df = self.load_dataset(f'{dataset}_preprocset')
        
        mask = self.df.index.month.isin(months)
            
        train_df = self.df[mask]
        val_df = self.df[~mask]

        self.train_index = train_df.index
        self.val_index = val_df.index
        
        X_t = train_df.drop(columns=['netload']).values
        X_v = val_df.drop(columns=['netload']).values
        y_t = train_df.netload.ewm(span=8).mean().values
        y_v = val_df.netload.ewm(span=8).mean().values

        self.scaler = StandardScaler(with_mean=True, with_std=True)
        X_ts = self.scaler.fit_transform(X_t)
        X_vs = self.scaler.transform(X_v)

        self.X_t, self.y_t = torch.as_tensor(X_ts).float().to(self.device), \
                             torch.as_tensor(y_t).float().to(self.device)
        self.X_v, self.y_v = torch.as_tensor(X_vs).float().to(self.device), \
                             torch.as_tensor(y_v).float().to(self.device)
        
        print(self.X_t.shape, self.y_t.shape) 

    def compute_lag(self, lag=12):
        for l in range(1, lag + 1):
            col = f'LoadTm{l}'
            self.df[col] = 0.0
            for i in range(0, len(self.df)):
                self.df[col][i] = self.df.loc[:,'netload'][i - l]

    def create_sequences(self):
        xs = []
        ys = []
        for i in range(len(self.X_t)-self.pred_len-1):
            xs.append(self.X_t[i:(i+self.pred_len),:])
            ys.append(self.y_t[i+self.pred_len])
            
        return torch.as_tensor(xs).float().to(self.device), \
               torch.as_tensor(ys).float().to(self.device)