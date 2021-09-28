from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import numpy as np
from modelUtils import rmse


class D:

    def __init__(
        self,
        maxPV,
        maxLoad,
        hidden_layer_sizes=31,
        learning_rate_init=0.001,
        alpha=0.05,
        activation='relu',
        random_state=42,
        max_iter=800,
        batch_size=32,
        learning_rate='constant',
        solver='adam'
    ) -> None:

        self.__regressor = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate_init,
            alpha=alpha,
            activation=activation,
            random_state=random_state,
            max_iter=max_iter,
            batch_size=batch_size,
            learning_rate=learning_rate,
            solver=solver
        )

        self.pv = []
        self.load = []
        self.r2 = []
        
        self.maxPV = maxPV
        self.maxLoad = maxLoad

        self.rmse_scores = {
            'pv_mean': 0,
            'pv_std': 0,
            'load_mean': 0,
            'load_std': 0
        }

    def fit(self, X, y):
        self.__regressor.fit(X, y)

    def score(self, X, y):
        return self.__regressor.score(X, y)
    
    def correct(self, pv_pred):
        
        for i in range(0, len(pv_pred)):
            if pv_pred[i] < 0:
                pv_pred[i] = 0
                
        return pv_pred
                       

    def predict(self, X):
        
        pred = self.__regressor.predict(X)
        pv_pr = [x[0] for x in pred]
        load_pr = [x[1] for x in pred]

        pv_pr = self.correct(pv_pr)

        return pv_pr, load_pr

    def clear_scores(self):

        self.pv = []
        self.load = []
        self.r2 = []

        self.rmse_scores = {
            'pv_mean': 0,
            'pv_std': 0,
            'load_mean': 0,
            'load_std': 0
        }

    def train(self, X, y, kf=KFold(n_splits=10), scaler=StandardScaler()):

        self.clear_scores()

        for train_index, val_index in kf.split(X):
            X_t, X_v = X.iloc[train_index, :], X.iloc[val_index, :]

            X_ts = scaler.fit_transform(X_t)
            X_vs = scaler.transform(X_v)

            y_t, y_v = y.iloc[train_index, :], y.iloc[val_index, :]

            self.__regressor.fit(X_ts, y_t)
            self.r2.append(self.score(X_vs, y_v))
            pv_pr, load_pr = self.predict(X_vs)
            
            pv_pr = self.correct(pv_pr)

            pv_rmse = rmse(pv_pr, y_v.pv, scale=self.maxPV)
            self.pv.append(pv_rmse)

            l_rmse = rmse(load_pr, y_v.load, scale=self.maxLoad)
            self.load.append(l_rmse)

        self.rmse_scores['pv_mean'] = np.mean(self.pv)
        self.rmse_scores['pv_std'] = np.std(self.pv)
        self.rmse_scores['load_mean'] = np.mean(self.load)
        self.rmse_scores['load_std'] = np.std(self.load)

        for k, v in self.rmse_scores.items():
            self.rmse_scores[k] = round(v, 2)
            
        X_s = scaler.fit_transform(X)
        self.__regressor.fit(X_s, y)

        return self.rmse_scores
