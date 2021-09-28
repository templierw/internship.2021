from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import datetime as dt
from joblib import dump, load
import os

def save_model(model, path, name):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    
    dump(model, f'{path}/{name}.joblib')
    
def load_model(path, name):
    return load(f'{path}/{name}.joblib')

def getDayOnly(df):
    df.index = pd.to_datetime(df.index)
    day = (df.index.time >= dt.time(hour=5)) & (df.index.time < dt.time(hour=22))
    df = df.loc[day]

    return df

def rmse(y_pred, y_true, scale=None, normalize=True, perc=True):
    rmse = mean_squared_error(y_pred, y_true)**0.5
    
    if normalize and scale is not None:
        rmse = rmse/(scale)
    
    return rmse*100 if perc else rmse

def get_train_val_split(X, y, test_size, scaled=True):
    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=test_size, random_state=42)

    if scaled:
        ss = StandardScaler()
        X_ts = ss.fit_transform(X_t)
        X_vs = ss.transform(X_v)
    
    return {
        'X_train': X_t,
        'X_train_scaled': X_ts if scaled else [],
        'X_val': X_v,
        'X_val_scaled': X_vs if scaled else [],
        'y_train': y_t,
        'y_val': y_v
    }
    
def get_res_df(X, y, pv_pr, load_pr):
    plot = pd.DataFrame(index=y.index)
    plot['pv_pred'] = pv_pr
    plot['pv_true'] = y.pv
    plot['load_pred'] = load_pr
    plot['load_true'] = y.load
    
    return plot

def train_reduced(df_train, results, name, model):

    X = df_train.drop(columns=['load','pv'])
    y = df_train[['pv', 'load']]

    res = model.train(X,y)
    save_model(model, os.environ['RES_DIR'], f"{name}_case{os.environ['CASE']}")
    
    results.loc[name, 'pv-rmse'] = res['pv_mean']
    results.loc[name, 'load-rmse'] = res['load_mean']
    
    return res, model

def train_full(df, results, name, model):
    X = df.drop(columns=['load','pv'])
    y = df[['pv', 'load']]
    
    res = model.train(X, y)
    
    results.loc[name, 'pv-rmse'] = res['pv_mean']
    results.loc[name, 'load-rmse'] = res['load_mean']
    
    return res, model

def test_hb(df_hb, results, name, model):
    X = df_hb.drop(columns=['load','pv'])
    y = df_hb[['pv', 'load']]

    ss = StandardScaler()
    X_s = ss.fit_transform(X)

    pv_pr, load_pr = model.predict(X_s)

    results.loc[name, 'hb-pv-rmse'] = round(rmse(pv_pr, y.pv, scale=float(os.environ['MAXPV'])),2)
    results.loc[name, 'hb-load-rmse'] = round(rmse(load_pr, y.load, scale=float(os.environ['MAXLOAD'])), 2)
    
    return pv_pr, load_pr, X, y
