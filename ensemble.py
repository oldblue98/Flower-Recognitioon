import lightgbm as lgb

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKfold
from sklearn.metrics import log_loss
from train import load_train_df

prams = [

]

oof_path = [

]

test_path = [
    
]

data_path = "./data/output/"

class LightGBM():
    def __init__(self, params):
        self.params = params

    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test):
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
        model = lgb.train(
            self.params,lgb_train, 
            valid_sets=lgb_valid,
            num_boost_round=1000,
            early_stopping_rounds=10
            )
        preds_val = model.predict(X_valid, num_iteration=model.best_iteration)
        preds_test = model.predict(X_test, num_iteration=model.best_iteration)
        return preds_val, preds_test

def load_oof_df(path):
    oof_df = pd.DataFrame()
        for p in path:
            one_df = pd.read_csv(p)
            oof_df = pd.concat([one_df, oof_df], axis=1)