import argparse
import json
import os
import datetime

import lightgbm as lgb

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKfold
from sklearn.metrics import log_loss

# 引数で config の設定を行う
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
CFG = json.load(open(options.config))

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

def load_df(path, output_label=True):
    oof_df = pd.DataFrame()
    for p in path:
        one_df = pd.read_csv(p).drop("label", axis=1)
        oof_df = pd.concat([one_df, oof_df], axis=1)
    if output_label:    
        label = pd.read_csv(p)["label"]
        return oof_df
    else:
        *args, **kwargs
    return a

def main():
    oof_df, oof_label = load_df(oof_path)
    test_df = load_df(test_path, output_label=False)

    y_preds = []
    scores_loss = []
    scores_acc = []
    folds = StratifiedKfold(n_splits=CFG["fold_num"], shuffle=True, random_state=CFG["seed"]).split(np.arange(oof_df.shape[0]), oof_label.values)
    for fold, (tr_idx, val_idx) in enumerate(folds):
        X_train, X_valid = 
    

if __name__ == '__main__':
    main()