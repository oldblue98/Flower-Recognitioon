import argparse
import json
import os
import datetime

import lightgbm as lgb

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# 引数で config の設定を行う
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
CFG = json.load(open(options.config))

# logger の設定
from logging import getLogger, StreamHandler,FileHandler, Formatter, DEBUG, INFO
logger = getLogger("logger")    #logger名loggerを取得
logger.setLevel(DEBUG)  #loggerとしてはDEBUGで
#handler1を作成
handler_stream = StreamHandler()
handler_stream.setLevel(DEBUG)
handler_stream.setFormatter(Formatter("%(asctime)s: %(message)s"))
#handler2を作成
config_filename = os.path.splitext(os.path.basename(options.config))[0]
handler_file = FileHandler(filename=f'./logs/ensemble_lgb.log')
handler_file.setLevel(DEBUG)
handler_file.setFormatter(Formatter("%(asctime)s: %(message)s"))
#loggerに2つのハンドラを設定
logger.addHandler(handler_stream)
logger.addHandler(handler_file)

prams = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': {'multi_logloss'},
    'num_class': 4,
    'learning_rate': 0.01,
    'max_depth': 4,
    'num_leaves':3,
    'lambda_l2' : 0.3,
    'num_iteration': 2000,
    "min_data_in_leaf":1,
    'verbose': 0
}

oof_path = [
    "tf_efficientnet_b2_tf_efficientnet_b2_oof.csv",
    "vit_base_patch16_224_vit_base_patch16_224_oof.csv",
    "vit_base_resnet50d_224_vit_base_resnet50d_224_oof.csv"
]

test_path = [
    "tf_efficientnet_b2_tf_efficientnet_b2_test.csv",
    "vit_base_patch16_224_vit_base_patch16_224_test.csv",
    "vit_base_resnet50d_224_vit_base_resnet50d_224_test.csv"
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
        config_filename = os.path.splitext(os.path.basename(options.config))[0]
        # for col in one_df.columns:
        #     oof_df[config_filename + "_" + ]
        oof_df = pd.concat([one_df, oof_df], axis=1)
    if output_label:    
        label = pd.read_csv(p)["label"]
        return oof_df, label
    else:
        return oof_df

def mean_df(path):
    oof_df = pd.DataFrame()
    count = 0
    for p in path:
        one_df = pd.read_csv(p).drop("label", axis=1)
        if count < 1:
            oof_df = one_df
            count += 1
            continue
        oof_df = oof_df + one_df
    oof_df = oof_df / len(path)
    # oof_df = np.argmax(oof_df, axis=1)
    return oof_df

def main():
    oof_df, oof_label = load_df(data_path + oof_path)
    # test_df = load_df(test_path, output_label=False)

    scores_loss = []
    scores_acc = []
    folds = StratifiedKFold(n_splits=CFG["fold_num"], shuffle=True, random_state=CFG["seed"]).split(np.arange(oof_df.shape[0]), oof_label.values)
    y_preds_valid = mean_df(oof_path)
    y_preds_test = mean_df(test_path)


    # スコア
    loss = log_loss(oof_label, y_preds_valid)
    acc = (oof_label == np.argmax(y_preds_valid, axis=1)).mean()
    print(f"\t log loss: {loss}")
    print(f"\t acc: {acc}")

    loss = sum(scores_loss) / len(scores_loss)
    print('===CV scores loss===')
    print(loss)
    acc = sum(scores_acc) / len(scores_acc)
    print('===CV scores acc===')
    print(acc)

    tst_preds = np.mean(y_preds_test, axis=0)

    # 予測結果を保存
    sub = pd.read_csv("./data/sample_submission.csv")
    sub['class'] = np.argmax(tst_preds, axis=1)
    label_dic = {"daisy":0, "dandelion":1, "rose":2,"sunflower":3, "tulip":4}
    sub["class"] = sub["class"].map(label_dic)
    logger.debug(sub.value_counts("class"))
    sub.to_csv(f'data/output/submission_ensemble_mean.csv', index=False)

if __name__ == '__main__':
    main()