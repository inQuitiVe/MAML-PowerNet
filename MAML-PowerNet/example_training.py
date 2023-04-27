#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# coding: utf-8

from re import A
import xgboost as xgb
import numpy as np
import os
import subprocess
import pickle
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib
import matplotlib.cm as cm
import pylab as plt
import time
from typing import Tuple
import pandas as pd

# 電路名稱
DESIGN="MEMC"

#建立資料夾存model 
p1 = subprocess.Popen(["mkdir -p ./"+DESIGN+"/model"], stdout=subprocess.PIPE, shell=True)
p1 = subprocess.Popen(["mkdir -p ./"+DESIGN+"/data/train"], stdout=subprocess.PIPE, shell=True)
p1 = subprocess.Popen(["mkdir -p ./"+DESIGN+"/data/val"], stdout=subprocess.PIPE, shell=True)

#等資料夾建立好
p1.wait()

DEBUG = False

if (not os.path.exists("./"+DESIGN+"/feature_name.npy")):
    all_data = pd.read_csv(''+DESIGN+'/data/raw_data/'+DESIGN+'_1.csv')
    X = all_data.loc[:, ~all_data.columns.isin(["IR-drop", "gate_name"]) ]
    Y = all_data["IR-drop"]
    np.save("./"+DESIGN+"/feature_name.npy",np.array(X.columns))

feature_name = np.load("./"+DESIGN+"/feature_name.npy",allow_pickle=True)

if DEBUG:
    print(feature_name)
    print(len(feature_name))

TRAINING_SET = np.arange(20)
VALIDATION_SET = np.array([20])

print("Training set is: ",TRAINING_SET)
print("Validation set is: ", VALIDATION_SET)



def dump_file(raw_data,pattern_num,dir_name):
    #feature 
    X = raw_data.loc[:, ~raw_data.columns.isin(["IR-drop", "gate_name"]) ]
    #golden IR-drop
    Y = raw_data["IR-drop"]
    if DEBUG:
        print(X)
    dump_svmlight_file(X,Y,"./"+DESIGN+"/data/"+dir_name+"/"+DESIGN+"_"+str(pattern_num)+".dat")

#splite training & validation set 
def load_data(training_set,validation_set):
    train_name_dict = [str(i+1) for i in training_set]
    feature_name = np.load("./"+DESIGN+"/feature_name.npy",allow_pickle=True)
    
    for pattern_num in train_name_dict:
        DIR = 'train'
        FILE_STR = DIR+'/'+DESIGN+'_'+str(pattern_num)
        if (not os.path.exists("./"+DESIGN+"/data/"+FILE_STR+".dat")):
            all_data = pd.read_csv(''+DESIGN+'/data/raw_data/'+FILE_STR+'.csv')
            dump_file(all_data,pattern_num,DIR)

    for pattern_num in validation_set:
        DIR = 'val'
        FILE_STR = DIR+'/'+DESIGN+'_'+str(pattern_num)
        if (not os.path.exists("./"+DESIGN+"/data/"+FILE_STR+".dat")):
            all_data = pd.read_csv(''+DESIGN+'/data/raw_data/'+FILE_STR+'.csv')
            dump_file(all_data,pattern_num,DIR)

    #所有在train folder底下的vector為訓練資料
    dtrain = xgb.DMatrix('./'+DESIGN+'/data/train/',feature_names=feature_name)
    #所有在val folder底下的vector為測試資料
    dval = xgb.DMatrix('./'+DESIGN+'/data/val',feature_names=feature_name)

    return dtrain,dval


def training(dtrain,dval):
    #  eval_metric 用 mae 評估模型效果
    #  objective 目標函數是 reg:squarederror
    #    objective 有以下參數可以使用
    #      reg:squarederror: regression with squared loss
    #      reg:squaredlogerror: regression with squared log loss
    #      reg:logistic: logistic regression
    #      reg:pseudohubererror: regression with Pseudo Huber loss, a twice differentiable alternative to absolute loss.
#     param = {'tree_method':'gpu_hist','objective':'reg:squarederror'}
    param = {'max_depth': 20, 'eta': 0.3,'eval_metric':'mae', 'objective': 'reg:squarederror'}
    # watchlist的第一項為early stop觀察的值
    watchlist = [(dtrain, 'train'),(dval, 'eval')]
    # 訓練的iteration數量，也就是我們的forest最多會有50000個tree
    num_round = 50000

    #開始訓練
    #  params 引入參數
    #  dtrain 輸入訓練資料
    #  evals  輸入評估函數
    #  early_stopping_rounds 如果超過10個round沒辦法產生更好的解就停止
    model = xgb.train(params=param, dtrain=dtrain, num_boost_round=num_round, evals=watchlist,early_stopping_rounds=10)
    #存入訓練好的model
    pickle.dump(model, open("./"+DESIGN+"/model/example_model.dat", "wb"))
    return model

dtrain,dval = load_data(TRAINING_SET,VALIDATION_SET)

#訓練模型
print("start tranining")
start = time.time()
model = training(dtrain,dval)
end = time.time()
print("total time: ", end-start)
print("model produced sucessfully")
