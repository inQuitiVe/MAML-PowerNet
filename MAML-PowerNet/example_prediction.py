#!/usr/bin/env python
# coding: utf-8


#**********************************************************************
#*  predict IR-drop                                                  **
#**********************************************************************
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import pylab as plt
import json
import pickle
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle
import multiprocessing
import math
import time
from sklearn.datasets import dump_svmlight_file
import subprocess
from scipy.ndimage import convolve
import os
from IPython.display import display
#增加pandas顯示的行數
pd.set_option('display.max_rows', 300)

def predict_result(raw_data,pattern_num,dir_name):
    #feature 
    X = raw_data.loc[:, ~raw_data.columns.isin(["IR-drop", "gate_name"]) ]
    #golden IR-drop
    Y = raw_data["IR-drop"]
    if DEBUG:
        print(X)
    feature = raw_data.loc[:, ~raw_data.columns.isin(["IR-drop", "gate_name"]) ]
    dpredict = xgb.DMatrix(feature)


    #輸出每個cell的IR-drop
    predicted_data =  pd.DataFrame()
    predicted_data['ptn'+str(pattern_num)+'_golden'] = raw_data['IR-drop'].values
    predicted_data['ptn'+str(pattern_num)+'_predict'] = model.predict(dpredict)
    predicted_data.to_csv("./"+DESIGN+"/predicted_result/"+"predicted_result_"+str(pattern_num)+".csv")


if __name__ == '__main__':
    #setting----------------
    DEBUG = False
    #使用的電路
    DESIGN="MEMC"
    #require trained model to use below options
    TRAINED_MODEL = "./"+DESIGN+"/model/example_model.dat"
    #-----------------------
    #使用的電路是
    print("Design: ",DESIGN)
    #開始計時
    start = time.time()
    
    #開始執行程式
    TRAINING_SET = np.arange(20)
    VALIDATION_SET = np.array([20])
    TEST_SET = np.arange(20,100)
    print("Test set is: ",TEST_SET)
    
    #create file
    p1 = subprocess.Popen(["mkdir -p ./"+DESIGN+"/predicted_result"], stdout=subprocess.PIPE, shell=True)
    p1.wait()

    model = pickle.load(open(TRAINED_MODEL, "rb"))
    model.set_param({"predictor": "cpu_predictor",'n_gpus': 0})

    start = time.time()
    for pattern_num in TEST_SET:
        FILE_NAME = DESIGN+'_'+str(pattern_num)
        raw_data = pd.read_csv(''+DESIGN+'/data/raw_data/'+FILE_NAME+'.csv')
        dir_name = 'predict'
        predict_result(raw_data,pattern_num,dir_name)

    #結束計時
    end = time.time()
    #輸出使用時間
    print("total time: ", end-start)
    print("prdiction done")






