#!/usr/bin/env python
# coding: utf-8


import pickle
import xgboost as xgb
import os
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
import scipy.stats as ss
import pylab as plt
import pandas as pd
import time
import matplotlib
import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix
import seaborn as sn

DESIGN="MEMC"

FILE_NAME = "MEMC"
PATTERN_BEGIN = 20
PATTERN_END = 99

TRAINING_SET = np.arange(20)
VALIDATION_SET = np.array([20])
TEST_SET = np.arange(20,100)


def load_all_data():
    all_cells_golden = pd.DataFrame()
    all_cells_predict = pd.DataFrame()
    for pattern in range(PATTERN_BEGIN,PATTERN_END+1):
        temp = pd.read_csv("./"+FILE_NAME+"/predicted_result/predicted_result_"+str(pattern)+".csv")
        all_cells_golden['ptn'+str(pattern)] = temp['ptn'+str(pattern)+'_golden']
        all_cells_predict['ptn'+str(pattern)] = temp['ptn'+str(pattern)+'_predict']
    return 1000*(0.95-all_cells_golden),1000*(0.95-all_cells_predict)
all_cells_golden,all_cells_predict = load_all_data()
print("all_cells_golden shape:", all_cells_golden.shape)



#### testing set
TEST_all_cells_golden = all_cells_golden
TEST_all_cells_predict = all_cells_predict
print("MAE(mV): ",format(np.mean(np.abs(TEST_all_cells_predict.values-TEST_all_cells_golden.values)),".2f"),"mV")
print("MaxE(mV): ",format(np.max(TEST_all_cells_predict.values-TEST_all_cells_golden.values),".2f"),'mV')
print("CC: ", format(np.corrcoef(TEST_all_cells_golden.values.flatten(),TEST_all_cells_predict.values.flatten())[0,1],".2f"))
print("NRMSE(%): ",format(np.sqrt(np.mean((TEST_all_cells_predict.values-TEST_all_cells_golden.values)**2))/np.mean(TEST_all_cells_predict.values)*100,".2f"),'%')


