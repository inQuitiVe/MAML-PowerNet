from torch.utils.data import DataLoader, Dataset
import torch
import os

from tqdm.auto import tqdm
import pandas as pd 
import numpy as np
import math
from typing import Tuple
class Dataset_wo_time(Dataset):

    def __init__(self,path,args,w=128,h=128):
        super(Dataset_wo_time).__init__()
        assert args.k%2 == 1  and args.k >=3
        assert w > (args.k-1)/2  and h > (args.k-1)/2
        self.path = path
        self.k = args.k
        # self.N = args.N
        self.l = args.l

        self.device = args.device


        
        raw_design = pd.read_csv(self.path)
        feature_map = [("p_i", "Pinternal"), 
                    ("p_s","Pswitch"),
                    ("p_l", "Pleak"), 
                    ("t_min", "Tarrival"), 
                    ("t_max", ["Tarrival", "Ttransition"]), 
                    ("x_min", "x"), 
                    ("x_max", ["x", "w"]), 
                    ("y_min", "y"),
                    ("y_max", ["y", "h"]), 
                    ("r_tog", "TCoutput"), 
                    ("ir", "IR-drop")]

        data_features = pd.DataFrame()
        for feat_name, raw_name in feature_map:
            if ( type(raw_name) == list):  
                data_features[feat_name] = sum([raw_design[raw_subname] for raw_subname in raw_design[raw_name]])
            else: 
                data_features[feat_name] = raw_design[raw_name]
        self.h = math.ceil(math.ceil(max(data_features["x_max"])) / self.l)
        self.w = math.ceil(math.ceil(max(data_features["y_max"])) / self.l)

        self.X_Y = []
        self.labels = {}
        for x in range((self.k-1)//2,self.w-(self.k-1)//2):
            for y in range((self.k-1)//2,self.h-(self.k-1)//2):
                self.X_Y.append([x,y])
                key = str(x) + "_" + str(y)
                self.labels[key] = 0
        

        P_i, P_s, P_sca, P_all = self.Decomp(
                data_features, 
                math.ceil(max(data_features["x_max"])), 
                math.ceil(max(data_features["y_max"])), 
                # data_features.shape[0],
                1000,
                35000,
                args.l,
                700
        )

        self.maps = torch.from_numpy(np.array([P_i,P_s,P_sca,P_all]))
        # print(self.map.size())

    def Decomp(self,features: pd.DataFrame, W: int, H: int, C: int, T: int, l: int,
            t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
        w, h, N = math.ceil(W/l), math.ceil(H/l), math.ceil(T/t)
        P_i, P_s, P_sca, P_all = np.zeros((4,w,h))
        
        for c in tqdm(range(C)):

            c_features = features.iloc[c]
            # print(c_features)
            p_sca = (c_features["p_i"] + c_features["p_s"]) * c_features["r_tog"] + c_features["p_l"]
            p_all = c_features["p_i"] + c_features["p_l"] + c_features["p_s"]
            x_n, x_x = math.floor(c_features["x_min"]/l), math.ceil(c_features["x_max"]/l)
            y_n, y_x = math.floor(c_features["y_min"]/l), math.ceil(c_features["y_max"]/l)
            for x in range( max(x_n,(self.k-1)//2) , min(x_x,self.w-(self.k-1)//2)):
                for y in range( max(y_n,(self.k-1)//2) , min(y_x,self.h-(self.k-1)//2)):
                    key = str(x) + "_" + str(y)
                    if self.labels[key] < c_features["ir"] :
                        self.labels[key] = c_features["ir"]

            s = (x_x - x_n) * (y_x - y_n)

            M = np.zeros((w,h)) 
            M[x_n-1:x_x, y_n-1:y_x] = 1
            P_i += M * c_features["p_i"]/s
            P_s += M * c_features["p_s"]/s
            P_sca += M * p_sca/s
            P_all += M * p_all/s
        # for x in range((self.k-1)//2,self.w-(self.k-1)//2):
        #     for y in range((self.k-1)//2,self.h-(self.k-1)//2):
        #         key = str(x) + "_" + str(y)
        #         if(self.labels[key] >0):
        #             print(self.labels[key])

        return P_i, P_s, P_sca, P_all


    def Get_input(self,x,y):
        x ,y = int(x) , int(y)
        assert x >= (self.k-1)/2 and y >= (self.k-1)/2 
        assert x <= self.w-(self.k-1)/2 and y <= self.h-(self.k-1)/2 
        Ixy_s = []
        for map in self.maps:
            Ixy=map[y-(self.k-1)//2 : y+(self.k-1)//2+1]
            Ixy=torch.stack([item[x-(self.k-1)//2 : x+(self.k-1)//2+1] for item in Ixy])
            # if Ixy.size()[1] != 31:
            #     print(x,y)
            Ixy_s.append(Ixy)
    




        return torch.stack(Ixy_s).to(self.device)

    def __len__(self):
        return len(self.X_Y)
  
    def __getitem__(self,idx):
        x,y =self.X_Y[idx]
        # print(idx)
        # print(x,y)

        key =str(x)+"_"+str(y)

        label = self.labels[key]

        # label = -1 # test has no label
        return x,y,label