from model import PowerNet
from torch import optim
import torch
import torch.nn as nn
from dataset_wo_time import Dataset_wo_time
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pathlib import Path
from tqdm.auto import tqdm
from random import sample
import copy


def training(args):
    model = PowerNet(4,1)
    model = model.float()
    model.to(args.device)

    opt = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    criterion = nn.L1Loss()

    epochs = args.epochs

    with open(args.data_txt) as f:
        paths = [line[:-1] for line in f.readlines()]
        design_names = [path.split("/")[-2] for path in paths]

    dataset_s = {}
    dataset_s_train_valid ={ }
    for design_name,path in zip(design_names,paths):

        data = Dataset_wo_time(path = path ,args=args)
        dataset_s[design_name] = data
        dataset_s_train_valid[design_name] = torch.utils.data.random_split(data, [0.8, 0.2])

    best_loss = 1000000

    for epoch in range(epochs):
        for design_name in design_names:
            train_loader =  DataLoader(dataset_s_train_valid[design_name][0], batch_size=args.batch_size, shuffle=True, num_workers=0)

            model.train()
            for batch in tqdm(train_loader):
                x_s,y_s,labels = batch
          
                input = torch.stack([dataset_s[design_name].Get_input(x,y) for x,y in zip(x_s,y_s)])
       
                output= model(input)
           
                loss = criterion(output,labels.to(args.device))
           
        
                opt.zero_grad()
                loss.backward()
                opt.step()

            
            design_names_temp = copy.deepcopy(design_names)
            design_names_temp.remove(design_name)

            val_design_name = sample(design_names_temp,1)[0]

            valid_loader =  DataLoader(dataset_s_train_valid[val_design_name][1], batch_size=args.batch_size, shuffle=False, num_workers=0)

            

            loss_es=[]
            model.eval()
            for batch in tqdm(valid_loader):
                x_s,y_s,labels = batch
    
                input = torch.stack([dataset_s[val_design_name].Get_input(x,y) for x,y in zip(x_s,y_s)])
              
                output= model(input)
        

                loss = criterion(output,labels.to(args.device))
                loss_es.append(loss.detach().cpu())
        
            if best_loss > sum(loss_es)/len(loss_es):
                best_loss = sum(loss_es)/len(loss_es)
                torch.save(model.state_dict(), "./model.pt")
                print("best model saved !")
            print("train design is ",design_name,"valid design is ",val_design_name)
            print("valid loss is",sum(loss_es)/len(loss_es))
        
        
def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--data_txt", type=Path, default="./path.txt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--k", type=int, default=31)
    parser.add_argument("--l", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
   
    


    args = parser.parse_args()
    return args
if __name__ == "__main__" :
    args = parse_args()
    training(args)