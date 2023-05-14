from model import PowerNet
from torch import optim
import torch
import torch.nn as nn
from dataset_wo_time import Dataset_wo_time
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pathlib import Path
from tqdm.auto import tqdm


def training(args):
    model = PowerNet(4,1)
    model = model.float()
    model.to(args.device)

    opt = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    criterion = nn.L1Loss()

    epochs = args.epochs
    dataset = Dataset_wo_time(args)
    train_set, valid_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    best_loss = 1000000

    for epoch in range(epochs):
        for batch in tqdm(train_loader):
            model.train()
            x_s,y_s,labels = batch
            # print("pos size is",positions)
            input = torch.stack([dataset.Get_input(x,y) for x,y in zip(x_s,y_s)])
            # print("input size is",input.size())
            output= model(input)
            # print(output.size())
            loss = criterion(output,labels.to(args.device))
            # loss_es.append(loss)
    
            opt.zero_grad()
            loss.backward()
            opt.step()

        loss_es=[]
        model.eval()
        for batch in tqdm(valid_loader):
            x_s,y_s,labels = batch
            # print("pos size is",positions)
            input = torch.stack([dataset.Get_input(x,y) for x,y in zip(x_s,y_s)])
            # print("input size is",input.size())
            output= model(input)
            # print(output.size())

            loss = criterion(output,labels.to(args.device))
            loss_es.append(loss)
    
        if best_loss > sum(loss_es)/len(loss_es):
            best_loss = sum(loss_es)/len(loss_es)
            torch.save(model.state_dict(), "./model.pt")
            print(sum(loss_es)/len(loss_es))
        
        
def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=Path, default="./images/images")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--k", type=int, default=31)
    parser.add_argument("--l", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
   
    


    args = parser.parse_args()
    return args
if __name__ == "__main__" :
    args = parse_args()
    training(args)