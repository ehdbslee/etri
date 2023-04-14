# ===== Torch Library ===== #
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision 
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset

from torcheval.metrics import R2Score

from torchsummary import summary

# ===== etc ===== #
import numpy as np
import pandas as pd

import copy
from copy import deepcopy

import argparse
import matplotlib.pyplot as plt

import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Embedding(nn.Module):
    def __init__(self, in_dim, emb_dim, n_layer):
        super(Embedding, self).__init__()

        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.n_layer = n_layer

        # ===== Create Linear Layers ===== #
        self.fc1 = nn.Linear(self.in_dim, self.emb_dim)

        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.n_layer-1):
            self.linears.append(nn.Linear(self.emb_dim, self.emb_dim))
            self.bns.append(nn.BatchNorm1d(self.emb_dim))

        self.fc2 = nn.Linear(self.emb_dim, self.emb_dim)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        for i in range(len(self.linears)):
            x = self.act(self.linears[i](x))
            x = self.bns[i](x)
        x = self.fc2(x)
        return x


class Net(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, n_layer, act, use_bn, use_xavier):
        super(Net, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.n_layer = n_layer
        self.act = act
        self.use_bn = use_bn
        self.use_xavier = use_xavier


        # ===== Create Linear Layers ===== #
        self.fc1 = nn.Linear(self.in_dim, self.hid_dim)

        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.n_layer-1):
            self.linears.append(nn.Linear(self.hid_dim, self.hid_dim))
            if self.use_bn: self.bns.append(nn.BatchNorm1d(self.hid_dim))

        self.fc2 = nn.Linear(self.hid_dim, self.out_dim)


        # ====== Create Activation Function ====== #
        if self.act == 'relu': self.act = nn.ReLU()
        elif self.act == 'leakyrelu': self.act = nn.LeakyReLU()
        elif self.act == 'tanh': self.act = nn.Tanh()
        elif self.act == 'sigmoid': self.act = nn.Sigmoid()
        else: raise ValueError('no valid activation function')


        # ====== Create Regularization Layer ======= #
        if self.use_xavier: self.xavier_init()


    def forward(self, x):
        x = self.act(self.fc1(x))
        for i in range(len(self.linears)):
            x = self.act(self.linears[i](x))
            if self.use_bn==True: x = self.bns[i](x)
        x = self.fc2(x)
        return x


    def xavier_init(self):
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight)
            linear.bias.data.fill_(0.01)


###############################################################################################
def test_single(partition, args, imputation=False):
    model = Net(args.in_dim, args.out_dim, args.hid_dim, args.n_layer, args.act, args.use_bn, args.use_xavier).to(device)
    
    model.load_state_dict(torch.load(f"train_models/{args.exp_name}/[{args.t}, {args.hid_dim}, {args.n_layer}]model.pth"))

    criterion = nn.MSELoss()
    metric = R2Score()
    
    ## Test ##
    test_loader = DataLoader(partition[f'{args.exp_name}'], batch_size=args.test_batch_size, shuffle=False)
    
    model.eval()

    test_loss = 0 
    r_square, total = 0, 0 

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        if args.mask == True:
            if imputation == True:
                for num in range(len(args.num_features)):
                    masking_idx = torch.zeros(inputs.shape[0], args.len_categories[num]).to(device) if num in args.except_emb else torch.ones(inputs.shape[0], args.len_categories[num]).to(device)
                    inputs[:,args.num_features[num]] = masking_idx*inputs[:,args.num_features[num]]
            
            else:
                feat_num = 0
                for num in range(len(args.len_categories)):
                    masking_idx = torch.where(torch.empty(inputs.shape[0], 1).uniform_(0, 1) > args.threshold, 1.0, 0.0).squeeze().to(device)
                    inputs[:,num] = masking_idx*inputs[:,num]
                    feat_num += args.len_categories[num]
            
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        metric.update(outputs, labels)

        test_loss += loss.item()
        r_square += metric.compute().item()

    test_loss /= len(test_loader)
    #test_accuracy = (r_square / len(test_loader)) * 100.0
    test_accuracy = (r_square / len(test_loader))

    return test_loss, test_accuracy


def test_multi(partition, args, imputation=False):
    model = Net(args.emb_dim, args.out_dim, args.hid_dim, args.n_layer, args.act, args.use_bn, args.use_xavier).to(device)
    emb_list = [Embedding(args.len_categories[i], args.emb_dim, args.emb_n_layer).to(device) for i in range(len(args.len_categories))]

    model.load_state_dict(torch.load(f"train_models/{args.exp_name}/[{args.t}, {args.emb_dim}, {args.emb_n_layer}, {args.hid_dim}, {args.n_layer}]model.pth"))
    for i in range(len(emb_list)):
        emb_list[i].load_state_dict(torch.load( f"train_models/{args.exp_name}/[{args.t}, {args.emb_dim}, {args.emb_n_layer}, {args.hid_dim}, {args.n_layer}]embedding_{i}.pth"))

    criterion = nn.MSELoss()
    metric = R2Score()
    
    ## Test ##
    test_loader = DataLoader(partition[f'{args.exp_name}'], batch_size=args.test_batch_size, shuffle=False)
    
    model.eval()
    for i in range(len(emb_list)):
        emb_list[i] = emb_list[i].eval()
        
    test_loss = 0
    r_square, total = 0, 0 

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        if imputation == True:
            embed_inputs, feat_num = 0, 0
            for num in range(len(args.len_categories)):

                col_inputs = inputs[:, feat_num:feat_num+args.len_categories[num]]
                masking_idx = torch.zeros(inputs.shape[0], args.emb_dim).to(device) if num in args.except_emb else torch.ones(inputs.shape[0], args.emb_dim).to(device)
                embed_inputs += masking_idx*emb_list[num](col_inputs)
                feat_num += args.len_categories[num]
        
        else:
            embed_inputs, feat_num = 0, 0
            for num in range(len(args.len_categories)):
                masking_idx = torch.tile(torch.where(torch.empty(inputs.shape[0], 1).uniform_(0, 1) > args.threshold, 1.0, 0.0), (1,args.emb_dim)).to(device)

                col_inputs = inputs[:, feat_num:feat_num+args.len_categories[num]]
                embed_inputs += masking_idx*emb_list[num](col_inputs)
                feat_num += args.len_categories[num]
            
        outputs = model(embed_inputs)

        loss = criterion(outputs, labels)
        metric.update(outputs, labels)

        test_loss += loss.item()
        r_square += metric.compute().item()

    test_loss /= len(test_loader)
    #test_accuracy = (r_square / len(test_loader)) * 100.0
    test_accuracy = (r_square / len(test_loader))

    return test_loss, test_accuracy


###############################################################################################
marker_list = ['o', 's', '*', 'x', 'D', '+']
color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def plot_r2(reg_acces, n_acces, mn_acces, p_acces, mp_acces, list_var):
    plt.figure(figsize=(15, 7))
    
    plt.plot(list_var, reg_acces, 'o-', color = color_list[0], label = 'Ideal')
    plt.plot(list_var, mp_acces, 's-', color = color_list[3], label = 'Multi modal w/ ED')
    plt.plot(list_var, mn_acces, 's--', color = color_list[3], label = 'Single modal w/ ED')
    plt.plot(list_var, p_acces, 'H-', color = color_list[4], label = 'Multi modal w/o ED')
    plt.plot(list_var, n_acces, 'H--', color = color_list[4], label = 'Single modal w/o ED')
    
    plt.xticks(list_var)
    plt.xlabel('Probability of Modal Imputation', fontsize=13)
    plt.ylabel('$R^2 Score$', fontsize=13)
    plt.grid()
    plt.legend(fontsize=13)

    plt.show()


def plot_r2_comb(n_acces, mn_acces, p_acces, mp_acces, list_var):
    plt.figure(figsize=(15, 7))
    plt.plot(list_var, n_acces, '*-', label = 'Single modal w/o ED')
    plt.plot(list_var, mn_acces, 's-', label = 'Single modal w/ ED')
    plt.plot(list_var, p_acces, 'h-', label = 'Multi modal w/o ED')
    plt.plot(list_var, mp_acces, 'H-', label = 'Multi modal w/ ED')
    
    plt.xticks(list_var, fontsize=12)
    plt.xlabel('Combination of Modal Imputation', fontsize=13)
    plt.ylabel('$R^2 Score$', fontsize=13)
    plt.grid()
    plt.legend(fontsize=13)

    plt.show()