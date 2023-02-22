import torch
import numpy as np
import hydroeval as he
import delorean
import matplotlib.pyplot as plt
import pandas as pd

class dset_maker(torch.utils.data.Dataset):
    def __init__(self,x,y,z,l,zscorex,clippedim):
        if l == 0:
            self.x = x[:z]
            self.y = y[:z]
            self.stdx,self.meanx = torch.std_mean(self.x)
            if (zscorex == True) and (clippedim == True):
                self.x = (self.x - self.meanx)/self.stdx
        else:
            self.x = x[:z-l]
            self.y = y[l:z]
            self.stdx,self.meanx = torch.std_mean(self.x)
            if (zscorex == True) and (clippedim == True):
                self.x = (self.x - self.meanx)/self.stdx


    def __getitem__(self,idx):
        x = self.x[idx].to('cuda')
        y = self.y[idx].to('cuda')
        return x, y,
    def __len__(self):
        return self.x.shape[0]

class a_linear(torch.nn.Module):
    def __init__(self,XXXX,XX,intermed_layer_size):
        super(a_linear,self).__init__()
        self.XXXX = XXXX
        self.XX = XX
        self.intermed_layer_size = intermed_layer_size
        self.relu = torch.nn.ReLU(inplace=True)
        self.linear1 = torch.nn.Linear(self.XXXX,self.intermed_layer_size) ### this is what gets changed based on important switch
        self.linear2 = torch.nn.Linear(self.intermed_layer_size,self.XX)
    def forward(self, x):
        o = self.relu(self.linear1(x.view(x.size(0),-1)))
        o = self.linear2(o)
        # o = self.relu(self.linear3(o))
        # o = self.relu(self.linear4(o))
        return o

# class a_simp(torch.nn.Module):
#     def __init__(self,XXXX,XX):
#         super(a_simp,self).__init__()
#         self.XXXX = XXXX
#         self.XX = XX
#         self.relu = torch.nn.ReLU(inplace=True)
#         self.linear1 = torch.nn.Linear(self.XXXX,24) ### this is what gets changed based on important switch
#         self.linear2 = torch.nn.Linear(24,12)
#         self.linear3 = torch.nn.Linear(50,20)
#         self.linear4 = torch.nn.Linear(20,10)
#         self.linear5 = torch.nn.Linear(12,self.XX)
#     def forward(self, x):
#         o = self.linear1(x.view(x.size(0),-1))
#         o = self.relu(self.linear2(o))
#         # o = self.relu(self.linear3(o))
#         # o = self.relu(self.linear4(o))
#         o = self.linear5(o)
#         return o
    
class a(torch.nn.Module):
    def __init__(self,XXXX,XX):
        super(a,self).__init__()
        chonz = 32
        
        #i/o sizes
        self.XXXX = XXXX
        self.XX = XX
        # into conv 
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=chonz, kernel_size=3, padding=1, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)
        # residual / latent space layers
        hidden_layers = []
        for i in range(15):
            hidden_layers.append(torch.nn.Conv2d(in_channels=chonz, out_channels=chonz, kernel_size=3, padding=1, bias=False))
            hidden_layers.append(torch.nn.BatchNorm2d(chonz))
            hidden_layers.append(torch.nn.ReLU(inplace=True))
        self.mid_layer = torch.nn.Sequential(*hidden_layers)
        # out of conv into perceptronesque / encoder
        self.conv3 = torch.nn.Conv2d(in_channels=chonz, out_channels=1, kernel_size=3, padding=1, bias=False) #anything below this is for shrinking 
        self.linear1 = torch.nn.Linear(self.XXXX,100) ### this is what gets changed based on important switch
        self.linear2 = torch.nn.Linear(100,50)
        self.linear3 = torch.nn.Linear(50,20)
        self.linear4 = torch.nn.Linear(20,10)
        self.linear5 = torch.nn.Linear(10,self.XX)
    #calls the layers explained in __init__ above
    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out = self.mid_layer(out1)
        o = self.conv3(out+out1)
        o = self.linear1(o.view(o.size(0),-1))
        o = self.relu(self.linear2(o))
        o = self.relu(self.linear3(o))
        o = self.relu(self.linear4(o))
        o = self.linear5(o)
        return o