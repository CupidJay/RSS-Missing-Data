'''
Created on 2018.07.05

@author: caoyh
'''
import torch
import torch.nn as nn
import numpy as np
from loader import perm_mask


__all__ = ['UciNet', 'UciSampleNet', 'UciFCNet', 'UciSampleMultiBranchNet']

class BasicNFLLayer(nn.Module):
    def __init__(self, params):
        super(BasicNFLLayer, self).__init__()
        self.dd, self.dH, self.dW, self.nMul, self.nPer = params
        mask = perm_mask(self.dd, self.dH, self.dW, self.nMul)
        print(params, mask)
        self.register_buffer('mask', torch.from_numpy(mask))

        self.nfl = nn.Sequential(
            #DGConv2d(in_channels=self.dd * self.nMul, out_channels=self.dd * self.nMul, kernel_size=self.dH,
            #          padding=0),
            nn.Conv2d(in_channels=self.dd * self.nMul, out_channels=self.dd * self.nMul, kernel_size=self.dH,
                      padding=0, groups=self.dd * self.nMul // self.nPer),
            nn.BatchNorm2d(self.dd * self.nMul),
            nn.ReLU()
        )

    def forward(self, x):
        #x = torch.stack([xi[self.mask] for xi in torch.unbind(x, dim=0)], dim=0)

        now_ind = self.mask.unsqueeze(0).repeat([x.size(0), 1])
        x = x.repeat([1, self.nMul])
        x = torch.gather(x, 1, now_ind)

        # print(self.dd, self.nMul, self.dH, self.dW)
        x = x.view(x.size(0), self.dd * self.nMul, self.dH, self.dW)
        x = self.nfl(x)
        res = x.view(x.size(0),-1)

        return res

    def get_out_features(self):
        return self.dd*self.nMul



#the valinna NFL, NFL module+dense classification head
class UciNet(nn.Module): # a simple CNN with only 1 active depthwise conv. layer and 2 FC layers. BN and ReLU are both used
    def __init__(self, cfg):
        super(UciNet,self).__init__()
        self.dd = cfg.MODEL.DD
        self.dH = cfg.MODEL.DH
        self.dW = cfg.MODEL.DW
        self.nMul = cfg.MODEL.N_MUL
        self.nPer = cfg.MODEL.N_PER_GROUP
        nFC = cfg.MODEL.FC.N_FC
        bFC = cfg.MODEL.FC.B_FC
        nClass = cfg.DATASETS.CLASS


        basic_layer = BasicNFLLayer
        params = [self.dd, self.dH, self.dW, self.nMul, self.nPer]
        self.nfl_layer = basic_layer(params)

        out_dim = self.nfl_layer.get_out_features()

        self.dense = nn.Sequential()
        in_dim = out_dim
        out_dim = nFC
        for i in range(cfg.MODEL.FC.NUM_LAYERS):
            self.dense.add_module('linear_{}'.format(i), nn.Linear(in_dim, out_dim))
            self.dense.add_module('batchnorm_{}'.format(i), nn.BatchNorm1d(out_dim))
            self.dense.add_module('relu_{}'.format(i), nn.ReLU())
            in_dim = nFC
            out_dim = nFC

        self.classifier = nn.Linear(in_dim, nClass)

    def forward(self,x):
        x = self.nfl_layer(x)
        x = x.view(x.size(0), -1)
        out = self.dense(x)
        out = self.classifier(out)

        return out


class UciSampleNet(nn.Module): # a simple CNN with only 1 active depthwise conv. layer and 2 FC layers. BN and ReLU are both used
    def __init__(self, cfg):
        super(UciSampleNet, self).__init__()
        self.dd = cfg.MODEL.DD
        self.dH = cfg.MODEL.DH
        self.dW = cfg.MODEL.DW
        self.nMul = cfg.MODEL.N_MUL
        self.nPer = cfg.MODEL.N_PER_GROUP
        nFC = cfg.MODEL.FC.N_FC
        bFC = cfg.MODEL.FC.B_FC
        nClass = cfg.DATASETS.CLASS

        self.nfl = nn.Sequential(
            # DGConv2d(in_channels=self.dd * self.nMul, out_channels=self.dd * self.nMul, kernel_size=self.dH,
            #          padding=0),
            nn.Conv2d(in_channels=self.dd * self.nMul, out_channels=self.dd * self.nMul, kernel_size=self.dH,
                      padding=0, groups=self.dd * self.nMul // self.nPer),
            nn.BatchNorm2d(self.dd * self.nMul),
            nn.ReLU()
        )

        self.dense = nn.Sequential()
        in_dim = self.dd * self.nMul
        out_dim = nFC
        for i in range(cfg.MODEL.FC.NUM_LAYERS):
            self.dense.add_module('linear_{}'.format(i), nn.Linear(in_dim, out_dim))
            self.dense.add_module('batchnorm_{}'.format(i), nn.BatchNorm1d(out_dim))
            self.dense.add_module('relu_{}'.format(i), nn.ReLU())
            in_dim = nFC
            out_dim = nFC

        self.classifier = nn.Linear(in_dim, nClass)

    def forward(self, x):
        x = self.nfl(x)
        x = x.view(x.size(0), -1)
        out = self.dense(x)

        out = self.classifier(out)

        return out


class UciSampleMultiBranchNet(nn.Module): # a simple CNN with only 1 active depthwise conv. layer and 2 FC layers. BN and ReLU are both used
    def __init__(self, cfg):
        super(UciSampleMultiBranchNet, self).__init__()
        self.dd = cfg.MODEL.DD
        self.dH = cfg.MODEL.DH
        self.dW = cfg.MODEL.DW
        self.nMul = cfg.MODEL.N_MUL
        self.nPer = cfg.MODEL.N_PER_GROUP
        nFC = cfg.MODEL.FC.N_FC
        bFC = cfg.MODEL.FC.B_FC
        nClass = cfg.DATASETS.CLASS

        self.nfl = nn.Sequential(
            # DGConv2d(in_channels=self.dd * self.nMul, out_channels=self.dd * self.nMul, kernel_size=self.dH,
            #          padding=0),
            nn.Conv2d(in_channels=self.dd * self.nMul, out_channels=self.dd * self.nMul, kernel_size=self.dH,
                      padding=0, groups=self.dd * self.nMul // self.nPer),
            nn.BatchNorm2d(self.dd * self.nMul),
            nn.ReLU()
        )

        self.fcs = nn.ModuleList()
        self.channel_group = self.dd
        self.num_fc = self.dd * self.nMul // self.channel_group

        for i in range(self.num_fc):
            self.fcs.append(nn.Sequential(
                nn.Linear(self.channel_group, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Linear(64, nClass)
            ))

    def forward(self, x):
        x = self.nfl(x)
        x = x.view(x.size(0), -1)

        fc_results = []
        out = 0
        #print(x.size())
        for i in range(self.num_fc):
            temp_i = x[:, i:i+self.channel_group]
            #print(temp_i.size())
            fc_i = self.fcs[i](x[:, i:i+self.channel_group])
            fc_results.append(fc_i)
            out += fc_i

        return out, fc_results


class UciFCNet(nn.Module): # a simple CNN with only 1 active depthwise conv. layer and 2 FC layers. BN and ReLU are both used
    def __init__(self, cfg):
        super(UciFCNet,self).__init__()
        self.dd = cfg.MODEL.DD
        self.dH = cfg.MODEL.DH
        self.nMul = cfg.MODEL.N_MUL
        nFC = cfg.MODEL.FC.N_FC
        nClass = cfg.DATASETS.CLASS

        self.dense = nn.Sequential()
        in_dim = self.dd
        out_dim = nFC
        for i in range(cfg.MODEL.FC.NUM_LAYERS):
            self.dense.add_module('linear_{}'.format(i), nn.Linear(in_dim, out_dim))
            self.dense.add_module('batchnorm_{}'.format(i), nn.BatchNorm1d(out_dim))
            self.dense.add_module('relu_{}'.format(i), nn.ReLU())
            in_dim = nFC
            out_dim = nFC

        self.classifier = nn.Linear(in_dim, nClass)


    def forward(self,x):
        #print(x.size())
        x = x.view(x.size(0), -1)
        out = self.dense(x)
        out = self.classifier(out)

        return out