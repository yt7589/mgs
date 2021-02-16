#
import torch
import torch.nn
import torch.nn.functional as F
#
from apps.mgc.mgc_registry import MgcRegistry
from apps.mgc.mgc_const import GruConst

class MgcGruModel(torch.nn.Module):
    def __init__(self, input_dim, class_num):
        super(MgcGruModel, self).__init__()
        self.refl = ''
        self.bottle_neck_size = 100
        self.gru1 = torch.nn.GRU(
            MgcRegistry.GRU1[GruConst.H_in], 
            MgcRegistry.GRU1[GruConst.hidden_size], 
            MgcRegistry.GRU1[GruConst.num_layers]
        )
        self.gru2 = torch.nn.GRU(
            MgcRegistry.GRU2[GruConst.H_in], 
            MgcRegistry.GRU2[GruConst.hidden_size], 
            MgcRegistry.GRU2[GruConst.num_layers]
        )
        self.gru3 = torch.nn.GRU(
            MgcRegistry.GRU3[GruConst.H_in], 
            MgcRegistry.GRU3[GruConst.hidden_size], 
            MgcRegistry.GRU3[GruConst.num_layers]
        )
        self.leakyReLU1 = torch.nn.LeakyReLU()
        self.dropout1 = torch.nn.Dropout(0.2)
        self.bn1 = torch.nn.BatchNorm1d(MgcRegistry.GRU3[GruConst.hidden_size])
        self.flatten1 = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(
            in_features=MgcRegistry.GRU3[GruConst.hidden_size], 
            out_features=self.bottle_neck_size
        )
        self.leakyReLU2 = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(in_features=self.bottle_neck_size, out_features=class_num)
        '''
        self.softmax1 = torch.nn.LogSoftmax()
        '''

    def forward(self, X):
        h1_0 = torch.randn(
            MgcRegistry.GRU1[GruConst.num_layers] * MgcRegistry.GRU1[GruConst.num_directions], 
            MgcRegistry.GRU1[GruConst.N], 
            MgcRegistry.GRU1[GruConst.hidden_size]
        )
        a1 = self.gru1(X, h1_0)
        h2_0 = torch.randn(
            MgcRegistry.GRU2[GruConst.num_layers] * MgcRegistry.GRU2[GruConst.num_directions], 
            MgcRegistry.GRU2[GruConst.N], 
            MgcRegistry.GRU2[GruConst.hidden_size]
        )
        a2 = self.gru2(a1[1], h2_0)
        h3_0 = torch.randn(
            MgcRegistry.GRU3[GruConst.num_layers] * MgcRegistry.GRU3[GruConst.num_directions], 
            MgcRegistry.GRU3[GruConst.N], 
            MgcRegistry.GRU3[GruConst.hidden_size]
        )
        a3 = self.gru3(a2[1], h3_0)
        N = a3[1].size()[1]
        L = a3[1].size()[2]
        a3 = a3[1].reshape((N, L))
        a4 = self.leakyReLU1(a3)
        a5 = self.dropout1(a4)
        a6 = self.bn1(a5)
        a7 = self.flatten1(a6)
        a8 = self.linear1(a7)
        a9 = self.leakyReLU2(a8)
        return self.linear2(a9)

    def build_model(self, input_dim):
        self.model = torch.nn.Sequential()
        gru1 = torch.nn.GRU(input_dim, 100, 2)
        self.model.add_module(gru1)
        gru2 = torch.nn.GRU(100, 500, 2)
        self.model.add_module(gru2)
        gru3 = torch.nn.GRU(500, 1000, 2)
        self.model.add_module(gru3)
        leakyReLU1 = torch.nn.LeakyReLU()
        self.model.add_module(leakyReLU1)
        dropout1 = torch.nn.Dropout(0.2)
        self.model.add_module(dropout1)
        bn1 = torch.nn.BatchNorm1d()
        self.model.add_module(bn1)
        flatten1 = torch.nn.Flatten()
        self.model.add_module(flatten1)
        linear1 = torch.nn.Linear(out_features=100)
        self.model.add_module(linear1)
        leakyReLU2 = torch.nn.LeakyReLU()
        self.model.add_module(leakyReLU2)
        linear2 = torch.nn.Linear(out_features=10)
        self.model.add_module(linear2)
        softmax1 = torch.nn.LogSoftmax()
        self.model.add_module(softmax1)


    def train(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass