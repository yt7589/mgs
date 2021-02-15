#
import numpy as np
import torch
import torch.nn
import torch.optim as optim
from apps.mgc.gtzan_data_source import GtzanDataSource
from apps.mgc.mgc_gru_model import MgcGruModel
from apps.mgc.mgc_registry import MgcRegistry
from apps.mgc.mgc_const import GruConst
from apps.mgc.gtzan_dataset import GtzanDataset

class MgcApp:
    def __init__(self):
        self.refl = 'apps.mgc.MgcApp'

    def startup(self, args={}):
        print('音乐流派分类')
        i_debug = 1
        if 1 == i_debug:
            self.exp()
            return
        ds = GtzanDataSource()
        X_train, y_train, X_valid, y_valid = ds.load_ds()
        print('X_train: {0}; y_train: {1};'.format(X_train.shape, y_train.shape))
        print('X_valid: {0}; y_valid: {1};'.format(X_valid.shape, y_valid.shape))

    def exp(self):
        print('step 1')
        ds = GtzanDataSource()
        X_train, y_train, X_valid, y_valid = ds.load_ds()
        train_dataset = GtzanDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)
        valid_dataset =GtzanDataset(X_valid, y_valid)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4,
                                          shuffle=False, num_workers=2)
        print('step 2')
        model = MgcGruModel(input_dim = GruConst.H_in, class_num=10)
        criterion = torch.nn.CrossEntropyLoss()
        #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        epochs = 1
        print('step 3')
        for i, data in enumerate(train_loader, 0):
            X = data['X']
            y = data['y']
            print('X: {0};'.format(X.size()))
            print('y: {0};'.format(y))
            break
        '''
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                print('{0}_{1}: {2}-{3};'.format(epoch, i, inputs, type(labels)))
                break
        '''



        '''
        X = torch.rand(
            MgcRegistry.GRU1[GruConst.L], 
            MgcRegistry.GRU1[GruConst.N], 
            MgcRegistry.GRU1[GruConst.H_in]
        )
        print('X: {0};'.format(X.shape))
        y = model(X)
        print('v0.0.1 y(a10): {0}; {1};'.format(len(y), y.size()))
        '''