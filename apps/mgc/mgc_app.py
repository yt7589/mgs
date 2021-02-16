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
from apps.mgc.my_net import MyNet

class MgcApp:
    def __init__(self):
        self.refl = 'apps.mgc.MgcApp'

    def startup(self, args={}):
        print('音乐流派分类')
        i_debug = 1
        if 1 == i_debug:
            self.exp()
            return
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
        log_per_batchs = 100
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                X = data['X']
                y = data['y']
                optimizer.zero_grad()
                X_ = X.reshape((1, X.shape[0], X.shape[1]))
                y_ = model(X_)
                #y_ = torch.argmax(y_raw, axis=1)
                print('y: {0}; y_: {1};'.format(y, y_))
                loss = criterion(y_, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                print('running_loss: {0};'.format(running_loss))
                if i % log_per_batchs == (log_per_batchs-1):
                    print('{0}_{1:05d}: loss: {2:0.3f};'.foramt(epoch, i, running_loss / log_per_batchs))
                break

    def exp(self):
        net = MyNet()
        print(net)
        params = list(net.parameters())
        print(len(params))
        print(params[0].size())  # conv1's .weight
        input = torch.randn(1, 1, 32, 32)
        #out = net(input)
        #print(out)
        net.zero_grad()
        #out.backward(torch.randn(1, 10))
        output = net(input)
        target = torch.randn(10)  # a dummy target, for example
        target = target.view(1, -1)  # make it the same shape as output
        criterion = torch.nn.MSELoss()
        loss = criterion(output, target)
        print(loss)

    def exp001(self):
        target = torch.randn(10)  # a dummy target, for example
        target = target.view(1, -1)
        print('target: {0};'.format(target))
        '''
        model = MgcGruModel(input_dim = GruConst.H_in, class_num=10)
        X = torch.rand(
            MgcRegistry.GRU1[GruConst.L], 
            MgcRegistry.GRU1[GruConst.N], 
            MgcRegistry.GRU1[GruConst.H_in]
        )
        print('X: {0};'.format(X.shape))
        y = model(X)
        print('v0.0.1 y(a10): {0}; {1};'.format(len(y), y.size()))
        '''