# basado en https://github.com/ojus1/Time2Vec-PyTorch


# import packages
import os
import numpy as np
import h5py
import platform
import torch
from torch.utils.data import DataLoader
## Time2Vector
from torch.utils.data import DataLoader
from time2vec.Pipeline import AbstractPipelineClass
from torch import nn
from time2vec.Model import Model

print(f"Python (v{platform.python_version()})")
print(f"Pytorch (v{torch.__version__})")

class Dataset_tesis(torch.utils.data.Dataset):
    def __init__(self, filepath_dataset: str, feature: str, velocity: bool =True):
        """
        Lee el archivo dataset para trabajar con DL.
        ------------
        Parámetros: 
        filepath_dataset: String
            Dirección donde se enceuntra el dataset.
        feature: np array
            SUA o MUA.
        velocity: Boolean
            Si quiere solo la velocidad para la salida.
        -------------
        Retorna:
        X: np array
            Dataset SUA o MUA, tasa de spikes estimada.
        y: np array
            Si velocity=True, entonces y solo contiene velocidad x e y del mono, si velocity=False, entonces y tiene posición, velocidad y aceleración x e y del mono.
        """
        with h5py.File(filepath_dataset, 'r') as f:
            self.X = f[f'X_{feature}'][()]
            self.y = f['y_task'][()]   
        if velocity:
            # select the x-y velocity components
            self.y = self.y[:,2:4] # data shape: n x 6 (x-y position, x-y velocity, x-y acceleration)

        assert len(self.X) == len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return len(self.X)
    
class T2v_tesis(AbstractPipelineClass):
    def __init__(self, model, trainable = True, name = 'Time2VecLayer'):
        self.model = model
    
    def train(self, ds,  batch_size=32, num_epochs=100):
        loss_fn = nn.CrossEntropyLoss()
        train_dl = DataLoader(ds, batch_size, shuffle=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        for ep in range(num_epochs):
            for x, y in train_dl:
                optimizer.zero_grad()

                # y_pred = self.model(x.unsqueeze(1).float())
                y_pred = self.model(x.float())
                loss = loss_fn(y_pred, y)

                loss.backward()
                optimizer.step()
                
                print(f"epoch: {ep}, loss:{loss.item()}")
    
    def preprocess(self, x):
        return x
    
    def decorate_output(self, x):
        return x


pipe = T2v_tesis(Model("sin", 65))

# lectura datos
# loop en futuro para leer varios archivos
# archivo más pequeño 4,8 
filepath_dataset = 'indy_20161005_06_baks.h5'
ds = Dataset_tesis(filepath_dataset, "sua")
pipe.train(ds= ds, num_epochs= 1)

