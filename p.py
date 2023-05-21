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
# tiempo
import time
# separar conjunto de datos: train test
# import sklearn
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.utils.validation import _num_samples


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
    
class Pipeline_t2v(AbstractPipelineClass):
    def __init__(self, model):
        self.model = model
    
    def train(self, ds,  batch_size=32, num_epochs=100):
        ini = time.time()
        self.model.train()
        loss_fn = nn.MSELoss()
        train_dl = DataLoader(ds, batch_size, shuffle=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        loss_minor = 999999999999
        for ep in range(num_epochs):
            for x, y in train_dl:
                optimizer.zero_grad()

                # y_pred = self.model(x.unsqueeze(1).float())
                y_pred = self.model(x.float())
                loss = loss_fn(y_pred, y.float())

                loss.backward()
                optimizer.step()
                
            if loss.item() <= 50 and loss.item() < loss_minor:
                loss_minor = loss.item()
                torch.save({
                    'epoch': ep,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,                    
                }, f"checkpoints/chp_{num_epochs}_{loss_minor}")
                
        fin = time.time()
        print(f"epoch: {ep}, loss:{loss.item()} time:{fin-ini}")
    
    def preprocess(self, x):
        return x
    
    def decorate_output(self, x):
        return x


# para separar dataset en train y test
# class TimeSeriesSplitCustom(TimeSeriesSplit):
#     """
#     Create time-series data cross-validation
#     Ref: https://stackoverflow.com/questions/62210221/walk-forward-with-validation-window-for-time-series-data-cross-validation

#     Parameters
#     ----------
#     n_splits : int, default=5
#         Number of splits. 
#     max_train_size : int, default=None
#         Maximum size for a single training set.
#     test_size : int, default=1
#         Used to limit the size of the test set.
#     min_train_size : int, default=1
#         Minimum size of the training set.

#     Returns
#     ----------
#     Indices of training and testing data.
#     """
#     def __init__(self, n_splits=5, max_train_size=None, test_size=1, min_train_size=1):
#         super().__init__(n_splits=n_splits, max_train_size=max_train_size)
#         self.test_size = test_size
#         self.min_train_size = min_train_size

#     def overlapping_split(self, X, y=None, groups=None):
#         min_train_size = self.min_train_size
#         test_size = self.test_size

#         n_splits = self.n_splits
#         n_samples = _num_samples(X)

#         if (n_samples - min_train_size) / test_size >= n_splits:
#             print('(n_samples -  min_train_size) / test_size >= n_splits')
#             print('default TimeSeriesSplit.split() used')
#             yield from super().split(X)

#         else:
#             shift = int(np.floor((n_samples - test_size - min_train_size) / (n_splits - 1)))
#             start_test = n_samples - (n_splits * shift + test_size - shift)
#             test_starts = range(start_test, n_samples - test_size + 1, shift)

#             if start_test < min_train_size:
#                 raise ValueError(("The start of the testing : {0} is smaller"
#                                     " than the minimum training samples: {1}.").format(start_test, min_train_size))

#             indices = np.arange(n_samples)

#             for test_start in test_starts:
#                 if self.max_train_size and self.max_train_size < test_start:
#                     yield (indices[test_start - self.max_train_size:test_start],
#                             indices[test_start:test_start + test_size])
#                 else:
#                     yield (indices[:test_start],
#                             indices[test_start:test_start + test_size])


# lectura datos
# loop en futuro para leer varios archivos
# archivo más pequeño 4,8 
filepath_dataset = 'indy_20161005_06_baks.h5'
ds = Dataset_tesis(filepath_dataset, "sua")
# separo dataset
# tscv = TimeSeriesSplitCustom(n_splits=1, test_size=int(0.1*len(ds.y)), min_train_size=int(0.5*len(ds.y)))
# for train_idx, test_idx in tscv.split(X, y):
#     # specify training set
#     X_train = X[train_idx,:]            
#     y_train = y[train_idx,:]
#     print("X_train inicio:\n", len(X_train), "\ny_train inicio:\n", len(y_train))   
#     # specify test set
#     X_test = X[test_idx,:]
#     y_test = y[test_idx,:]
#     print("X_test inicio:\n", len(X_test), "\ny_test inicio:\n", len(y_test))
    
pipe = Pipeline_t2v(Model(activation= "sin", in_features= ds.X.shape[1], hiddem_dim= 65))

for e in range(170, 251, 10):
    pipe.train(ds= ds, num_epochs= e)

