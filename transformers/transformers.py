# Siguiendo los pasos del tutorial

import math
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset, DataLoader
# tiempo
import time

from process_input import datasetPreprocessing, generateVocabulary, generateBigVocabulary, DatasetTransformers, readDataset
from model import TransformerModel, generate_square_subsequent_mask


"""split_dataset
"""
def split_dataset(filepath_dataset: str, feature: str, decimal: int, velocity: bool = True, scaled:bool = True):
    with h5py.File(filepath_dataset, 'r') as f:
        X = f[f'X_{feature}'][()]
        Y = f['y_task'][()]   
    if velocity:
        # select the x-y velocity components
        Y = Y[:,2:4] # data shape: n x 6 (x-y position, x-y velocity, x-y acceleration)
    if scaled:
        # Scaling the dataset to have mean=0 and variance=1, gives quick model convergence.
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = X.round(decimals=decimal)
    
    limit_train = int(len(X)*.8)
    limit_eval = int(len(X)*.9)
    ds_train = DatasetTransformers(X[:limit_train], Y[:limit_train], decimal=decimal)
    ds_eval = DatasetTransformers(X[limit_train:limit_eval], Y[limit_train:limit_eval], decimal=decimal)
    ds_test = DatasetTransformers(X[limit_eval:], Y[limit_eval:], decimal=decimal)   
    
    return ds_train, ds_eval, ds_test, X

######################################################################
# Initiate an instance
# --------------------
#

# lectura datos
# 'indy_20161005_06_baks.h5' es el archivo más pequeño 4,8     
filename_dataset = 'indy_20161005_06_baks.h5'
# tokenización
decimal=1   # decimal al que se desea redondear
ini = time.time()
datasetPreprocessing(filepath_dataset=f"datos/03_baks/{filename_dataset}", filename_dataset=filename_dataset, rounded_decimal=decimal)
fin = time.time()
print(f"tokenization time:{fin-ini}")

# A priori parece mejor generar solo un archivo grande para todos los dataset
# vocabulario
dir_datasets = './datos/05_rounded'
vocabulary, maxi, mini = generateBigVocabulary(dir_datasets=dir_datasets, decimal=decimal)

batch_size = 32

# Esto debería ir en eval y en opt...

# Leer archivo
# Transformar datos a índices
# Separar en 5 ventanas el dataset (por el momento no, así que ventana = 1)
# Separar en train y test (80-20) por cada ventana
# La parte de train volver a separar en train y eval (80-20)
# Normalizar si es necesario los datos (por el momento probar sin esto, sino explota)
# En Ahmadi usan algo para dejar los datos secuenciales, posibilidad de aplicarlo también
# Entrenar el modelo con train y eval
# Evaluar el modelo con X_test y obtener Y_test_pred
# Comparar Y_test_pred con Y_test.
# Calcular RMSE y CC.







train_ds, eval_ds, test_ds, X = split_dataset(f"datos/05_rounded/{filename_dataset[:-3]}_rounded_{decimal}.h5", "sua", decimal, velocity=True, scaled=False)
train_dl = DataLoader(train_ds, batch_size, shuffle=False)
eval_dl = DataLoader(eval_ds, batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size, shuffle=False)
# Genero vocabulario, ahora por un solo archivo 
# después busco el máximo y genero un array que abarque a todos los archivos.
print(f"Generando vocabulario ... ")
vocabulary, maxi, mini = generateVocabulary(X, decimal=decimal)
print(len(vocabulary), maxi, mini)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################################################################
# The model hyperparameters are defined below. The ``vocab`` size is
# equal to the length of the vocabulary.
#
ntokens = len(vocabulary)  # size of vocabulary
emsize = 20  # embedding dimension
d_hid = 20  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)


######################################################################
# Run the model
# -------------
#


######################################################################
# We use `CrossEntropyLoss <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>`__
# with the `SGD <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`__
# (stochastic gradient descent) optimizer. The learning rate is initially set to
# 5.0 and follows a `StepLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html>`__
# schedule. During training, we use `nn.utils.clip_grad_norm\_ <https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>`__
# to prevent gradients from exploding.
#

import time

# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
lr = 5.0  # learning rate
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train(model: nn.Module, epoch: int) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 10
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(batch_size).to(device)
    

    num_batches = len(train_ds) // batch_size
    # for batch, i in enumerate(range(0, train_data.size(0) - 1, batch_size)):
        # data, targets = get_batch(train_data, i)
    for batch, (data, targets) in enumerate(train_dl):
        seq_len = data.size(0)
        if seq_len != batch_size:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(data, src_mask)
        # loss = criterion(output.view(-1, ntokens), targets)
        loss = criterion(output, targets.float())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            #ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss/1000:5.2f} | loss.item {loss.item()/1000:5.2f} ')
            total_loss = 0
            start_time = time.time()

def evaluate(model: nn.Module, eval_data) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(batch_size).to(device)
    with torch.no_grad():
        # for i in range(0, eval_data.size(0) - 1, batch_size):
        #     data, targets = get_batch(eval_data, i)
        for i, (data, targets) in enumerate(eval_data):
            seq_len = data.size(0)
            if seq_len != batch_size:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            # output_flat = output.view(-1, ntokens)
            output_flat = output
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)

######################################################################
# Loop over epochs. Save the model if the validation loss is the best
# we've seen so far. Adjust the learning rate after each epoch.

best_val_loss = float('inf')
epochs = 100

with TemporaryDirectory() as tempdir:
    best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model, epoch=epoch)
        val_loss = evaluate(model, eval_dl)
        # val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)

        scheduler.step()
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states


######################################################################
# Evaluate the best model on the test dataset
# -------------------------------------------
#

test_loss = evaluate(model, test_dl)
#test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f}')
print('=' * 89)

print("HOla")
