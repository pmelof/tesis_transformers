# Siguiendo los pasos del tutorial

import math
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

from process_input import datasetPreprocessing, generateBigVocabulary, DatasetTransformers, readDataset
from model import TransformerModel, generate_square_subsequent_mask


def splitDataset(filepath_dataset: str, feature: str, decimal: int, velocity: bool = True, scaled:bool = False):   
    X, Y = readDataset(filepath_dataset= filepath_dataset, feature= feature, velocity= velocity)
    if scaled:
        # Scaling the dataset to have mean=0 and variance=1, gives quick model convergence.
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = X.round(decimals=decimal)

    # Separar en ventanas, pero aún no, mientras solo separo en train y test del 100% de datos
    limit_train = int(len(X)*.8)
    X_train = X[:limit_train]
    Y_train = Y[:limit_train]

    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=.8, shuffle=False)

    ds_train = DatasetTransformers(X_train, Y_train, decimal=decimal)
    ds_eval = DatasetTransformers(X_valid, Y_valid, decimal=decimal)
    ds_test = DatasetTransformers(X[limit_train:], Y[limit_train:], decimal=decimal)   
       
    return ds_train, ds_eval, ds_test

######################################################################
# Initiate an instance
# --------------------
#

# lectura datos
# 'indy_20161005_06_baks.h5' es el archivo más pequeño 4,8     
filename_dataset = 'indy_20161005_06_baks.h5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
decimal=1   # decimal al que se desea redondear
dir_datasets = './datos/05_rounded'
batch_size = 32



# tokenización
datasetPreprocessing(filepath_dataset=f"datos/03_baks/{filename_dataset}", filename_dataset=filename_dataset, rounded_decimal=decimal)

# A priori parece mejor generar solo un archivo grande para todos los dataset
# vocabulario
vocabulary, maxi, mini = generateBigVocabulary(dir_datasets=dir_datasets, decimal=decimal)

config_model = {
    "ntoken": len(vocabulary),  # size of vocabulary
    "d_model" : 20,  # embedding dimension
    "d_hid" : 20,  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    "nlayers" : 2,  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    "nhead" : 2,  # number of heads in ``nn.MultiheadAttention``
    "dropout" : 0.2,  # dropout probability    
}                
model = TransformerModel(**config_model).to(device)

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

train_ds, eval_ds, test_ds = splitDataset(f"datos/05_rounded/{filename_dataset[:-3]}_rounded_{decimal}.h5", "sua", decimal, velocity=True, scaled=False)
train_dl = DataLoader(train_ds, batch_size, shuffle=False)
eval_dl = DataLoader(eval_ds, batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size, shuffle=False)

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
best_model_params_path = os.path.join("transformers/best_params", "best_model_params.pt")
    
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
   
def train(model: nn.Module, epochs: int) -> None:
    best_val_loss = float('inf')
    model.train()  # turn on train mode

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
                
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



######################################################################
# Loop over epochs. Save the model if the validation loss is the best
# we've seen so far. Adjust the learning rate after each epoch.


epochs = 10


train(model, epochs=epochs)
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
