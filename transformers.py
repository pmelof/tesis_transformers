# Siguiendo los pasos del tutorial

import math
import numpy as np
import h5py
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


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        # self.decoder = nn.Linear(d_model, ntoken)
        self.decoder = nn.Linear(d_model*116, 2) # salida de Y: velocidad (v_x, v_y)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = output.view(src.shape[0], -1)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


######################################################################
# Load and batch data
# -------------------
#
''' 
Los datos de entrada como vectores ya se tienen, 
son los archivos genenerados por baks (o binning, incluso t2v pero este último 
es más difícil de entender para generar el vocabulario)

La tokenización será redondiar los valores dados por baks a un valor entero 
o a un decimal dado. Esto es realizado en el preprocesamiento.

Del tutorial la función "batchify" no necesito implementarla, mis datos con baks 
ya estan de la forma que requiero.

Lo que si debo hacer es separar el dataset en train, validate y test
'''

def dataset_preprocessing(filepath_dataset: str , filename_dataset: str , rounded_decimal: int = 1 ):
    """
    Lee el archivo dataset baks y redondea los elementos al decimal indicado como parámetro.
    Este preprocesamiento es para SUA y MUA.
    Guarda los datos preprocesados en un archivo .h5 con: X_sua, X_mua e y_task.
    ------------
    Parámetros: 
    filepath_dataset: String
        Dirección donde se enceuntra el dataset baks.
    filename_dataset: String
        Nombre del archivo dataset baks.        
    rounded_decimal: Int
        Decimal al que se quiere redondear los datos, por defecto 1.
    -------------
    """
    with h5py.File(filepath_dataset, 'r') as f:
        X_sua = f[f'X_sua'][()]
        X_mua = f[f'X_mua'][()]
        y_task = f['y_task'][()]   
        
    assert len(X_sua) == len(X_mua) and len(X_sua) == len(y_task),\
            "Largo de X_sua, X_mua e y_task no coinciden"
    
    # Para SUA
    i=0
    while i < len(X_sua):
        j=0
        while j < len(X_sua[0]):
            X_sua[i][j] = round(X_sua[i][j], rounded_decimal)
            j = j + 1
        i = i + 1
        
    # Para MUA
    i=0
    while i < len(X_mua):
        j=0
        while j < len(X_mua[0]):
            X_mua[i][j] = round(X_mua[i][j], rounded_decimal)
            j = j + 1
        i = i + 1
        
    # guardar en un archivo
    with h5py.File(f"datos/05_rounded/{filename_dataset[:-3]}_rounded_{rounded_decimal}.h5", 'w') as f:      
        f['X_sua'] = X_sua
        f['X_mua'] = X_mua
        f['y_task'] = y_task


def generate_vocabulary(data, decimal):
    '''
    Recibe un dataset leído de rounded por la clase Dataset_for_embeddings,
    Sirve para SUA o MUA.
    Busca el valor máximo de los datos del dataset.
    Entrega un array entre 0 y el valor max, junto con el valor max.
    ------------
    Parámetros: 
    data: Tensor
        Tensor que contiene SUA o MUA de baks rounded.
    decimal: Int
        Decimal al que fue redondeado la data.
    -------------
    Retorna:
    vocabulary: Array
        Arreglo de 0 hasta un máximo.
    maxi: Int
        Valor máximo encontrado dentro de data.
    '''
    maxi = 0 # 0 spikes en ese intervalo de tiempo.
    ini = time.time()
    for i in data:
        if maxi < max(i):
            maxi = max(i)
    # calcular salto según decimal
    num = 1
    for i in range(decimal):
        num = num/10
    
    vocabulary = np.arange(0, maxi+1, num)
    fin = time.time()
    print(f"vocabulary time:{fin-ini}")
    return vocabulary, maxi


class Dataset_transformers(torch.utils.data.Dataset):
    def __init__(self, X, Y, decimal):
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
        self.decimal = decimal
        self.X = np.multiply(X, 10**self.decimal).astype(int)
        self.Y = Y      

        assert len(self.X) == len(self.Y)

    def __getitem__(self, idx):           
        return self.X[idx], self.Y[idx]
    
    def __len__(self):
        return len(self.X)
    


def split_dataset(filepath_dataset: str, feature: str, decimal: int, velocity: bool = True):
    with h5py.File(filepath_dataset, 'r') as f:
        X = f[f'X_{feature}'][()]
        Y = f['y_task'][()]   
    if velocity:
        # select the x-y velocity components
        Y = Y[:,2:4] # data shape: n x 6 (x-y position, x-y velocity, x-y acceleration)
    
    limit_train = int(len(X)*.8)
    limit_eval = int(len(X)*.9)
    ds_train = Dataset_transformers(X[:limit_train], Y[:limit_train], decimal=decimal)
    ds_eval = Dataset_transformers(X[limit_train:limit_eval], Y[limit_train:limit_eval], decimal=decimal)
    ds_test = Dataset_transformers(X[limit_eval:], Y[limit_eval:], decimal=decimal)   
    
    return ds_train, ds_eval, ds_test, X
    
######################################################################
# Functions to generate input and target sequence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``get_batch()`` generates a pair of input-target sequences for
# the transformer model. It subdivides the source data into chunks of
# length ``bptt``.
'''
Esto  lo realizare con la clase DataLoader, el batch por el momento de forma arbitraria = 32 o 64.s
'''


######################################################################
# Initiate an instance
# --------------------
#
# lectura datos
# 'indy_20161005_06_baks.h5' es archivo más pequeño 4,8     
filename_dataset = 'indy_20161005_06_baks.h5'
# tokenización
ini = time.time()
decimal=1
dataset_preprocessing(filepath_dataset=f"datos/03_baks/{filename_dataset}", filename_dataset=filename_dataset, rounded_decimal=decimal)
fin = time.time()
print(f"tokenization time:{fin-ini}")

batch_size = 32

train_ds, eval_ds, test_ds, X = split_dataset(f"datos/05_rounded/{filename_dataset[:-3]}_rounded_{decimal}.h5", "sua", decimal, velocity=True)
train_dl = DataLoader(train_ds, batch_size, shuffle=False)
eval_dl = DataLoader(eval_ds, batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size, shuffle=False)
# Genero vocabulario, ahora por un solo archivo 
# después busco el máximo y genero un array que abarque a todos los archivos.
print(f"Generando vocabulario ... ")
vocabulary, maxi = generate_vocabulary(X, decimal=decimal)
print(len(vocabulary), maxi)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


######################################################################
# The model hyperparameters are defined below. The ``vocab`` size is
# equal to the length of the vocabulary.
#
ntokens = len(vocabulary)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
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

criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
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
        loss = criterion(output, targets)

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
                  f'loss {cur_loss:5.2f} ')
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
epochs = 3

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
