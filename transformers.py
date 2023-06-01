# Siguiendo los pasos del tutorial

import math
import numpy as np
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
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
        self.decoder = nn.Linear(d_model, ntoken)

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
######################################################################
# The model hyperparameters are defined below. The ``vocab`` size is
# equal to the length of the vocabulary.
#

