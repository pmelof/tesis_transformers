######################################################################
# Se utilizó de guía un tutorial que realiza un entrenamiento de un modelo ``nn.TransformerEncoder`` en una tarea de modelado de lenguaje.

"""
Se utilizó de guía un tutorial que realiza lo siguiente:
En este tutorial, entrenamos un modelo ``nn.TransformerEncoder`` en una tarea de modelado de lenguaje. 
La tarea de modelado del lenguaje consiste en asignar una probabilidad de que una palabra determinada (o una secuencia de palabras) siga una secuencia de palabras. 
Primero se pasa una secuencia de tokens a la capa de incrustación, seguida de una capa de codificación posicional para tener en cuenta el orden de la palabra.
El ``nn.TransformerEncoder`` consta de varias capas de [nn.TransformerEncoderLayer](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)_.
Junto con la secuencia de entrada, se requiere una máscara de atención cuadrada porque las capas de atención propia en ``nn.TransformerEncoder`` solo pueden atender las posiciones anteriores de la secuencia. 
Para la tarea de modelado del lenguaje, se deben enmascarar todos los tokens de las posiciones futuras. Para producir una distribución de probabilidad sobre las palabras de salida, la salida del modelo ``nn.TransformerEncoder`` se pasa a través de una capa lineal seguida de una función log-softmax.
"""

######################################################################

import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer



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


