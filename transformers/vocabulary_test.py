

import numpy as np
import os
import h5py
import torch
import time


def generateVocabulary(data, decimal):
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
    mini: Int
        Valor mínimo encontrado dentro de data.
    '''
    maxi = 0 # 0 spikes en ese intervalo de tiempo.
    mini = 99999999999 # valor muy alto.
    ini = time.time()
    for i in data:
        if maxi < max(i):
            maxi = max(i)
        if mini > min(i):
            mini = min(i)
    # calcular salto según decimal
    num = 1
    for i in range(decimal):
        num = num/10
    
    vocabulary = np.arange(mini, maxi+1, num)
    fin = time.time()
    print(f"vocabulary time:{fin-ini}")
    return vocabulary, maxi, mini


datasets = os.listdir('./datos/05_rounded')
print(datasets)
decimal = 1
max_global = 0
min_global = 99999999999

for data in datasets:
    with h5py.File(f'./datos/05_rounded/{data}', 'r') as f:
        X_sua = f[f'X_sua'][()]
        X_mua = f[f'X_mua'][()]

    _, max_sua, min_sua = generateVocabulary(X_sua, decimal=decimal)
    print("sua", max_sua, min_sua)
    _, max_mua, min_mua = generateVocabulary(X_mua, decimal=decimal)
    print("mua", max_mua, min_mua)
    if (max_sua > max_global or max_mua > max_global):
        max_global = max(max_sua, max_mua)
    if (min_sua < min_global or min_mua > min_global):
        min_global = min(min_sua, min_mua)

print(max_global, min_global, round(max_global, decimal), round(min_global, decimal))
print(round(55.6, 2))
print("---"*10)
print(5.6 == 5.60)
print(np.arange(0, 5.6, 0.01))

num = 1
for i in range(3):
    num = num/10
    print(i)
print(num)    
