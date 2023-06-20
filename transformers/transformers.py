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

    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=.1, shuffle=False)

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

# Esto debería ir en eval y en opt...

# Leer archivo
# Transformar datos a índices
# Separar en 5 ventanas el dataset (por el momento no, así que ventana = 1)
# Separar en train y test (80-20) por cada ventana
# La parte de train volver a separar en train y eval (80-20 o 90-10)
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


config_model = {
    "input_dim" : len(train_ds[0][0]), # dimensión de entrada de la base de datos. ej = 116
    "output_dim": len(train_ds[0][1]), # dimensión de salida de la base de datos. ej = 2
    "n_token": len(vocabulary),  # size of vocabulary
    "d_model" : 20,  # embedding dimension
    "d_hid" : 20,  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    "n_layers" : 2,  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    "n_head" : 2,  # number of heads in ``nn.MultiheadAttention``
    "dropout" : 0.2,  # dropout probability   
}                
model = TransformerModel(**config_model).to(device)

# phases = ['opt', 'eval']

# if 'opt' in phases:
#     print(f"Optimizando hiperparámetros para el archivo {filename_dataset} redondeado")

#     import os
#     #os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#     import json
#     import h5py
#     import pickle
#     import numpy as np
#     from sklearn.model_selection import train_test_split
#     from sklearn.preprocessing import StandardScaler
#     #from bmi.preprocessing import TimeSeriesSplitCustom, transform_data
#     #from bmi.utils import seed_tensorflow
#     #from bmi.decoders import QRNNDecoder, LSTMDecoder, MLPDecoder
#     #from tensorflow.keras.callbacks import EarlyStopping
#     from sklearn.metrics import mean_squared_error
#     import optuna
#     from optuna.integration import TFKerasPruningCallback
#     from optuna.trial import TrialState
#     #import time as timer
    
#     # primero lee el archivo - listo en train_ds, eval_ds, test_ds (todavía no se aplica batch_size)
#     # luego define max_timesteps = 5, max_layers = 1, max_units = 600 y la config
    
#     max_timesteps = 5
#     max_layers = 2
#     max_d_model = 120 # d_model
    
#     train_ds, eval_ds, _ = splitDataset(f"datos/05_rounded/{filename_dataset[:-3]}_rounded_{decimal}.h5", "sua", decimal, velocity=True, scaled=False)
#     train_dl = DataLoader(train_ds, batch_size, shuffle=False)
#     eval_dl = DataLoader(eval_ds, batch_size, shuffle=False)
    
#     def objective(trial):
#         config = {           
#             "input_dim" : len(train_ds[0][0]), # dimensión de entrada de la base de datos. ej = 116
#             "output_dim": len(train_ds[0][1]), # dimensión de salida de la base de datos. ej = 2
#             "n_token": len(vocabulary),  # size of vocabulary
#             "d_model": trial.suggest_int("d_model", 20, max_d_model, step=20), # embedding dimension
#             "d_hid" : 20,  # dimension of the feedforward network model in ``nn.TransformerEncoder``
#             "n_layers" : trial.suggest_int("n_layers", 1, max_layers),  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
#             "n_head" : 2,  # number of heads in ``nn.MultiheadAttention``
#             "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),  # dropout probability
#             "batch_size": trial.suggest_categorical("batch_size", [32, 64, 96]),
#             "epochs": 100,
#             "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),           
#             "optimizer": trial.suggest_categorical("optimizer", ['Adam', 'RMSProp']),
#             # "timesteps": trial.suggest_int("timesteps", 1, max_timesteps),
#             # "window_size": 2,
#             }
#         print(config)  
        
#         model = TransformerModel(config['input_dim'], config['output_dim'], config['n_token']/
#             config['d_model'], config['d_hid'], config['n_layers'], config['n_head'], config['dropout']).to(device)

#         print ("hasta aqui opt, en progreso...")
        
#         train(model, epochs=epochs, batch_size=config['batch_size'], train_dl=train_dl, optimizer=config['optimizer'], learning_rate=config['learning_rate'])
#         best_model_params_path = os.path.join("transformers/best_params", "best_model_params.pt")
#         model.load_state_dict(torch.load(best_model_params_path)) # load best model states

# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
lr = 0.5  # learning rate => basstante bueno, desde epoca 30 lr 0.11 no mejora..
lr = 0.1  # learning rate
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
best_model_params_path = os.path.join("transformers/best_params", "best_model_params.pt")
    
def evaluate(model: nn.Module, eval_data, batch_size: int) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(batch_size).to(device)
    criterion = nn.MSELoss()
    with torch.no_grad():
        Y_pred = []
        for data, targets in eval_data:
            seq_len = data.size(0)
            if seq_len != batch_size:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            Y_pred.append(output)
            total_loss += seq_len * criterion(output, targets).item()
    return total_loss / len(eval_data), Y_pred
   
def train(model: nn.Module, epochs: int, batch_size: int, train_dl: DataLoader, optimizer: str, learning_rate: float) -> None:
    best_val_loss = float('inf')
    criterion = nn.MSELoss()
    if optimizer.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    for epoch in range(1, epochs + 1):
        model.train()  # turn on train mode
        epoch_start_time = time.time()
        total_loss = 0.
        log_interval = 10
        start_time = time.time()
        src_mask = generate_square_subsequent_mask(batch_size).to(device)

        num_batches = len(train_ds) // batch_size
        for batch, (data, targets) in enumerate(train_dl):
            seq_len = data.size(0)
            if seq_len != batch_size:  # only on last batch
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            loss = criterion(output, targets.float())

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            if batch % log_interval == 0 and batch > 0:
                # lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                #ppl = math.exp(cur_loss)
                print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss/1000:5.2f}')
                total_loss = 0
                start_time = time.time()

        val_loss, Y_pred = evaluate(model, eval_dl, batch_size)        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            """_summary_
                tosave = {
                    "config" : config,
                    "wt" : model.state_dict()
                }
            """
            torch.save(model.state_dict(), best_model_params_path)
            print(config_model)

        # val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss/1000:5.2f}')
        print('-' * 89)
        
        # scheduler.step()



######################################################################
# Loop over epochs. Save the model if the validation loss is the best
# we've seen so far. Adjust the learning rate after each epoch.


epochs = 10


train(model, epochs=epochs, batch_size=batch_size, train_dl=train_dl, optimizer='Adam', learning_rate=lr)
model.load_state_dict(torch.load(best_model_params_path)) # load best model states


######################################################################
# Evaluate the best model on the test dataset
# -------------------------------------------
#

test_loss, Y_pred_test = evaluate(model, test_dl, batch_size)
# Como se tienen las velocidades x e y en tensores de 32, debo convertir cada tensor en array
Y_pred = np.array([])
suma = 0
for t in Y_pred_test:
    tmp= []
    tmp = np.append(tmp, t.numpy()).reshape(len(t),2)
    suma = suma + len(t)
    Y_pred = np.append(Y_pred, tmp)
Y_pred = Y_pred.reshape(suma,2)

# guardar en un archivo
with h5py.File(f"datos/07_results/{filename_dataset[:-3]}_rounded_{decimal}.h5", 'w') as f:      
    f['Y_pred'] = Y_pred
    f['Y_real'] = test_ds.Y
#test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss/1000:5.2f}')
print('=' * 89)


# Para graficar resultados
import matplotlib.pyplot as plt
plt.subplot(2, 2, 1)
plt.plot(np.transpose(Y_pred)[0][:100], '--', color = 'tab:red', label = 'Transformers')
plt.plot(np.transpose(test_ds.Y)[0][:100], color = 'tab:blue', label = 'True')
plt.ylabel('x-velocidad')
plt.xlabel('Tiempo [s]')
plt.title('Velocidad real vs velocidad QRNN', fontdict = {'fontsize':14, 'fontweight':'bold'})
#plt.legend(bbox_to_anchor=(1.25, -0.1), loc = 'lower right')
plt.subplot(2, 2, 2)
plt.plot(np.transpose(Y_pred)[1][:100], '--', color = 'tab:red', label = 'Transformers')
plt.plot(np.transpose(test_ds.Y)[1][:100], color = 'tab:blue', label = 'True')
plt.ylabel('y-velocidad')
plt.xlabel('Tiempo [s]')
#plt.title('Velocidad real vs velocidad QRNN', fontdict = {'fontsize':14, 'fontweight':'bold'})
plt.legend(bbox_to_anchor=(1.75, -0.01), loc = 'lower right')
# set the spacing between subplots
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show
plt.savefig("./datos/07_results/intento1.png")

print("FIN")
