"""
Evaluando transformers con pytorch
"""

# import packages
import os
import argparse
import json
import h5py
import numpy as np
from sklearn.metrics import mean_squared_error
import time as timer

from torch.utils.data import DataLoader
from transformers import splitDataset2, train, evaluate, reshapeOutput
from model import TransformerModel
import torch
from process_input import generateBigVocabulary, readDataset, transform_data
import matplotlib.pyplot as plt


def pearson_corrcoef(ytrue, ypred, multioutput="uniform_average"):
    """
    Compute Pearson's coefficient correlation score.
    
    Parameters
    ----------
    ytrue : ndarray
        Ground truth (correct) target values.
    ypred : ndarray
        Estimated target values.
    multioutput : str, {'raw_values', 'uniform_average'}
        Defines aggregating of multiple output values. 
        'raw_values' : Returns a full set of errors in case of multioutput input.
        'uniform_average' : Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    score : float or ndarray
        A scalar or array of floating point values.
    """
    assert ytrue.shape == ypred.shape, "both data must have same shape"
    if ytrue.ndim == 1:
        ytrue = np.expand_dims(ytrue, axis=1)
        ypred = np.expand_dims(ypred, axis=1)

    pearson_score = []
    for i in range(ytrue.shape[1]): # Loop through outputs
        score = np.corrcoef(ytrue[:,i], ypred[:,i], rowvar=False)[0,1] # choose the cross-covariance
        pearson_score.append(score)
    pearson_score = np.asarray(pearson_score)
    if multioutput == 'raw_values':
        return pearson_score
    elif multioutput == 'uniform_average':
        return np.average(pearson_score)

# Leer archivo tokenizado
# Generar vocabulario grande, o podría traerlo como parámetro 
# Definir variables a utilizar: filename_dataset, device, decimal, dir_datasets, vocabulary
# (podrían ser pasados por argumentos).
# Se lee configuración de hiperparámetros para el archivo e cuestión y se guardan en "config"
# Separar en 5 ventanas el dataset (por el momento no, así que ventana = 1)
# Separar en train y test por cada ventana, train lo vuelvo a separar en train y eval (90-10) con splitDataset. 
# La parte test no se toca hasta que se evalue el modelo entrenado.
# Secuencializar datos con timesteps.
# Entrenar el modelo con train y eval
# Evaluar el modelo con X_test y obtener Y_test_pred
# Comparar Y_test_pred con Y_test.
# Calcular RMSE y CC.


def main(args): 
    print("="*100)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decimal=1   # decimal al que se desea redondear
    monkey_name = 'indy'
    feature = 'sua'
    # filename_dataset = 'loco_20170216_02_baks_rounded_1.h5'
    # config_filepath = 'datos/06_parameters/loco_01/loco_20170216_02_baks_rounded_1.json'
    # output_filepath = 'datos/07_results/loco_01/loco_20170216_02_baks_rounded_1.h5'
    # filename_dataset = 'loco_20170217_02_baks_rounded_1.h5'
    # config_filepath = 'datos/06_parameters/loco_01/loco_20170217_02_baks_rounded_1.json'
    # output_filepath = 'datos/07_results/loco_01/loco_20170217_02_baks_rounded_1.h5'
    filename_dataset = 'loco_20170215_02_baks_rounded_1.h5'
    config_filepath = 'datos/06_parameters/loco_01/loco_20170215_02_baks_rounded_1.json'
    output_filepath = 'datos/07_results/loco_01/loco_20170215_02_baks_rounded_1.h5'
    
    dir_datasets = './datos/05_rounded/loco'
    vocabulary, _, _ = generateBigVocabulary(dir_datasets=dir_datasets, decimal=decimal)
    
    # Lectura dataset
    X, Y = readDataset(f"{dir_datasets}/{filename_dataset}", "sua", velocity=True)

    run_start = timer.time()

    print("Hyperparameter configuration setting")
    if config_filepath:
        # open JSON hyperparameter configuration file
        print(f"Using hyperparameter configuration from a file: {config_filepath}")
        with open(config_filepath, 'r') as f:
            config = json.load(f)
        
    else:
        # define model configuration
        config = {           
            "d_model": args.d_model, # embedding dimension
            "n_layers" : args.n_layers,  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
            "dropout": args.dropout,  # dropout probability
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,           
            "optimizer": args.optimizer,
            "timesteps": args.timesteps,
            # "window_size": 2,
            }
    config["input_dim"] = len(X[0]) # dimensión de entrada de la base de datos. ej = 116
    config["output_dim"] = len(Y[0]) # dimensión de salida de la base de datos. ej = 2
    config["n_token"] = len(vocabulary)  # size of vocabulary
    config["d_hid"] = 20  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    config["n_head"] = 2 # number of heads in ``nn.MultiheadAttention``
    print(f"Hyperparameter configuration: {config}")

    # setear semilla
    torch.manual_seed(0)

    rmse_test_folds = []
    cc_test_folds = []
    history_loss_train = []
    history_loss_test = []

    # best_epochs = []
    # Creando las 5 ventanas
    # windows = [.5, .6, .7, .8, .9] # % del tamaño de train (lo que quede se separá 90-10 en train y eval)
    windows = [.9]    
    for window in windows:           
        train_ds, eval_ds, test_ds = splitDataset2(X, Y, decimal, limit_sup_train=window, limit_sup_eval=.1 , scaled=False)

        # Secuencializando los datos
        if config['timesteps'] != 1:
            train_ds.X, train_ds.Y = transform_data(train_ds.X, train_ds.Y, config['timesteps'])
            eval_ds.X, eval_ds.Y = transform_data(eval_ds.X, eval_ds.Y, config['timesteps'])
            test_ds.X, test_ds.Y = transform_data(test_ds.X, test_ds.Y, config['timesteps'])
            config['input_dim'] = config["input_dim"]*config['timesteps']
        

        train_dl = DataLoader(train_ds, config['batch_size'], shuffle=False)
        eval_dl = DataLoader(eval_ds, config['batch_size'], shuffle=False)
        test_dl = DataLoader(test_ds, config['batch_size'], shuffle=False)
    
        # Crear modelo
        model = TransformerModel(
            input_dim=config['input_dim'], 
            output_dim=config['output_dim'], 
            n_token=config['n_token'],
            d_model=config['d_model'], 
            d_hid=config['d_hid'], 
            n_layers=config['n_layers'], 
            n_head=config['n_head'], 
            dropout=config['dropout']
        ).to(device)
        #total_count, _, _ = count_params(model)
        
        # Entrenar modelo
        best_model_params_path = os.path.join("transformers/best_params", "best_model_params.pt")
        train_start = timer.time()
        history_loss = train(model, 
            epochs=config['epochs'], 
            batch_size=config['batch_size'], 
            train_dl=train_dl, 
            optimizer=config['optimizer'], 
            learning_rate=config['learning_rate'],
            train_ds=train_ds, 
            eval_dl=eval_dl, 
            best_model_params_path=best_model_params_path,
            ),
        train_end = timer.time()
        train_time = (train_end - train_start) / 60
        
        # best_epoch = np.argmin(history_loss) + 1
        # best_epochs.append(best_epoch)
        # print(f"Training with the best epoch at {best_epoch}")
        print(f"Training the model took {train_time:.2f} minutes")
        
        # Cargo los mejores pesos del modelo entrenado
        best_model_params_path = os.path.join("transformers/best_params", "best_model_params.pt")
        model.load_state_dict(torch.load(best_model_params_path)) # load best model states   
        # Evaluo el modelo
        test_loss, Y_pred = evaluate(model, test_dl, config['batch_size'])

        # Evaluando rendimiento
        print("Evaluando rendimiento del modelo")
        Y_pred_test = reshapeOutput(Y_pred)
        rmse_test = mean_squared_error(test_ds.Y, Y_pred_test, squared=False)
        cc_test = pearson_corrcoef(test_ds.Y, Y_pred_test)

        rmse_test_folds.append(rmse_test)
        cc_test_folds.append(cc_test)
        history_loss_train.append(history_loss)
        history_loss_test.append(test_loss)

    # for i in range(args.n_folds):
    for i in range(len(windows)):
        print(f"Fold-{i+1} | RMSE test = {rmse_test_folds[i]:.2f}, CC test = {cc_test_folds[i]:.2f}")

    print (f"Storing results into file: {output_filepath}")
    with h5py.File(output_filepath, 'w') as f:
        f['Y_test'] = test_ds.Y
        f['Y_pred_test'] = Y_pred_test
        f['rmse_test_folds'] = np.asarray(rmse_test_folds)
        f['cc_test_folds'] = np.asarray(cc_test_folds)
        f['history_loss_train'] = np.asarray(history_loss_train)
        f['history_loss_test'] = np.asarray(history_loss_test)

    run_end = timer.time()
    run_time = (run_end - run_start) / 60
    print (f"Whole processes took {run_time:.2f} minutes")

    # Graficando resultado
    plt.subplot(2, 2, 1)
    plt.plot(np.transpose(Y_pred_test)[0][:300], '--', color = 'tab:red', label = 'Transformers')
    plt.plot(np.transpose(test_ds.Y)[0][:300], color = 'tab:blue', label = 'True')
    plt.ylabel('x-velocidad')
    plt.xlabel('Tiempo [s]')
    plt.title('Velocidad real vs velocidad QRNN', fontdict = {'fontsize':14, 'fontweight':'bold'})
    #plt.legend(bbox_to_anchor=(1.25, -0.1), loc = 'lower right')
    plt.subplot(2, 2, 2)
    plt.plot(np.transpose(Y_pred_test)[1][:300], '--', color = 'tab:red', label = 'Transformers')
    plt.plot(np.transpose(test_ds.Y)[1][:300], color = 'tab:blue', label = 'True')
    plt.ylabel('y-velocidad')
    plt.xlabel('Tiempo [s]')
    #plt.title('Velocidad real vs velocidad QRNN', fontdict = {'fontsize':14, 'fontweight':'bold'})
    # plt.legend(bbox_to_anchor=(1.75, -0.01), loc = 'lower right')
    # set the spacing between subplots
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    plt.show
    plt.savefig(f"{output_filepath[:-3]}.png")
    print(f"Imagen guardada en {output_filepath[:-3]}.png")
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--input_filepath',   type=str,   help='Path to the dataset file')
    parser.add_argument('--output_filepath',  type=str,   help='Path to the result file')
    parser.add_argument('--seed',             type=float, default=42,      help='Seed for reproducibility')
    parser.add_argument('--feature',          type=str,   default='sua',   help='Type of spiking activity (sua or mua)')
    # parser.add_argument('--n_folds',          type=int,   default=5,       help='Number of cross validation folds')
    # parser.add_argument('--min_train_size',   type=float, default=0.5,     help='Minimum (fraction) of training data size')
    parser.add_argument('--config_filepath',  type=str,   default='',      help='JSON hyperparameter configuration file')
    parser.add_argument('--timesteps',        type=int,   default=5,       help='Number of timesteps')
    parser.add_argument('--n_layers',         type=int,   default=1,       help='Number of layers')
    parser.add_argument('--d_model',          type=int,   default=20,     help='Embedding dimension')
    # parser.add_argument('--window_size',      type=int,   default=2,       help='Window size')
    parser.add_argument('--dropout',          type=float, default=0.1,     help='Dropout rate')
    parser.add_argument('--optimizer',        type=str,   default='Adam',  help='Optimizer')
    parser.add_argument('--epochs',           type=int,   default=50,      help='Number of epochs')
    parser.add_argument('--batch_size',       type=int,   default=32,      help='Batch size')
    parser.add_argument('--learning_rate',    type=float, default=0.001,   help='Learning rate')
    # parser.add_argument('--verbose',          type=int,   default=0,       help='Wether or not to print the output')
    
    args = parser.parse_args()
    main(args)