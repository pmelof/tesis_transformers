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
from transformers_p import splitDataset2, train, evaluate, reshapeOutput, pearson_corrcoef
from model import TransformerModel
import torch
from process_input import generateBigVocabulary, readDataset, transform_data
import matplotlib.pyplot as plt




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
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    decimal=args.decimal   # decimal al que se desea redondear (1)
    monkey_name = args.monkey_name # indy o loco
    feature = args.feature # sua o mua
    if args.scaled == 0:
        scaled = False
    else:
        scaled = True
    filename_dataset = args.filename_dataset    # datos redondeados con extensión .h5
    # filename_dataset = 'indy_20161007_02_baks_rounded_1.h5'
    # filename_dataset = 'indy_20161017_02_baks_rounded_1.h5'
    if args.only_velocity == 0:
        only_velocity = False
    else:
        only_velocity = True
    dir_datasets = f'my_transformers/data/rounded/{monkey_name}' # Directorio donde se encuentran los archivos.'
    if args.use_weights == 0:
        use_weights = False
    else:
        use_weights = True
    dir_datasets = f'my_transformers/data/rounded/{monkey_name}' # Directorio donde se encuentran los archivos.'
    
    # Genero vocabulario
    if scaled == False:     
        vocabulary, _, _ = generateBigVocabulary(dir_datasets=dir_datasets, decimal=decimal)
    # Otra opción cuando se normaliza: crear vocabulario con números negativos y positivos, rango amplio para que funcione para todos los archivos.
    else:
        # el valor mínimo encontrado fue 4.2 y el valor máximo de 132.7.
        vocabulary = np.arange(-10, 150, 10**(-decimal)).round(decimal)
    
    # Directorio archivos de salida (resultados) 
    if only_velocity:
        if scaled:
            dir_output = f'my_transformers/eval/only_velocity/normalized'
        else:
            dir_output = f'my_transformers/eval/only_velocity/not_normalized'       
    else:
        if scaled:
            dir_output = f'my_transformers/eval/all_ytask/normalized'
        else:
            dir_output = f'my_transformers/eval/all_ytask/not_normalized'   

    # path donde se encuentra el archivo con la configuración del modelo creada por opt
    config_filepath = os.path.join(dir_output.replace('eval', 'opt'), f'params/{monkey_name}/{feature}', filename_dataset.replace('.h5', '.json'))
    # path donde se guarden los resultados del modelo
    output_filepath = os.path.join(dir_output, f'results/{monkey_name}/{feature}', filename_dataset)
    
    # Lectura dataset
    print("Archivo a evaluar:", filename_dataset)
    X, Y = readDataset(f"{dir_datasets}/{filename_dataset}", feature=feature, only_velocity=only_velocity)

    run_start = timer.time()

    print("Seteando configuración de hiperparámetros")
    if config_filepath:
        # Abriendo archivo JSON con configuración de hiperparámetros
        print(f"Usando configuración de hiperparámetros del archivo: {config_filepath}")
        with open(f"{config_filepath}", 'r') as f:
            config = json.load(f)
        
    else:
        # Definiendo configuración del modelo a partir de los argumentos
        config = {           
            "d_model": args.d_model, # embedding dimension
            "dropout": args.dropout,  # dropout probability
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,           
            "optimizer": args.optimizer,
            "timesteps": args.timesteps,
            }        
    config["input_dim"] = len(X[0]) # Dimensión de entrada de la base de datos. ej = 116
    config["output_dim"] = len(Y[0]) # Dimensión de salida de la base de datos. ej = 2
    config["n_token"] = len(vocabulary)  # Tamaño del vocabulario
    config["d_hid"] = 20  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    config["n_layers"] = 1 # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    config["n_head"] = 2 # number of heads in ``nn.MultiheadAttention``
    config["batch_size"] = 96
    config["optimizer"] = "Adam"
    config["timesteps"] = 2
    print(f"Configuración de hiperparámetros: {config}")
    
    # setear semilla
    torch.manual_seed(0)

    rmse_test_folds = []
    cc_test_folds = []
    loss_test_folds = []
    
    # Creando las 5 ventanas
    windows = [.5, .6, .7, .8, .9] # % del tamaño de train (lo que quede se separá 90-10 en train y eval)
    save_history = {}     
    for window in windows: 
        print("FOLDS:", window)
        config["input_dim"] = len(X[0]) # dimensión de entrada de la base de datos. ej = 116
        
        if scaled:
            train_ds, eval_ds, test_ds = splitDataset2(X, Y, decimal, vocabulary=vocabulary, limit_sup_train=window, limit_sup_eval=.1 , scaled=scaled)
        else:
            train_ds, eval_ds, test_ds = splitDataset2(X, Y, decimal, limit_sup_train=window, limit_sup_eval=.1 , scaled=scaled)
        
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
        
        if use_weights:
            best_weights_path = os.path.join(f'{dir_output.replace("eval", "opt")}/best_weights/{monkey_name}/{feature}', f"{filename_dataset[:-3]}_best.pt")
            print(f"Se usan los pesos del modelo previamente entrenados guardados en el archivo: {best_weights_path}")
        else:
            # Entrenar modelo
            print("Entrenando modelo")
            best_weights_path = os.path.join(f'{dir_output}/best_weights/{monkey_name}/{feature}', f"{filename_dataset.replace('.h5', '.pt')}")
            # best_model_params_path = os.path.join(f"my_transformers/best_params/best_weights_{monkey_name}/opt", f"{filename_dataset.replace('.h5', '.pt')}")
            train_start = timer.time()
            history = train(model, 
                epochs=config['epochs'], 
                batch_size=config['batch_size'], 
                train_dl=train_dl, 
                optimizer=config['optimizer'], 
                learning_rate=config['learning_rate'],
                train_ds=train_ds, 
                eval_ds=eval_ds, 
                eval_dl=eval_dl, 
                best_model_params_path=best_weights_path)
            train_end = timer.time()
            train_time = (train_end - train_start) / 60

            print(f"El entrenamiento duró {train_time:.2f} minutos")
             
            save_history[f"history_train_fold_{window}"] = history           # Historial de train: RMSE, CC y loss por época.
            history_filepath = f"{output_filepath.replace('.h5', '.json')}"
            print(f"Guardando historial de train en el archivo: {history_filepath}")
            with open(history_filepath, 'w') as f:
                json.dump(save_history, f, default=str, indent=2)
        
        # Cargo los mejores pesos del modelo entrenado
        model.load_state_dict(torch.load(best_weights_path)) # load best model states   
        # Evaluo el modelo
        loss_test, Y_pred = evaluate(model, test_dl, config['batch_size'])

        # Evaluando rendimiento
        print("Evaluando rendimiento del modelo")
        Y_pred_test = reshapeOutput(Y_pred)
        rmse_test = mean_squared_error(test_ds.Y, Y_pred_test, squared=False)
        cc_test = pearson_corrcoef(test_ds.Y, Y_pred_test)
        # Guardo después de evaluar por cada ventana: loss, rmse y cc.
        rmse_test_folds.append(rmse_test)
        cc_test_folds.append(cc_test)
        loss_test_folds.append(loss_test) 
 
    for i in range(len(windows)):
        print(f"Fold-{i+1} | RMSE test = {rmse_test_folds[i]:.2f}, CC test = {cc_test_folds[i]:.2f}")

    print (f"Guardando resultados en el archivo: {output_filepath}")
    with h5py.File(output_filepath, 'w') as f:
        f['Y_test'] = test_ds.Y                                     # Y real 
        f['Y_pred_test'] = Y_pred_test                              # Y predicción
        f['rmse_test_folds'] = np.asarray(rmse_test_folds)          # RMSE por ventana
        f['cc_test_folds'] = np.asarray(cc_test_folds)              # CC por ventana
        f['loss_test_folds'] = np.asarray(loss_test_folds)          # loss por ventana
        ###### history es un diccionario, debo ver si puedo cambiarlo a np.array
        # f['history_train'] = np.asarray(history)
        # f['history_train'] = np.asarray(history_train)    # loss en train por época
        # f['history_rmse_train'] = np.asarray(history_rmse_train)    # RMSE en train por época
        # f['history_cc_train'] = np.asarray(history_cc_train)        # CC en train por época
     
    run_end = timer.time()
    run_time = (run_end - run_start) / 60
    print (f"Whole processes took {run_time:.2f} minutes")



    # Graficando resultado
    plt.subplot(2, 1, 1)
    plt.plot(np.transpose(Y_pred_test)[0][:200], '--', color = 'tab:red', label = 'Transformers')
    plt.plot(np.transpose(test_ds.Y)[0][:200], color = 'tab:blue', label = 'True')
    plt.ylabel('x-velocidad')
    plt.xlabel('Tiempo [s]')
    plt.title('Velocidad real vs velocidad Transformers', fontdict = {'fontsize':14, 'fontweight':'bold'})
    #plt.legend(bbox_to_anchor=(1.25, -0.1), loc = 'lower right')
    plt.subplot(2, 1, 2)
    plt.plot(np.transpose(Y_pred_test)[1][:200], '--', color = 'tab:red', label = 'Transformers')
    plt.plot(np.transpose(test_ds.Y)[1][:200], color = 'tab:blue', label = 'True')
    plt.ylabel('y-velocidad')
    plt.xlabel('Tiempo [s]')
    #plt.title('Velocidad real vs velocidad QRNN', fontdict = {'fontsize':14, 'fontweight':'bold'})
    # plt.legend(bbox_to_anchor=(1.75, -0.01), loc = 'lower right')
    # set the spacing between subplots
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    plt.show
    plt.savefig(f"{output_filepath.replace('.h5', '.png')}")
    print(f"Imagen guardada en {output_filepath.replace('.h5', '.png')}")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Hiperparámetros
    parser.add_argument('--decimal',            type=int,   default=1,      help='Decimal al que se desea redondear los archivos. ej: 1')
    parser.add_argument('--monkey_name',        type=str,   default='loco', help='Nombre del mono de los archivos a trabajar. indy o loco')
    parser.add_argument('--feature',            type=str,   default='sua',  help='Tipo de spike. sua o mua')
    parser.add_argument('--scaled',             type=int,   default=1,      help='Normalizar o no los datos. 1=True o 0=False.')
    parser.add_argument('--filename_dataset',   type=str,                   help='Nombre del archivo a trabajar. (datos redondeadoos con extensión .h5).')
    parser.add_argument('--only_velocity',      type=int,   default=1,      help='Salida del modelo solo con velocidad o todo y_task. 1=True o 0=False')
    parser.add_argument('--use_weights',        type=int,   default=0,      help='¿Se desea ejecutar train otra vez? o mejor usar los pesos guardados. 1=True o 0=False')
    # parser.add_argument('--filename_config',    type=str,                   help='Nombre del archivo con las configuraciones del modelo.')
    # para el modelo de Transformers
    parser.add_argument('--d_model',            type=int,   default=20,     help='Dimensión del embedding')
    parser.add_argument('--n_layers',           type=int,   default=1,      help='Número de capas')
    parser.add_argument('--dropout',            type=float, default=0.2,    help='Dropout rate')
    parser.add_argument('--batch_size',         type=int,   default=96,     help='Batch size')
    parser.add_argument('--epochs',             type=int,   default=50,     help='Número de épocas')
    parser.add_argument('--learning_rate',      type=float, default=0.0001,  help='Learning rate')
    parser.add_argument('--optimizer',          type=str,   default='Adam', help='Optimizer') 
    parser.add_argument('--timesteps',          type=int,   default=2,      help='Cantidad de arreglos para secuencializar. 1, ..., 5.')
     
    args = parser.parse_args()
    main(args)
