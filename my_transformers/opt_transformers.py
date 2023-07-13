"""
Optimize hyperparameters for deep learning based BMI decoders using Optuna
"""

# import packages
import os
import argparse
import json
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
import optuna
from optuna.trial import TrialState
import time as timer
   
from torch.utils.data import DataLoader
from transformers_p import  splitDataset2, train, evaluate, reshapeOutput, pearson_corrcoef
from model import TransformerModel
import torch
from process_input import generateBigVocabulary, readDataset, transform_data


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.stopped_epoch = 0

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def main(args):
    def objective(trial):
        max_d_model = 120 # d_model

        config = {           
            "input_dim" : len(X[0]), # dimensión de entrada de la base de datos. ej = 116
            "output_dim": len(Y[0]), # dimensión de salida de la base de datos. ej = 2
            "n_token": len(vocabulary),  # size of vocabulary
            "d_model": trial.suggest_int("d_model", 20, max_d_model, step=50), # embedding dimension (d_model = 20, 70, 120)
            "d_hid" : 20,  # dimension of the feedforward network model in ``nn.TransformerEncoder``
            "n_layers" : 1,  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
            "n_head" : 2,  # number of heads in ``nn.MultiheadAttention``
            "dropout": trial.suggest_float("dropout", 0.1, 0.3, step=0.1),  # dropout probability (dropout = 0.1, 0.2, 0.3)
            "batch_size": 96,
            "epochs": 3,
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 9e-4, step=1e-4), # (learning_rate = 0.0001, ..., 0.0009)       
            "optimizer": 'Adam',
            "timesteps": 2
            }
        print(config)  
        
        # setear semilla
        torch.manual_seed(0)
        
        rmse_eval_folds = []
        cc_eval_folds = []
        eval_loss_folds = []
        best_epochs = []
        # Creando las 5 ventanas
        windows = [.5, .6, .7, .8, .9] # % del tamaño de train (lo que quede se separá 90-10 en train y eval)
        # windows = [.8, .9]
        for window in windows:  
            print("FOLDS:", window)
            config["input_dim"] = len(X[0]) # dimensión de entrada de la base de datos. ej = 116
        
            if scaled:
                train_ds, eval_ds, _ = splitDataset2(X, Y, decimal, vocabulary=vocabulary, limit_sup_train=window, limit_sup_eval=.1 , scaled=scaled)
            else:
                train_ds, eval_ds, _ = splitDataset2(X, Y, decimal, limit_sup_train=window, limit_sup_eval=.1 , scaled=scaled)
            
            # Secuencializando los datos
            if config['timesteps'] != 1:
                train_ds.X, train_ds.Y = transform_data(train_ds.X, train_ds.Y, config['timesteps'])
                eval_ds.X, eval_ds.Y = transform_data(eval_ds.X, eval_ds.Y, config['timesteps'])
                config['input_dim'] = config["input_dim"]*config['timesteps']
            
            train_dl = DataLoader(train_ds, config['batch_size'], shuffle=False)
            eval_dl = DataLoader(eval_ds, config['batch_size'], shuffle=False)
            
            # Paradas anticipadas
            earlystop = EarlyStopper(patience=8, min_delta=1000)
            
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

            # Entrenar modelo
            print("Entrenando modelo")
            weights_path = os.path.join(f'{dir_output}/best_weights/{monkey_name}/{feature}', f"{filename_dataset.replace('.h5', '.pt')}")
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
                best_model_params_path=weights_path,
                early_stopper=earlystop)
            train_end = timer.time()
            train_time = (train_end - train_start) / 60
            
            if earlystop.stopped_epoch != 0:
                stop_epoch = earlystop.stopped_epoch
            else:
                stop_epoch = config['epochs']
            
            best_epoch = np.argmin([x["loss_epoch"] for x in history]) + 1           
            # best_epoch = np.argmin(history['loss_epochs']) + 1
            
            best_epochs.append(best_epoch)
            print(f"El entrenamiento paro en la época {stop_epoch} siendo la mejor época {best_epoch}")
            print(f"El entrenamiento duró {train_time:.2f} minutos")
            
            # Cargo los mejores pesos del modelo entrenado
            model.load_state_dict(torch.load(weights_path)) # load best model states   
            # Evaluo el modelo
            eval_loss, Y_pred = evaluate(model, eval_dl, config['batch_size'])

            # Evaluando rendimiento
            print("Evaluando rendimiento del modelo")
            Y_pred_eval = reshapeOutput(Y_pred)
            rmse_eval = mean_squared_error(eval_ds.Y, Y_pred_eval, squared=False)
            cc_eval = pearson_corrcoef(eval_ds.Y, Y_pred_eval) 
            rmse_eval_folds.append(rmse_eval)
            cc_eval_folds.append(cc_eval)
            eval_loss_folds.append(eval_loss)
        
        epochs = int(np.asarray(best_epochs).mean())
        objective.epochs = epochs
        rmse_eval_mean = np.asarray(rmse_eval_folds).mean()
        cc_eval_mean = np.asarray(cc_eval_folds).mean()
        eval_loss_mean = np.asarray(eval_loss_folds).mean()
        print("="*50)
        print("Resultados trial")
        print("RMSE folds:", np.asarray(rmse_eval_folds))
        print("rmse_eval_mean: ", rmse_eval_mean)
        print("CC folds:", np.asarray(cc_eval_folds))
        print("cc_eval_mean: ", cc_eval_mean)
        print("EVAL loss folds:", np.asarray(eval_loss_folds))
        print("eval_loss_mean: ", eval_loss_mean)
        print("="*50)
        
        # Quiero que por cada trial vea cual es mejor en rmse_mean, 
        # para el primer trial simplemente que lo guarde en objective.rmse_eval_mean, 
        # si encuentra uno mejor que lo actualice...
        try:
            objective_rmse_eval_mean = objective.rmse_eval_mean
        except:
            objective_rmse_eval_mean = rmse_eval_mean+1
            
        if (objective_rmse_eval_mean > rmse_eval_mean):
            # También guardar mejores pesos modelo en un archivo... best_weights_path
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
            weights_path = os.path.join(f'{dir_output}/best_weights/{monkey_name}/{feature}', f"{filename_dataset.replace('.h5', '.pt')}")
            model.load_state_dict(torch.load(weights_path))
            best_weights_path = os.path.join(f'{dir_output}/best_weights/{monkey_name}/{feature}', f"{filename_dataset[:-3]}_best.pt")
            torch.save(model.state_dict(), best_weights_path)
            
            objective.rmse_folds = np.asarray(rmse_eval_folds)
            objective.cc_folds = np.asarray(cc_eval_folds)
            objective.eval_loss_folds = np.asarray(eval_loss_folds)
            objective.rmse_eval_mean = rmse_eval_mean
            objective.cc_eval_mean = cc_eval_mean
            objective.eval_loss_mean = eval_loss_mean
            objective.history = history
        return rmse_eval_mean


    print("="*100)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    decimal=args.decimal   # decimal al que se desea redondear
    monkey_name = args.monkey_name
    feature = args.feature
    if args.scaled == 0:
        scaled = False
    else:
        scaled = True
    filename_dataset = args.filename_dataset
    if args.only_velocity == 0:
        only_velocity = False
    else:
        only_velocity = True
    n_startup_trials = args.n_startup_trials
    n_trials = args.n_trials
    timeout = args.timeout
    dir_datasets = f'my_transformers/data/rounded/{monkey_name}' # Directorio donde se encuentran los archivos.'
    # filename_dataset = 'indy_20160627_01_baks.h5' # más grande
    
    # Genero vocabulario
    if scaled == False:     
        vocabulary, _, _ = generateBigVocabulary(dir_datasets=dir_datasets, decimal=decimal)
    # Otra opción cuando se normaliza: crear vocabulario con números negativos y positivos, rango amplio para que funcione para todos los archivos.
    else:
        # el valor mínimo encontrado fue 4.2 y el valor máximo de 132.7.
        vocabulary = np.arange(-10, 150, 10**(-decimal)).round(decimal)
    
    # Directorio archivos de salida (hiperparámetros)
    if only_velocity:
        if scaled:
            dir_output = f'my_transformers/opt/only_velocity/normalized'
        else:
            dir_output = f'my_transformers/opt/only_velocity/not_normalized'       
    else:
        if scaled:
            dir_output = f'my_transformers/opt/all_ytask/normalized'
        else:
            dir_output = f'my_transformers/opt/all_ytask/not_normalized'

    print("Archivo a trabajar:", filename_dataset)
    # Lectura dataset
    X, Y = readDataset(f"{dir_datasets}/{filename_dataset}", feature, only_velocity=only_velocity)
    
    run_start = timer.time()
    # create and optimize study
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    params_path = f'{dir_output}/params/{monkey_name}/{feature}'
    output_filename = filename_dataset.replace('.h5', '.pkl')
    print(f"Guradando trial del estudio en el archivo: {params_path}/{output_filename}")
    with open(f'{params_path}/{output_filename}', 'wb') as f:
        pickle.dump(study, f)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Estadísticas del estudio: ")
    print("  Número de trials terminados: ", len(study.trials))
    print("  Número de trials podados: ", len(pruned_trials))
    print("  Número de trials completos: ", len(complete_trials))

    print("Mejores hiperparámetros:")
    best_hyperparams = {}       
    best_hyperparams = study.best_params                            # Return parameters of the best trial in the study.
    best_hyperparams["epochs"] = objective.epochs                   # Cantidad de épocas suficientes.
    best_hyperparams["filename"] = filename_dataset                 # Nombre del archivo ejecutado
    best_hyperparams["best_rmse_mean"] = study.best_value           # Return the best objective value in the study.
    best_hyperparams["rmse_eval_mean"] = objective.rmse_eval_mean   # RMSE promedio de la evaluación (mejor trial).
    best_hyperparams["cc_eval_mean"] = objective.cc_eval_mean       # CC promedio de la evaluación (mejor trial).
    best_hyperparams["eval_loss_mean"] = objective.eval_loss_mean   # loss promedio de la evaluación (mejor trial).
    best_hyperparams["rmse_folds"] = objective.rmse_folds           # RMSE de todas las evaluaciones.
    best_hyperparams["cc_folds"] = objective.cc_folds               # CC de todas las evaluaciones.
    best_hyperparams["eval_loss_folds"] = objective.eval_loss_folds # loss de todas las evaluaciones.
    run_end = timer.time()
    run_time = (run_end - run_start) / 60
    best_hyperparams["run_time"] = run_time                         # Tiempo total ejecución.
    best_hyperparams["time_best_trial"] = study.best_trial.duration # Tiempo mejor trial.
    best_hyperparams["history_train"] = objective.history           # Historial de train: RMSE, CC y loss por época.

    for key, value in best_hyperparams.items():
        print("    {}: {}".format(key, value))

    params_filepath = f"{params_path}/{filename_dataset.replace('.h5', '.json')}"
    print(f"Guardando los mejores parámetros en el archivo: {params_filepath}")
    with open(params_filepath, 'w') as f:
        json.dump(best_hyperparams, f, default=str, indent=2)
        
        # break

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Antes de optimizar
    parser.add_argument('--decimal',            type=int,   default=1,      help='Decimal al que se desea redondear los archivos. ej: 1')
    parser.add_argument('--monkey_name',        type=str,   default='indy', help='Nombre del mono de los archivos a trabajar. indy o loco')
    parser.add_argument('--feature',            type=str,   default='sua',  help='Tipo de spike. sua o mua')
    parser.add_argument('--scaled',             type=int,   default=0,      help='Normalizar o no los datos. 1=True o 0=False.')
    parser.add_argument('--filename_dataset',   type=str,                   help='Nombre del archivo a trabajar. (datos redondeadoos con extensión .h5)')
    parser.add_argument('--only_velocity',      type=int,   default=1,      help='Salida del modelo solo con velocidad o todo y_task. 1=True o 0=False')
    # Parámetros optuna
    parser.add_argument('--n_startup_trials',   type=int,   default=1,      help='Cantidad mínima de trials.')
    parser.add_argument('--n_trials',           type=int,   default=20,     help='Máximo de trials del estudio.')
    parser.add_argument('--timeout',            type=int,   default=100,    help='Tiempo límite del estudio [segundos].')
     
    args = parser.parse_args()
    main(args)
# main()