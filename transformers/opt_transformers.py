"""
Optimize hyperparameters for deep learning based BMI decoders using Optuna
"""

# import packages
import os
# import argparse
import json
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
import optuna
from optuna.trial import TrialState
import time as timer
   
from torch.utils.data import DataLoader
from transformers import  splitDataset2, train, evaluate, reshapeOutput
from model import TransformerModel
import torch
from process_input import generateBigVocabulary, readDataset, transform_data, datasetPreprocessing



# Leer archivo tokenizado 
# Generar vocabulario (?), o podría traerlo como parámetro 
# Definir variables a utilizar: filename_dataset, device, decimal, dir_datasets, vocabulary
# (podrían ser pasados por argumentos).
# Variables que se necesitan definir aquí: max_timesteps (?), max_layers, max_d_model, train_ds, eval_ds
# Separar en 5 ventanas el dataset (por el momento no, así que ventana = 1)
# Separar en train y test (80-20) por cada ventana, aquí solo ocupo train
# y lo vuelvo a separar en train y eval (90-10) con splitDataset.
# Normalizar si es necesario los datos (por el momento probar sin esto, sino explota)
# En Ahmadi usan algo para dejar los datos secuenciales, posibilidad de aplicarlo también
# Entrenar el modelo con train y eval


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


def main():
    def objective(trial):
        max_timesteps = 5
        max_layers = 2
        max_d_model = 120 # d_model

        config = {           
            "input_dim" : len(X[0]), # dimensión de entrada de la base de datos. ej = 116
            "output_dim": len(Y[0]), # dimensión de salida de la base de datos. ej = 2
            "n_token": len(vocabulary),  # size of vocabulary
            "d_model": trial.suggest_int("d_model", 20, max_d_model, step=20), # embedding dimension
            "d_hid" : 20,  # dimension of the feedforward network model in ``nn.TransformerEncoder``
            "n_layers" : trial.suggest_int("n_layers", 1, max_layers),  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
            "n_head" : 2,  # number of heads in ``nn.MultiheadAttention``
            "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),  # dropout probability
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 96]),
            "epochs": 100,
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),           
            "optimizer": trial.suggest_categorical("optimizer", ['Adam', 'RMSProp']),
            "timesteps": trial.suggest_int("timesteps", 1, max_timesteps),
            # "window_size": 2,
            }
        print(config)  
        
        # setear semilla
        torch.manual_seed(0)
        
        rmse_valid_folds = []
        best_epochs = []
        # Creando las 5 ventanas
        # windows = [.5, .6, .7, .8, .9] # % del tamaño de train (lo que quede se separá 90-10 en train y eval)
        windows = [.9]
        for window in windows:  
            # Normalizar no se si sirva, pero de hacerlo debe ser antes de que los datos
            # sean tokenizados (pasados a índices de vocabulario).        
            train_ds, eval_ds, _ = splitDataset2(X, Y, decimal, limit_sup_train=window, limit_sup_eval=.1 , scaled=False)

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
            best_model_params_path = os.path.join(f"transformers/best_params/best_weights_{monkey_name}", f"{filename_dataset.replace('.h5', '.pt')}")
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
                early_stopper=earlystop),
            train_end = timer.time()
            train_time = (train_end - train_start) / 60
            
            if earlystop.stopped_epoch != 0:
                stop_epoch = earlystop.stopped_epoch
            else:
                stop_epoch = config['epochs']
            best_epoch = np.argmin(history_loss) + 1
            best_epochs.append(best_epoch)
            print(f"Training stopped at epoch {stop_epoch} with the best epoch at {best_epoch}")
            print(f"Training the model took {train_time:.2f} minutes")
            
            # Cargo los mejores pesos del modelo entrenado
            model.load_state_dict(torch.load(best_model_params_path)) # load best model states   
            # Evaluo el modelo
            eval_loss, Y_pred = evaluate(model, eval_dl, config['batch_size'])

            # Evaluando rendimiento
            print("Evaluando rendimiento del modelo")
            Y_pred_eval = reshapeOutput(Y_pred)
            rmse_valid = mean_squared_error(eval_ds.Y, Y_pred_eval, squared=False)
            rmse_valid_folds.append(rmse_valid)
        
        epochs = int(np.asarray(best_epochs).mean())
        objective.epochs = epochs
        rmse_valid_mean = np.asarray(rmse_valid_folds).mean()
        print("="*50)
        print("rmse_valid_mean: ", rmse_valid_mean)
        print("eval_loss: ", eval_loss)
        print("="*50)
        objective.rmse_valid_mean = rmse_valid_mean
        objective.eval_loss = eval_loss # guarda el último eval_loss
        return rmse_valid_mean


    print("="*100)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decimal=1   # decimal al que se desea redondear
    monkey_name = 'indy'
    feature = 'sua'
    # filename_dataset = 'indy_20160627_01_baks.h5' # más grande
    
    # tokenización archivos del mono con el que se trabajará
    if monkey_name == 'indy':
        if len(os.listdir('./datos/05_rounded/indy')) < 37:
            for filename_dataset in os.listdir('./datos/03_baks/indy'):
                datasetPreprocessing(filepath_dataset=f"datos/03_baks/indy/{filename_dataset}", filename_dataset=filename_dataset, filepath_output="datos/05_rounded/indy",  rounded_decimal=decimal)
        dir_datasets = './datos/05_rounded/indy'
    else:
        if len(os.listdir('./datos/05_rounded/loco')) < 10:
            for filename_dataset in os.listdir('./datos/03_baks/loco'):
                datasetPreprocessing(filepath_dataset=f"datos/03_baks/loco/{filename_dataset}", filename_dataset=filename_dataset, filepath_output="datos/05_rounded/loco",  rounded_decimal=decimal)
        dir_datasets = './datos/05_rounded/loco'
    
    vocabulary, _, _ = generateBigVocabulary(dir_datasets=dir_datasets, decimal=decimal)

    # num = 1
    for filename_dataset in os.listdir(dir_datasets):
        if (filename_dataset == 'indy_20160411_02_baks_rounded_1.h5' or  
            filename_dataset == 'indy_20160420_01_baks_rounded_1.h5' or 
            filename_dataset == 'indy_20160622_01_baks_rounded_1.h5' or 
            filename_dataset == 'indy_20160624_03_baks_rounded_1.h5' or 
            filename_dataset == 'indy_20160630_01_baks_rounded_1.h5' or 
            filename_dataset == 'indy_20160927_04_baks_rounded_1.h5' or 
            filename_dataset == 'indy_20161005_06_baks_rounded_1.h5' or 
            filename_dataset == 'indy_20161007_02_baks_rounded_1.h5' or 
            filename_dataset == 'indy_20161024_03_baks_rounded_1.h5' or 
            filename_dataset == 'indy_20161026_03_baks_rounded_1.h5' or 
            filename_dataset == 'indy_20161212_02_baks_rounded_1.h5' or 
            filename_dataset == 'indy_20161220_02_baks_rounded_1.h5' or 
            filename_dataset == 'indy_20170124_01_baks_rounded_1.h5' or 
            filename_dataset == 'indy_20160927_06_baks_rounded_1.h5'
            ):
            continue
        print("Archivo a trabajar:", filename_dataset)
        # Lectura dataset
        X, Y = readDataset(f"{dir_datasets}/{filename_dataset}", feature, velocity=True)
        
        run_start = timer.time()
        # create and optimize study
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=10))
        study.optimize(objective, n_trials=200, timeout=10000)

        output_filepath = f'datos/06_parameters/{monkey_name}'
        # output_filepath = f'datos/06_parameters/test'
        output_filename = filename_dataset.replace('.h5', '.pkl')
        print(f"Storing study trial into a file: {output_filepath}/{output_filename}")
        with open(f'{output_filepath}/{output_filename}', 'wb') as f:
            pickle.dump(study, f)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best hyperparameters:")
        # best_trial = study.best_trial.params # Return the best trial in the study. --> .params
        # best_params = study.best_params # Return parameters of the best trial in the study.
        # best_value = study.best_value # Return the best objective value in the study.
        best_hyperparams = {}       
        # best_hyperparams['best_trial'] = best_trial
        best_hyperparams = study.best_params
        # best_hyperparams['best_value'] = best_value
        best_hyperparams["epochs"] = objective.epochs
        best_hyperparams["filename"] = filename_dataset
        best_hyperparams["best_rmse"] = study.best_value
        best_hyperparams["rmse_valid_mean"] = objective.rmse_valid_mean
        best_hyperparams["eval_loss"] = objective.eval_loss
        run_end = timer.time()
        run_time = (run_end - run_start) / 60
        best_hyperparams["run_time"] = run_time

        for key, value in best_hyperparams.items():
            print("    {}: {}".format(key, value))

        param_filepath = f"{output_filepath}/{filename_dataset.replace('.h5', '.json')}"
        print(f"Storing best params into a file: {param_filepath}")
        with open(param_filepath, 'w') as f:
            json.dump(best_hyperparams, f)
        
        break
        # num = num + 1
        # if num == 14:
        #     break
    
# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     # Hyperparameters
#     parser.add_argument('--input_filepath',   type=str,   help='Path to the dataset file')
#     parser.add_argument('--output_filepath',  type=str,   help='Path to the result file')
#     parser.add_argument('--seed',             type=float, default=42,     help='Seed for reproducibility')
#     parser.add_argument('--feature',          type=str,   default='mua',  help='Type of spiking activity (sua or mua)')
#     parser.add_argument('--decoder',          type=str,   default='qrnn', help='Deep learning based decoding algorithm')
#     parser.add_argument('--n_folds',          type=int,   default=5,      help='Number of cross validation folds')
#     parser.add_argument('--min_train_size',   type=float, default=0.5,    help='Minimum (fraction) of training data size')
#     parser.add_argument('--test_size',        type=float, default=0.1,    help='Testing data size')
#     parser.add_argument('--verbose',          type=int,   default=0,      help='Wether or not to print the output')
#     parser.add_argument('--n_trials',         type=int,   default=2,      help='Number of trials for optimization')
#     parser.add_argument('--timeout',          type=int,   default=300,    help='Stop study after the given number of seconds')
#     parser.add_argument('--n_startup_trials', type=int,   default=1,      help='Number of trials for which pruning is disabled')
    
#     args = parser.parse_args()
    # main(args)
main()