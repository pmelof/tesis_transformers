"""
Optimize hyperparameters for deep learning based BMI decoders using Optuna
"""

# import packages
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import json
import h5py
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
# from bmi.preprocessing import TimeSeriesSplitCustom, transform_data
# from bmi.utils import seed_tensorflow
# from bmi.decoders import QRNNDecoder, LSTMDecoder, MLPDecoder
# from tensorflow.keras.callbacks import EarlyStopping
# from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState
import time as timer
from transformers.process_input import readDataset, DatasetTransformers


    
from torch.utils.data import DataLoader
from transformers import splitDataset, train, eval
from model import TransformerModel
import torch
from process_input import generateBigVocabulary


# Leer archivo tokenizado 
# Generar vocabulario (?), o podría traerlo como parámetro 
# Definir variables a utilizar: filename_dataset, device, decimal, dir_datasets, vocabulary
# (podrían ser pasados por argumentos).
# Variables que se necesitan definir aquí: max_timesteps (?), max_layers, max_d_model, train_ds, eval_ds
# Separar en 5 ventanas el dataset (por el momento no, así que ventana = 1)
# Separar en train y test (80-20) por cada ventana, aquí solo ocupo train
# y lo vuelvo a separar en train y eval (90-10) con splitDataset.
# La parte de train volver a separar en train y eval (80-20)
# Normalizar si es necesario los datos (por el momento probar sin esto, sino explota)
# En Ahmadi usan algo para dejar los datos secuenciales, posibilidad de aplicarlo también
# Entrenar el modelo con train y eval
# Evaluar el modelo con X_test y obtener Y_test_pred
# Comparar Y_test_pred con Y_test.
# Calcular RMSE y CC.

def main(args):
    
    # Variables que defino aquí o ingreso por parámetros, por el momento serán definidas aquí:
    filename_dataset = 'indy_20161005_06_baks.h5'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decimal=1   # decimal al que se desea redondear
    dir_datasets = './datos/05_rounded'
    vocabulary, _, _ = generateBigVocabulary(dir_datasets=dir_datasets, decimal=decimal)
    
    print(f"Optimizando hiperparámetros para el archivo {filename_dataset} redondeado")
    run_start = timer.time()
    
    # Ira dentro de objetive? ----------------------------------
    # max_timesteps = 5
    max_layers = 2
    max_d_model = 120 # d_model
    train_ds, eval_ds, _ = splitDataset(f"{dir_datasets}/{filename_dataset[:-3]}_rounded_{decimal}.h5", "sua", decimal, velocity=True, scaled=False)
    # ----------------------------------------------------------
    
    def objective(trial):
        config = {           
            "input_dim" : len(train_ds[0][0]), # dimensión de entrada de la base de datos. ej = 116
            "output_dim": len(train_ds[0][1]), # dimensión de salida de la base de datos. ej = 2
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
            # "timesteps": trial.suggest_int("timesteps", 1, max_timesteps),
            # "window_size": 2,
            }
        print(config)  
        
        # Tal vez llamar splitDataset aquí, en un futuro agregar la división en 5 ventanas.
        train_dl = DataLoader(train_ds, config['batch_size'], shuffle=False)
        eval_dl = DataLoader(eval_ds, config['batch_size'], shuffle=False)
        
        # Podría normalizar y secuencializar como Ahmadi en un futuro
        
        # Paradas anticipadas
        # # early stopping callback
        # earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=args.verbose, mode='min', restore_best_weights=True)
        # # pruning trial callback
        # pruning = TFKerasPruningCallback(trial, 'val_loss')
        # # define callbacks
        # callbacks = [earlystop, pruning]
        
        model = TransformerModel(config['input_dim'], config['output_dim'], config['n_token']/
            config['d_model'], config['d_hid'], config['n_layers'], config['n_head'], config['dropout']).to(device)

        # fit model
        train_start = timer.time()
        train(model, epochs=config['epochs'], batch_size=config['batch_size'], train_dl=train_dl, optimizer=config['optimizer'], learning_rate=config['learning_rate'])
        train_end = timer.time()
        train_time = (train_end - train_start) / 60
        
        # if earlystop.stopped_epoch != 0:
        #     stop_epoch = earlystop.stopped_epoch + 1
        # else:
        #     stop_epoch = config['epochs']
        
        best_model_params_path = os.path.join("transformers/best_params", "best_model_params.pt")
        model.load_state_dict(torch.load(best_model_params_path)) # load best model states
    
    
    run_start = timer.time()
    
    print(f"Reading dataset from file: {args.input_filepath}")
    X, Y = readDataset(args.input_filepath, args.feature, args.velocity)
    ds = DatasetTransformers(X, Y, args.decimal)
    
    def objective(trial):
        # define hyperparameter space
        if args.decoder == 'qrnn':
            max_timesteps = 5
            max_layers = 1
            max_units = 600
        elif args.decoder == 'lstm':
            max_timesteps = 5
            max_layers = 1
            max_units = 250
        elif args.decoder == 'mlp':
            max_timesteps = 1
            max_layers = 3
            max_units = 400
        config = {"input_dim": X.shape[-1],
                  "output_dim": y.shape[-1],
                  "timesteps": trial.suggest_int("timesteps", 1, max_timesteps),
                  "n_layers": trial.suggest_int("n_layers", 1, max_layers),
                  "units": trial.suggest_int("units", 50, max_units, step=50),
                  "window_size": 2,
                  "batch_size": trial.suggest_categorical("batch_size", [32, 64, 96]),
                  "epochs": 100,
                  "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
                  "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
                  "optimizer": trial.suggest_categorical("optimizer", ['Adam', 'RMSProp']),
                  "loss": 'mse',
                  "metric": 'mse'}
        print(f"Hyperparameter configuration: {config}")

        # set seed for reproducibility
        seed_tensorflow(args.seed)
        
        rmse_valid_folds = []
        best_epochs = []
        
        # Separar en ventanas, pero aún no, mientras solo separo en train y test del 100% de datos
        limit_train = int(len(ds.X)*.8)
        X_train = ds.X[:limit_train]
        Y_train = ds.Y[:limit_train]
        
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=args.test_size, shuffle=False)

        # Posibilidad de normalizar aquí... pero no lo haré (aún).
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_valid = scaler.transform(X_valid)
        
        # Poner como datos secuenciales igual que Ahmadi
        # transform data into sequence data
        X_train, y_train = transform_data(X_train, y_train, timesteps=config["timesteps"])
        X_valid, y_valid = transform_data(X_valid, y_valid, timesteps=config["timesteps"])
        
        # Definir parada anticipada
        # early stopping callback
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=args.verbose, mode='min', restore_best_weights=True)
        # pruning trial callback
        pruning = TFKerasPruningCallback(trial, 'val_loss')
        # define callbacks
        callbacks = [earlystop, pruning]

        # Correr modelo Transformers
        
            # Create and compile model
            print("Compiling and training a model")
            if args.decoder == 'qrnn':
                model = QRNNDecoder(config)
            elif args.decoder == 'lstm':
                model = LSTMDecoder(config)
            elif args.decoder == 'mlp':
                model = MLPDecoder(config)
            # fit model
            train_start = timer.time()
            history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=config['epochs'], verbose=args.verbose, callbacks=callbacks)
            train_end = timer.time()
            train_time = (train_end - train_start) / 60
    
            if earlystop.stopped_epoch != 0:
                stop_epoch = earlystop.stopped_epoch + 1
            else:
                stop_epoch = config['epochs']
            best_epoch = np.argmin(history.history['val_loss']) + 1
            best_epochs.append(best_epoch)
            print(f"Training stopped at epoch {stop_epoch} with the best epoch at {best_epoch}")
            print(f"Training the model took {train_time:.2f} minutes")

            # predict using the trained model
            y_valid_pred = model.predict(X_valid, batch_size=config['batch_size'], verbose=args.verbose)

            # evaluate performance
            print("Evaluating the model performance")
            rmse_valid = mean_squared_error(y_valid, y_valid_pred, squared=False)
            rmse_valid_folds.append(rmse_valid)

        epochs = int(np.asarray(best_epochs).mean())
        objective.epochs = epochs
        rmse_valid_mean = np.asarray(rmse_valid_folds).mean()
        return rmse_valid_mean

    # create and optimize study
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials))
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    print(f"Storing study trial into a file: {args.output_filepath}")
    with open(args.output_filepath, 'wb') as f:
        pickle.dump(study, f)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best hyperparameters:")
    best_params = study.best_trial.params
    best_params["epochs"] = objective.epochs
    for key, value in best_params.items():
        print("    {}: {}".format(key, value))

    param_filepath = f"{args.output_filepath.split('.')[0]}.json"
    print(f"Storing best params into a file: {param_filepath}")
    with open(param_filepath, 'w') as f:
        json.dump(best_params, f)

    run_end = timer.time()
    run_time = (run_end - run_start) / 60
    print (f"Whole processes took {run_time:.2f} minutes")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--input_filepath',   type=str,   help='Path to the dataset file')
    parser.add_argument('--output_filepath',  type=str,   help='Path to the result file')
    parser.add_argument('--seed',             type=float, default=42,     help='Seed for reproducibility')
    parser.add_argument('--feature',          type=str,   default='mua',  help='Type of spiking activity (sua or mua)')
    parser.add_argument('--decoder',          type=str,   default='qrnn', help='Deep learning based decoding algorithm')
    parser.add_argument('--n_folds',          type=int,   default=5,      help='Number of cross validation folds')
    parser.add_argument('--min_train_size',   type=float, default=0.5,    help='Minimum (fraction) of training data size')
    parser.add_argument('--test_size',        type=float, default=0.1,    help='Testing data size')
    parser.add_argument('--verbose',          type=int,   default=0,      help='Wether or not to print the output')
    parser.add_argument('--n_trials',         type=int,   default=2,      help='Number of trials for optimization')
    parser.add_argument('--timeout',          type=int,   default=300,    help='Stop study after the given number of seconds')
    parser.add_argument('--n_startup_trials', type=int,   default=1,      help='Number of trials for which pruning is disabled')
    
    args = parser.parse_args()
    main(args)
