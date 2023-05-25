# basado en https://github.com/ojus1/Time2Vec-PyTorch

# import packages
import os
import numpy as np
import h5py
import platform
import torch
from torch.utils.data import DataLoader
## Time2Vector
from torch.utils.data import DataLoader
from time2vec.Pipeline import AbstractPipelineClass
from torch import nn
from time2vec.Model import ModelT2v
# tiempo
import time
# separar conjunto de datos: train test
import sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.validation import _num_samples

# print(f"Python (v{platform.python_version()})")
# print(f"Pytorch (v{torch.__version__})")

# para separar dataset en train y test
class TimeSeriesSplitCustom(TimeSeriesSplit):
    """
    Create time-series data cross-validation
    Ref: https://stackoverflow.com/questions/62210221/walk-forward-with-validation-window-for-time-series-data-cross-validation

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. 
    max_train_size : int, default=None
        Maximum size for a single training set.
    test_size : int, default=1
        Used to limit the size of the test set.
    min_train_size : int, default=1
        Minimum size of the training set.

    Returns
    ----------
    Indices of training and testing data.
    """
    def __init__(self, n_splits=5, max_train_size=None, test_size=1, min_train_size=1):
        super().__init__(n_splits=n_splits, max_train_size=max_train_size)
        self.test_size = test_size
        self.min_train_size = min_train_size

    def overlapping_split(self, X, y=None, groups=None):
        min_train_size = self.min_train_size
        test_size = self.test_size

        n_splits = self.n_splits
        n_samples = _num_samples(X)

        if (n_samples - min_train_size) / test_size >= n_splits:
            print('(n_samples -  min_train_size) / test_size >= n_splits')
            print('default TimeSeriesSplit.split() used')
            yield from super().split(X)

        else:
            shift = int(np.floor((n_samples - test_size - min_train_size) / (n_splits - 1)))
            start_test = n_samples - (n_splits * shift + test_size - shift)
            test_starts = range(start_test, n_samples - test_size + 1, shift)

            if start_test < min_train_size:
                raise ValueError(("The start of the testing : {0} is smaller"
                                    " than the minimum training samples: {1}.").format(start_test, min_train_size))

            indices = np.arange(n_samples)

            for test_start in test_starts:
                if self.max_train_size and self.max_train_size < test_start:
                    yield (indices[test_start - self.max_train_size:test_start],
                            indices[test_start:test_start + test_size])
                else:
                    yield (indices[:test_start],
                            indices[test_start:test_start + test_size])

class Dataset_for_embeddings(torch.utils.data.Dataset):
    def __init__(self, filepath_dataset: str, feature: str, velocity: bool =True, n_splits= 5):
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
        self.n_splits = n_splits
        with h5py.File(filepath_dataset, 'r') as f:
            self.X = f[f'X_{feature}'][()]
            self.y = f['y_task'][()]   
        if velocity:
            # select the x-y velocity components
            self.y = self.y[:,2:4] # data shape: n x 6 (x-y position, x-y velocity, x-y acceleration)

        assert len(self.X) == len(self.y)

        # separo dataset
        self.tscv = TimeSeriesSplitCustom(n_splits= self.n_splits, test_size=int(0.1*len(self.y)), min_train_size=int(0.5*len(self.y)))

    def split(self):
        return self.tscv.split(self.X, self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return len(self.X)
    
    # def make_ds_embeddings(self, model):
        
        
    
class Pipeline_t2v(AbstractPipelineClass):
    def __init__(self, model):
        self.model = model
        
    def train(self, ds,  batch_size, num_epochs, umbral_best):
        # Initializing the weights with a uniform distribution
        nn.init.uniform_(self.model.fc1.weight)
        ini = time.time()
        loss_fn = nn.MSELoss()
        train_dl = DataLoader(ds, batch_size, shuffle=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # optimizer= torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.9)

        loss_minor = 999999999999
        best_chp = 'not_found'        
        for ep in range(1, num_epochs+1):      
            for x, y in train_dl:
                optimizer.zero_grad()

                # no se necesita unsqueeze, porque y.shape = 32, 2
                # y_pred = self.model(x.unsqueeze(1).float())
                _, y_pred = self.model(x.float())
                loss = loss_fn(y_pred, y.float())

                loss.backward()
                optimizer.step()
                
            if loss.item() <= umbral_best and loss.item() < loss_minor:
                loss_minor = loss.item()
                best_chp = f"checkpoints/chp_{num_epochs}_{ep}_{round(loss_minor, 3)}"
                torch.save({
                    'epoch': ep,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,                    
                }, best_chp)
                
            print(f"\rep: {ep} de {num_epochs} - loss:{loss.item()}", end="")
        fin = time.time()
        print(f"\nloss:{loss.item()} - best_chp:{best_chp} - time:{fin-ini}")
        return best_chp, loss_minor
    
    def preprocess(self, x):
        return x
    
    def decorate_output(self, x):
        return x  
            
# main
def main(phases):
    # lectura datos
    # 'indy_20161005_06_baks.h5' es archivo más pequeño 4,8     
    filepath_dataset = 'indy_20161005_06_baks.h5'
    ds = Dataset_for_embeddings(f"datos/03_baks/{filepath_dataset}", "sua", n_splits= 5)

    model_embeddings = ModelT2v(activation= "sin", in_features= ds.X.shape[1], hiddem_dim= 64)
    batch_size = 32
    # genera input embeddings
    if 'train_t2v' in phases:
        pipe = Pipeline_t2v(model_embeddings)
        true_best_loss = 999999
        true_best_chp = ""
        umbral_best = 30
        print(f"Umbral: {umbral_best}")            
        # for e in range(170, 251, 10):    
        for e in range(210, 231, 5):    
            best_chp, best_loss = pipe.train(ds= ds, batch_size= batch_size, num_epochs= e, umbral_best= umbral_best)
            if best_loss < true_best_loss:
                true_best_loss = best_loss
                true_best_chp = best_chp
                
        print(f"best of best true_best_loss:{true_best_loss} true_best_chp:'{true_best_chp}'")
    else:
        true_best_chp = 'checkpoints/chp_225_190_12.751'
        
    if 'apply_t2v' in phases:
        drop_last = True
        checkpoint = torch.load(true_best_chp)
        model_embeddings.load_state_dict(checkpoint['model_state_dict'])
        model_embeddings.eval()
        
        eval_dl = DataLoader(ds, batch_size, shuffle=False)
        X_embeddings = []
        Y_target = []
        for x, y in eval_dl:
            x_embedding, _ = model_embeddings(x.float())
            X_embeddings.append(x_embedding)
            Y_target.append(y)
            
        if drop_last:
            if X_embeddings[-1].shape[0] != batch_size:
                X_embeddings.pop()
                Y_target.pop()
            else:
                # rellenar con ceros ?
                pass
            
        with h5py.File(f"ds_embeddings/{filepath_dataset[:-3]}_t2v.h5", 'w') as f:
            x_cat = torch.cat(X_embeddings)
            f['X_embeddings'] = x_cat.detach().cpu().numpy()
            y_cat = torch.cat(Y_target)
            f['Y_target'] = y_cat.detach().cpu().numpy()
            
    if 'informer' in phases:
        # class transformers.InformerConfig ############
        from transformers import InformerConfig, InformerModel

        # Initializing an Informer configuration 
        configuration = InformerConfig(prediction_length=12)
        """
        prediction_length (int) — The prediction length for the decoder. In other words, the prediction horizon of the model. This value is typically dictated by the dataset and we recommend to set it appropriately.
        context_length (int, optional, defaults to prediction_length) — The context length for the encoder. If None, the context length will be the same as the prediction_length.
        distribution_output (string, optional, defaults to "student_t") — The distribution emission head for the model. Could be either “student_t”, “normal” or “negative_binomial”.
        loss (string, optional, defaults to "nll") — The loss function for the model corresponding to the distribution_output head. For parametric distributions it is the negative log likelihood (nll) - which currently is the only supported one.
        input_size (int, optional, defaults to 1) — The size of the target variable which by default is 1 for univariate targets. Would be > 1 in case of multivariate targets.
        scaling (string or bool, optional defaults to "mean") — Whether to scale the input targets via “mean” scaler, “std” scaler or no scaler if None. If True, the scaler is set to “mean”.
        lags_sequence (list[int], optional, defaults to [1, 2, 3, 4, 5, 6, 7]) — The lags of the input time series as covariates often dictated by the frequency of the data. Default is [1, 2, 3, 4, 5, 6, 7] but we recommend to change it based on the dataset appropriately.
        num_time_features (int, optional, defaults to 0) — The number of time features in the input time series.
        num_dynamic_real_features (int, optional, defaults to 0) — The number of dynamic real valued features.
        num_static_categorical_features (int, optional, defaults to 0) — The number of static categorical features.
        num_static_real_features (int, optional, defaults to 0) — The number of static real valued features.
        cardinality (list[int], optional) — The cardinality (number of different values) for each of the static categorical features. Should be a list of integers, having the same length as num_static_categorical_features. Cannot be None if num_static_categorical_features is > 0.
        embedding_dimension (list[int], optional) — The dimension of the embedding for each of the static categorical features. Should be a list of integers, having the same length as num_static_categorical_features. Cannot be None if num_static_categorical_features is > 0.
        d_model (int, optional, defaults to 64) — Dimensionality of the transformer layers.
        encoder_layers (int, optional, defaults to 2) — Number of encoder layers.
        decoder_layers (int, optional, defaults to 2) — Number of decoder layers.
        encoder_attention_heads (int, optional, defaults to 2) — Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (int, optional, defaults to 2) — Number of attention heads for each attention layer in the Transformer decoder.
        encoder_ffn_dim (int, optional, defaults to 32) — Dimension of the “intermediate” (often named feed-forward) layer in encoder.
        decoder_ffn_dim (int, optional, defaults to 32) — Dimension of the “intermediate” (often named feed-forward) layer in decoder.
        activation_function (str or function, optional, defaults to "gelu") — The non-linear activation function (function or string) in the encoder and decoder. If string, "gelu" and "relu" are supported.
        dropout (float, optional, defaults to 0.1) — The dropout probability for all fully connected layers in the encoder, and decoder.
        encoder_layerdrop (float, optional, defaults to 0.1) — The dropout probability for the attention and fully connected layers for each encoder layer.
        decoder_layerdrop (float, optional, defaults to 0.1) — The dropout probability for the attention and fully connected layers for each decoder layer.
        attention_dropout (float, optional, defaults to 0.1) — The dropout probability for the attention probabilities.
        activation_dropout (float, optional, defaults to 0.1) — The dropout probability used between the two layers of the feed-forward networks.
        num_parallel_samples (int, optional, defaults to 100) — The number of samples to generate in parallel for each time step of inference.
        init_std (float, optional, defaults to 0.02) — The standard deviation of the truncated normal weight initialization distribution.
        use_cache (bool, optional, defaults to True) — Whether to use the past key/values attentions (if applicable to the model) to speed up decoding.
        attention_type (str, optional, defaults to “prob”) — Attention used in encoder. This can be set to “prob” (Informer’s ProbAttention) or “full” (vanilla transformer’s canonical self-attention).
        sampling_factor (int, optional, defaults to 5) — ProbSparse sampling factor (only makes affect when attention_type=“prob”). It is used to control the reduced query matrix (Q_reduce) input length.
        distil (bool, optional, defaults to True) — Whether to use distilling in encoder.
        """

        # Randomly initializing a model (with random weights) from the configuration
        model = InformerModel(configuration)

        # Accessing the model configuration
        configuration = model.config
        #########
    
    
    # https://huggingface.co/docs/transformers/model_doc/time_series_transformer
    # # train y test
    # for train_idx, test_idx in ds.split():
    #     # specify training set
    #     X_train = ds.X[train_idx,:]            
    #     y_train = ds.y[train_idx,:]
    #     print("X_train inicio:", len(X_train), " y_train inicio:", len(y_train))   
    #     # specify test set
    #     X_test = ds.X[test_idx,:]
    #     y_test = ds.y[test_idx,:]
    #     print("X_test inicio:", len(X_test), " y_test inicio:", len(y_test))        

    # return



# main(['train_t2v', 'apply_t2v'])
main(['apply_t2v'])
    


