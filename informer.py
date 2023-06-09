from transformers.transformers import InformerConfig, InformerModel
import h5py
import time
from torch.utils.data import DataLoader

    # def __init__(self, filepath_dataset: str, n_splits= 5):
    #     self.n_splits = n_splits


        # separo dataset
        # self.tscv = TimeSeriesSplitCustom(n_splits= self.n_splits, test_size=int(0.1*len(self.y)), min_train_size=int(0.5*len(self.y)))

    # def split(self):
    #     return self.tscv.split(self.X, self.y)

def get_ds_train_eval(filepath_dataset, porc_train):
    """
    test_dl ? = ...
    no me acuerdo cuando se usaba o para que se usaba
    decidimos solo usar train y eval
    """
    with h5py.File(filepath_dataset, 'r') as f:
        X = f['X_embeddings'][()]
        Y = f['Y_target'][()]   
        
        len_train = round(len(X) * porc_train, 0)
        ds_train_X = X[:len_train]
        ds_train_Y = Y[:len_train]
        ds_eval_X = X[len_train:]
        ds_eval_Y = Y[len_train:]
        
        return ds_train_X, ds_train_Y, ds_eval_X, ds_eval_Y


class Dataset_for_informer(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        assert len(self.X) == len(self.Y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def __len__(self):
        return len(self.X)
    
ds_train_X, ds_train_Y, ds_eval_X, ds_eval_Y = get_ds_train_eval(filepath_dataset= "datos/04_t2v/indy_20161005_06_baks_t2v.h5", porc_train= 0.8)
ds_train = Dataset_for_informer(ds_train_X, ds_train_Y)
ds_eval = Dataset_for_informer(ds_eval_X, ds_eval_Y)

batch_size = 32
train_dl = DataLoader(ds_train, batch_size, shuffle=False)
eval_dl = DataLoader(ds_eval, batch_size, shuffle=False)
    
"""
en conversacion VM+PM  25-05-2023 22:50
haciendo un simil con transformer de texto. en toxto
entra una frase y sale otra frase
cada frase es de un tamaño arbitrario, no necesariamente son todas del mismolar.  tipicamente se define un tamaño max y se reellena con ceros o se trunca

nuestro mono pódemos ver cada frase como cada alcance de objetivo
sin embargo el dato de cuando se cambio el objetivo en este momento no lo tenemos 
entonces no sabemos donde termina una frase y donde comienza otra
vamos a asumir un tamaño arbitrario de cada frase y supondremos que todas las frases son del mismo tamaño
claramente esto no es real y el resultado puede ser cualquier cosa

nuestro objetivo en este momento es usar el traforme y obtener un resultado
posteriormente se deberia entrenar con las frases correctas , para ello se debe agregar a cada embedding un id de objetivo
la idea es identificar todos los embeddings que persiguen el mismo objetivo
y estos constituirian una frase

"""    
    
# Initializing an Informer
# vm=> 2 porque son dos los datos a predecir Vx, Vy
configuration = InformerConfig(
    prediction_length = 2,
    # context_length = 2, # en time2vec/Model.py -> ModelT2v line12  => self.fc1 = nn.Linear(hiddem_dim, 2)
    input_size = 2,
)
# Accessing the model configuration
# configuration = model_informer.config

#------------------------------------------
# Randomly initializing a model (with random weights) from the configuration
model_informer = InformerModel(configuration)


def train(model: InformerModel, num_epochs: int):
    # Initializing the weights with a uniform distribution
    model.init_weights()
    ini = time.time()

    loss_minor = 999999999999
    best_chp = 'not_found'        
    for ep in range(1, num_epochs+1):
        batch_size = 64
        sequence_length = 64 # creo que es size nuestro X embedding
        input_size = 2 # yo creo Y size
        # num_features here is equal to config.num_time_features+config.num_dynamic_real_features
        num_features = 0
        batch = {
            "past_values" : torch.zeros([batch_size, sequence_length, input_size], dtype=torch.float32 ),
            "past_time_features" : torch.zeros([batch_size, sequence_length, num_features], dtype=torch.float32 ),
            "past_observed_mask" : torch.zeros([batch_size, sequence_length, input_size], dtype=torch.bool ),
            
            # paty
            # past_values == X
            
            
            # inputs_embeds (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size), optional) — Optionally, instead of passing input_ids you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert input_ids indices into associated vectors than the model’s internal embedding lookup matrix.
            
            
            # static_categorical_features: Optional[torch.Tensor] = None,
            # static_real_features: Optional[torch.Tensor] = None,
            # future_values: Optional[torch.Tensor] = None,
            # future_time_features: Optional[torch.Tensor] = None,
        }
        for x, y in train_dl:
            # during training, one provides both past and future values
            # as well as possible additional features
            outputs = model(
                past_values=batch["past_values"],
                past_time_features=batch["past_time_features"],
                past_observed_mask=batch["past_observed_mask"],
                # static_categorical_features=batch["static_categorical_features"],
                # static_real_features=batch["static_real_features"],
                # future_values=batch["future_values"],
                # future_time_features=batch["future_time_features"],
            )


        
        
            loss = outputs.loss
            loss.backward()

            
        # if loss.item() <= umbral_best and loss.item() < loss_minor:
        #     loss_minor = loss.item()
        #     best_chp = f"checkpoints/chp_{num_epochs}_{ep}_{round(loss_minor, 3)}"
        #     torch.save({
        #         'epoch': ep,
        #         'model_state_dict': self.model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': loss,                    
        #     }, best_chp)
            
        print(f"\rep: {ep} de {num_epochs} - loss:{loss.item()}", end="")
    fin = time.time()
    print(f"\nloss:{loss.item()} - best_chp:{best_chp} - time:{fin-ini}")
    return best_chp, loss_minor    

# ---------------------
# InformerForPrediction
from huggingface_hub import hf_hub_download
import torch
from transformers.transformers import InformerForPrediction

file = hf_hub_download(
    repo_id="kashif/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

model = InformerForPrediction.from_pretrained("huggingface/informer-tourism-monthly")

# during training, one provides both past and future values
# as well as possible additional features
outputs = model(
    past_values=batch["past_values"],
    past_time_features=batch["past_time_features"],
    past_observed_mask=batch["past_observed_mask"],
    static_categorical_features=batch["static_categorical_features"],
    static_real_features=batch["static_real_features"],
    future_values=batch["future_values"],
    future_time_features=batch["future_time_features"],
)

loss = outputs.loss
loss.backward()

# during inference, one only provides past values
# as well as possible additional features
# the model autoregressively generates future values
outputs = model.generate(
    past_values=batch["past_values"],
    past_time_features=batch["past_time_features"],
    past_observed_mask=batch["past_observed_mask"],
    static_categorical_features=batch["static_categorical_features"],
    static_real_features=batch["static_real_features"],
    future_time_features=batch["future_time_features"],
)

mean_prediction = outputs.sequences.mean(dim=1)