from transformers import InformerConfig, InformerModel
import h5py

class Dataset_for_informer(torch.utils.data.Dataset):
    def __init__(self, filepath_dataset: str, n_splits= 5):
        self.n_splits = n_splits
        with h5py.File(filepath_dataset, 'r') as f:
            self.X = f['X_embeddings'][()]
            self.Y = f['Y_target'][()]   

        assert len(self.X) == len(self.Y)

        # separo dataset
        # self.tscv = TimeSeriesSplitCustom(n_splits= self.n_splits, test_size=int(0.1*len(self.y)), min_train_size=int(0.5*len(self.y)))

    # def split(self):
    #     return self.tscv.split(self.X, self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def __len__(self):
        return len(self.X)
    
    
# Initializing an Informer
# vm=> 2 porque son dos los datos a predecir Vx, Vy
configuration = InformerConfig(
    prediction_length = 2,
    context_length = 2, # en time2vec/Model.py -> ModelT2v line12  => self.fc1 = nn.Linear(hiddem_dim, 2)
    # distribution_output: str = "student_t",
    # loss: str = "nll",
    # input_size: int = 1,
    # lags_sequence: List[int] = None,
    # scaling: Optional[Union[str, bool]] = "mean",
    # num_dynamic_real_features: int = 0,
    # num_static_real_features: int = 0,
    # num_static_categorical_features: int = 0,
    # num_time_features: int = 0,
    # cardinality: Optional[List[int]] = None,
    # embedding_dimension: Optional[List[int]] = None,
    # d_model: int = 64,
    # encoder_ffn_dim: int = 32,
    # decoder_ffn_dim: int = 32,
    # encoder_attention_heads: int = 2,
    # decoder_attention_heads: int = 2,
    # encoder_layers: int = 2,
    # decoder_layers: int = 2,
    # is_encoder_decoder: bool = True,
    # activation_function: str = "gelu",
    # dropout: float = 0.05,
    # encoder_layerdrop: float = 0.1,
    # decoder_layerdrop: float = 0.1,
    # attention_dropout: float = 0.1,
    # activation_dropout: float = 0.1,
    # num_parallel_samples: int = 100,
    # init_std: float = 0.02,
    # use_cache=True,
    # # Informer arguments
    # attention_type: str = "prob",
    # sampling_factor: int = 5,
    # distil: bool = True,
    # **kwargs,    
)

# Randomly initializing a model (with random weights) from the configuration
model = InformerModel(configuration)

# Accessing the model configuration
configuration = model.config

# ---------------------
# The InformerModel forward method, overrides the __call__ special method.
from huggingface_hub import hf_hub_download
import torch
from transformers import InformerModel

file = hf_hub_download(
    repo_id="kashif/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

model = InformerModel.from_pretrained("huggingface/informer-tourism-monthly")


    
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

last_hidden_state = outputs.last_hidden_state