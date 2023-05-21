from .periodic_activations import SineActivation, CosineActivation
from torch import nn

class Model(nn.Module):
    def __init__(self, activation, hiddem_dim):
        super(Model, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(116, hiddem_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, hiddem_dim)
        
        self.fc1 = nn.Linear(hiddem_dim, 2)
    
    def forward(self, x):
        #x = x.unsqueeze(1)
        x = self.l1(x)
        x = self.fc1(x)
        return x
