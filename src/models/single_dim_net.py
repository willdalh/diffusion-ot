import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import timestep_encoding

single_dim_block = lambda ic, oc: nn.Sequential(
    nn.Linear(ic, oc),
    nn.LeakyReLU(),
)

class SingleDimNet(nn.Module):
    """
    Simple fully connected network for data with only one shape dimension
    """
    def __init__(self, in_features, out_features, data_shape, device="cuda" if torch.cuda.is_available() else "cpu", n_T = 1000):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.data_shape = data_shape
        self.n_T = n_T

        self.model = nn.Sequential(
            single_dim_block(in_features+1, 64),
            single_dim_block(64, 128),
            single_dim_block(128, 64),
            single_dim_block(64, out_features),
        )

        self.to(device)

    def forward(self, x, t):
        tt_full = t.unsqueeze(-1).type(torch.float32)
        tt = tt_full / self.n_T

        for i, (name, layer) in enumerate(self.model.named_children()):
            if i == 0:
                x = layer(torch.cat([x, tt], dim=-1))
            else:
                x = layer(x)
        return x
        # return self.model(torch.cat([x, tt], dim=-1))
    

        
        