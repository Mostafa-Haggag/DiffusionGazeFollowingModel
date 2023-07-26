from torch import nn
import torch.nn.functional as F

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        # if we assume x has size of (96,256,128)--->(96,256,1024)
        x, gate = self.proj(x).chunk(2, dim=-1)
        # Chuncking functions will turn this vector into 2 vectors 96,256,512
        return x * F.gelu(gate)