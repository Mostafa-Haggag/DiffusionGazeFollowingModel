from torch import nn
from einops import rearrange
from .utils import Normalize
from .basictransformerblock import BasicTransformerBlock

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, 
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels 
        d_head = in_channels // n_heads
        # dimension per head times number of heads 
        inner_dim = n_heads * d_head
        # group norm
        self.norm = Normalize(in_channels) 
        

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
    
        # the number of transformers that i want to use. right now
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        ) 
        # project back to the normal space
        self.proj_out = nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0)  

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        # you get an input of 96,128,16,16, 128 is the number of features 
        b, c, h, w = x.shape
        x_in = x
        # you are doing a group norm with a group of 32
        x = self.norm(x) 
        # project input into high dimension space 
        x = self.proj_in(x)
        # make shape 96 ,256,128
        x = rearrange(x, 'b c h w -> b (h w) c') 
        for block in self.transformer_blocks:
            x = block(x, context=context)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in