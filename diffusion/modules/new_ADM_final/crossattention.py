from einops import rearrange, repeat################
from torch import nn, einsum
import torch
from .utils import exists,default

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.query_dim=query_dim#128 dependong on the unbnenet
        self.heads=heads#8
        self.dim_head=dim_head#16
        inner_dim = dim_head * heads
        self.inner_dim=inner_dim#128
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        
        '''
        The scale parameter is set to dim_head ** -0.5, which is a scaling factor used to prevent the dot product of the query and key from getting too larg
        '''
        
        self.heads = heads
        
        '''
        The heads parameter is saved to the instance variable self.heads.
        '''
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        # depend on the size of the inner dim and context vector 
        '''
        the module initializes several linear layers (self.to_q, self.to_k, and self.to_v) that are used to project 
        the query, key, and value tensors to the dimension of the attention heads. The size of the output of these 
        linear layers is inner_dim, which is equal to dim_head * heads.
        '''
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        # The to_out sequential module is used to project 
        # the concatenated attention head vectors to the 
        # output dimension query_dim. It consists of a linear layer followed by a dropout layer.


    def forward(self, x, context=None):
        
        '''
        . It takes in an input tensor x, which is the query tensor, 
        and an optional context tensor. If the context tensor is not provided, the method assumes the context tensor to be the same as the query tensor. 
        
        '''
        h = self.heads

        q = self.to_q(x)
        # First, the query tensor x is linearly transformed to a tensor q using the weight matrix self.to_q
        context = default(context, x)# 96,16,1280 this allow to make this module work
        # for cross attention and self attention
        # 96,2,1024
        k = self.to_k(context) 
        v = self.to_v(context) 

        # Similarly, the context tensor is linearly transformed to tensors k and v using the weight matrices self.to_k and self.to_v, respectively. 
        # go from (96,256,8*16) to (96*8,256,16)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
   
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)