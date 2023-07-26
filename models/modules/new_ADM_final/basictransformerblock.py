from torch import nn
from .feedforward import FeedForward
from .crossattention import CrossAttention
# n_heads
# n_heads
# context_dim
class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        # is a self-attention
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # This is cross attnetion
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
    def forward(self, x, context=None):
        # layer normalization then self attention then residual connection still the size is (96,256,128)
        x = self.attn1(self.norm1(x))+ x
        # cross attention with normalized query then residual connection
        x = self.attn2(self.norm2(x), context=context) + x
        # normalization of output of previous step then feedforwards network then residual connection
        x = self.ff(self.norm3(x)) + x
        return x