import torch
import math
import numpy as np 
import random
import torch.nn.functional as F
from config import get_config
from gaze.datasets import get_dataset
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from math import prod
from omegaconf import OmegaConf

def unravel_index(
    indices: torch.LongTensor,
    shape,
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord
def batch_argmax(tensor, batch_dim=1):
    """
    Assumes that dimensions of tensor up to batch_dim are "batch dimensions"
    and returns the indices of the max element of each "batch row".
    More precisely, returns tensor `a` such that, for each index v of tensor.shape[:batch_dim], a[v] is
    the indices of the max element of tensor[v].
    """
    if batch_dim >= len(tensor.shape):
        raise NoArgMaxIndices()
    batch_shape = tensor.shape[:batch_dim]
    non_batch_shape = tensor.shape[batch_dim:]
    flat_non_batch_size = prod(non_batch_shape)
    tensor_with_flat_non_batch_portion = tensor.reshape(*batch_shape, flat_non_batch_size)

    dimension_of_indices = len(non_batch_shape)

    # We now have each batch row flattened in the last dimension of tensor_with_flat_non_batch_portion,
    # so we can invoke its argmax(dim=-1) method. However, that method throws an exception if the tensor
    # is empty. We cover that case first.
    if tensor_with_flat_non_batch_portion.numel() == 0:
        # If empty, either the batch dimensions or the non-batch dimensions are empty
        batch_size = prod(batch_shape)
        if batch_size == 0:  # if batch dimensions are empty
            # return empty tensor of appropriate shape
            batch_of_unraveled_indices = torch.ones(*batch_shape, dimension_of_indices).long()  # 'ones' is irrelevant as it will be empty
        else:  # non-batch dimensions are empty, so argmax indices are undefined
            raise NoArgMaxIndices()
    else:   # We actually have elements to maximize, so we search for them
        indices_of_non_batch_portion = tensor_with_flat_non_batch_portion.argmax(dim=-1)
        batch_of_unraveled_indices = unravel_index(indices_of_non_batch_portion, non_batch_shape)

    if dimension_of_indices == 1:
        # above function makes each unraveled index of a n-D tensor a n-long tensor
        # however indices of 1D tensors are typically represented by scalars, so we squeeze them in this case.
        batch_of_unraveled_indices = batch_of_unraveled_indices.squeeze(dim=-1)
    return batch_of_unraveled_indices


class NoArgMaxIndices(BaseException):

    def __init__(self):
        super(NoArgMaxIndices, self).__init__(
            "no argmax indices: batch_argmax requires non-batch shape to be non-empty")
def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    Interpolate between time steps as many time steps that i want 
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001*8
    beta_end = scale * 0.02*8
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
### Function for extracting 
def extract(a, t, x_shape):
    # tensor of size a  1000
    # t is the time steps that i am sampling from. they are between 0 to 1000. my batch size is 32 so i have 32 time steps 
    b, *_ = t.shape# 
    
    # extract all the different values for this specific index has shape of 32
    out = a.gather(-1, t) 
    # from the 1000 tensor called a i want to extract the time steps passed by in the tensor t 
    # reshape into (32,1,1,1) depends mainly on x_shape
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) 
    
def exists(x):
    # the value is not set to none = we have a value
    return x is not None  

def default(val, d):
    # return the value as it is OR call d or return d
    if exists(val):
        return val 
    return d() if callable(d) else d 
# normalization functions
def normalize_to_neg_value_to_value(img,value):
    return img * 2 * value - value
#################################
def unnormalize_to_neg_value_to_value(t,value):
    return (t + value) * 1/(2*value)
########### Normalize negative 0.1 to 0.1
def normalize_to_neg_point_one_to_point_one(img):
    return img * 0.2 - 0.1
def unnormalize_to_neg_point_one_to_point_one(t):
    return (t + 0.1) * 1/0.2

########### Normalize negative 2 to 2
def normalize_to_neg_two_to_two(img):
    return img * 4 - 2
def unnormalize_neg2_to_zero_to_one(t):
    return (t + 2) * 0.25

########### Normalize negative 1 to 1
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5
###########
def identity(t, *args, **kwargs):
    return t
def q_sample( x_start, t,sqrt_alphas_cumprod,sqrt_one_minus_alphas_cumprod ,noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        ) # caluclating the varianceeeeee *noise +mean *x0 = posterioier .
        
from utils import get_head_mask, get_label_map,get_heatmap_peak_coords

def plottings(schedular_name,step):
    timesteps = 1000
    if(schedular_name=='linear'):
        betas = linear_beta_schedule(timesteps)
    elif(schedular_name=='cosine'): 
        betas = cosine_beta_schedule(timesteps)
    elif(schedular_name=='sigmoid'): 
        betas = sigmoid_beta_schedule(timesteps)
    # print(betas)
    alphas = 1. - betas 
    alphas_cumprod = torch.cumprod(alphas, dim=0)  
    # alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
    timesteps, = betas.shape
    num_timesteps = int(timesteps) 
    sqrt_alphas_cumprod           = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    x1,x2,x3=get_config()
    config = OmegaConf.merge(x2, x3)
    # config=get_config()
    source_loader, _ = get_dataset(config)
    the_source = iter(source_loader)
    data = next(the_source)
    t=torch.arange(0,num_timesteps,step).long()
    heatmap_new = data[3]
    heatmap_new_2 = data[5].squeeze()
    # print(heatmap_new.shape)
    data=batch_argmax(heatmap_new,1)
    print(data)
    print(data/64)
    # print(heatmap_new[0].shape)
    # print(heatmap_new[0])
    # print(get_heatmap_peak_coords(heatmap_new[1].detach().cpu().numpy()))
    print(heatmap_new_2)
    print(heatmap_new_2-data/64)

if __name__ == "__main__":
    # Make runs repeatable as much as possible
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    timesteps = 1000
    betas = linear_beta_schedule(timesteps)
    plottings('linear',32)