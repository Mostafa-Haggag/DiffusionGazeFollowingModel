import torch
import math
import numpy as np 
import random
import torch.nn.functional as F
from config import get_config
from gaze.datasets import get_dataset
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from omegaconf import OmegaConf

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
        
from utils import get_head_mask, get_label_map
def understanding_bug(schedular_name,step):
    timesteps = 1000
    if(schedular_name=='linear'):
        betas = linear_beta_schedule(timesteps)
    elif(schedular_name=='cosine'): 
        betas = cosine_beta_schedule(timesteps)
    elif(schedular_name=='sigmoid'): 
        betas = sigmoid_beta_schedule(timesteps)
    alphas = 1. - betas 
    alphas_cumprod = torch.cumprod(alphas, dim=0)  
    # alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
    timesteps, = betas.shape
    num_timesteps = int(timesteps) 
    sqrt_alphas_cumprod           = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    x1,x2,x3=get_config()
    config = OmegaConf.merge(x2, x3)
    source_loader, _ = get_dataset(config)
    source_iter = iter(source_loader)
    n_iter = len(source_loader)
    t=torch.arange(0,num_timesteps,step).long()
    print(t)
    plotting_data = []
    for batch in range(n_iter):# iterate for number of things
            data_source = next(source_iter)
            (
                s_rgb,
                s_heads,# the face
                s_masks,
                s_gaze_heatmaps,
                _,
                gaze_points,
                s_gaze_inside,# boolean yes or no
                _,
                _,coordinates_train
            ) = data_source
            print(gaze_points.squeeze().shape)
            heatmap_new = normalize_to_neg_value_to_value(gaze_points.squeeze(),3)
            # print(heatmap_new.shape)
            noise = torch.randn_like(heatmap_new)
            new_heatmap = heatmap_new.repeat(t.shape[0],1)
            # print(t.shape)
            print(noise.shape)
            new_noise = noise.repeat(t.shape[0],1)
            # print(new_noise.shape)
            x = q_sample(new_heatmap,
                 t,
                 sqrt_alphas_cumprod,
                 sqrt_one_minus_alphas_cumprod, 
                 noise = new_noise
                 )
            x  = x / x.std(axis=(1), keepdims=True) 
            x = torch.clamp(x, min=-1 * 3, max=3)
            x = unnormalize_to_neg_value_to_value(x,3)
            sigma = 3
            my_list = []
            for gaze_x, gaze_y in x:
                gaze_heatmap_i = torch.zeros(64, 64)

                gaze_heatmap = get_label_map(
                gaze_heatmap_i, [gaze_x * 64, gaze_y * 64], sigma, pdf="Gaussian"
                )
                my_list.append(gaze_heatmap)            
            plotting_data.append(torch.stack(my_list,0))


if __name__ == "__main__":
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    timesteps = 1000
    betas = linear_beta_schedule(timesteps)
    understanding_bug('cosine',32)