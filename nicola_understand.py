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
# def wrap_clamp_tensor(input_tensor, min_val, max_val):
#     """
#     Clamp a PyTorch tensor to the range [min_val, max_val] by wrapping it around.
    
#     Args:
#     input_tensor (torch.Tensor): The input tensor to be clamped.
#     min_val (float): The minimum value of the desired range.
#     max_val (float): The maximum value of the desired range.
    
#     Returns:
#     torch.Tensor: The clamped tensor within the specified range.
#     """
#     print(input_tensor.shape)
#     if (input_tensor> min_val) and (input_tensor< max_val):
#         range_size = max_val - min_val
#         return min_val + ((input_tensor - min_val) % range_size)
#     else:
#         return input_tensor
def wrap_clamp_tensor(input_tensor, min_val, max_val):
    """
    Clamp a PyTorch tensor along the first dimension to the range [min_val, max_val] by wrapping it around.
    
    Args:
    input_tensor (torch.Tensor): The input tensor to be clamped.
    min_val (float): The minimum value of the desired range.
    max_val (float): The maximum value of the desired range.
    
    Returns:
    torch.Tensor: The clamped tensor within the specified range.
    """
    
    # Create a condition for values outside the range
    lower_bound_condition = input_tensor < min_val
    upper_bound_condition = input_tensor > max_val

    # Calculate range size
    range_size = max_val - min_val

    # Apply clamping element-wise
    clamped_tensor = input_tensor.clone()
    clamped_tensor[lower_bound_condition] = min_val + ((input_tensor[lower_bound_condition] - min_val) % range_size)
    clamped_tensor[upper_bound_condition] = min_val + ((input_tensor[upper_bound_condition] - min_val) % range_size)
    
    return clamped_tensor
from utils import get_head_mask, get_label_map

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
    # config=get_config()
    x1,x2,x3=get_config()
    config = OmegaConf.merge(x2, x3)
    source_loader, _ = get_dataset(config)
    the_source = iter(source_loader)
    data = next(the_source)
    # print(data[3].shape)
    # print(data[5].shape)
    t=torch.arange(0,num_timesteps,step).long()
    print(t)
    plotting_data = []
    normal_range=11
    myvalue= np.arange(1.0, 12.0, 1.0).tolist()
    for i in range(normal_range):
        # print(data[5][2].squeeze().shape)
        heatmap_new = normalize_to_neg_value_to_value(data[5][2].squeeze(),myvalue[i])
        noise = torch.randn_like(heatmap_new)
        new_heatmap = heatmap_new.repeat(t.shape[0],1)
        new_noise = noise.repeat(t.shape[0],1)

        x = q_sample(new_heatmap,
                 t,
                 sqrt_alphas_cumprod,
                 sqrt_one_minus_alphas_cumprod, 
                 noise = new_noise
                 )
        # print(x.shape)
        x  = x / x.std(axis=(1), keepdims=True) 
        # print(x)
        # x = torch.clamp(x, min=-1 * myvalue[i], max=myvalue[i])
        x = wrap_clamp_tensor(x, -1*myvalue[i], myvalue[i])

        # print(x)
        x = unnormalize_to_neg_value_to_value(x,myvalue[i])
        # print(x)
        sigma = 3
        my_list = []
        for gaze_x, gaze_y in x:
            gaze_heatmap_i = torch.zeros(64, 64)

            gaze_heatmap = get_label_map(
            gaze_heatmap_i, [gaze_x * 64, gaze_y * 64], sigma, pdf="Gaussian"
            )
            my_list.append(gaze_heatmap)
            # print(gaze_heatmap.shape)
        
        plotting_data.append(torch.stack(my_list,0))
    # print(len(plotting_data))
            # print(gaze_heatmap.numpy().reshape(step,-1).shape)
    plt.rc('axes', titlesize=10)     # fontsize of the axes title
    plt.rc('axes', labelsize=10)
    fig = plt.figure(figsize=((64*step)/96 , (13*64)/96),dpi=96)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(normal_range, step),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

    for i in range(step):
        for j in range(len(plotting_data)):
            tensordata = plotting_data[j][i]# specific picture
            data_std,data_mean = torch.std_mean(tensordata.reshape(-1), dim=0)
            tensordata_final = tensordata.numpy()
            # grid[i+(j)*step].set_title(i, fontdict=None, loc='center', color = "k")

            # grid[i+(j)*step].set(xlabel='mean\n:{:.2f}\nstd\n:{:.2f}'.format(data_mean,data_std),
            #         ylabel='Linear\nschedular',
            #         )
            grid[i+(j)*step].imshow(tensordata_final,cmap="jet", alpha=0.8)
            grid[i+(j)*step].set_xticklabels([])
            grid[i+(j)*step].set_yticklabels([])
        grid[i+(j)*step].set(xlabel='step:\n{0}'.format(t[i])
                     )
    grid[0].set(ylabel='norm\n{:.2f}'.format(myvalue[0]))
    grid[32].set(ylabel='norm\n{:.2f}'.format(myvalue[1]))
    grid[64].set(ylabel='norm\n{:.2f}'.format(myvalue[2]))
    grid[96].set(ylabel='norm\n{:.2f}'.format(myvalue[3]))
    grid[128].set(ylabel='norm\n{:.2f}'.format(myvalue[4]))
    grid[160].set(ylabel='norm\n{:.2f}'.format(myvalue[5]))
    grid[192].set(ylabel='norm\n{:.2f}'.format(myvalue[6]))
    grid[224].set(ylabel='norm\n{:.2f}'.format(myvalue[7]))
    grid[256].set(ylabel='norm\n{:.2f}'.format(myvalue[8]))
    grid[288].set(ylabel='norm\n{:.2f}'.format(myvalue[9]))
    grid[320].set(ylabel='norm\n{:.2f}'.format(myvalue[10]))

    if(schedular_name=='linear'):
        fig.suptitle('Linear scheudular')
    elif(schedular_name=='cosine'): 
        fig.suptitle('Cosine scheudular')
    elif(schedular_name=='sigmoid'): 
        fig.suptitle('Sigmoid scheudular')
    # Set the axes labels font size
    plt.show()
if __name__ == "__main__":
    # Make runs repeatable as much as possible
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    timesteps = 1000
    betas = linear_beta_schedule(timesteps)
    plottings('linear',32)