import multiprocessing
import os
import random
from datetime import datetime
# from einops import reduce
import functools
from operator import is_not
from functools import partial
import numpy as np
import torch
import torch.nn as nn
# from dotenv import load_dotenv
from joblib import Parallel, delayed
from skimage.transform import resize
from timm.utils import AverageMeter
from torchvision.transforms import transforms
import torch.nn.functional as F
# from torchvision import transforms as utils
from torchvision.utils import save_image
# visualization
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from torchvision.transforms import transforms
from skimage.transform import resize
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.patches as patches
from PIL import Image
# from einops import rearrange
# import math 
import wandb
import io
from omegaconf import OmegaConf

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    print("AMP is not available")

from config import get_config
from gaze.datasets import get_dataset
from diffusion import get_model, load_pretrained
from optimizer import get_optimizer
from schedular import LinearWarmupCosineAnnealingLR,ConstantLRWithWarmup
from diffusion.modules.new_ADM_final.script_util import create_gaussian_diffusion
from diffusion.modules.new_ADM_final.resample import create_named_schedule_sampler
from diffusion.modules.new_ADM_final.nn import update_ema

import copy
from utils import (
    get_angular_error,
    get_auc,
    get_heatmap_peak_coords,
    get_l2_dist,
    get_memory_format,
    get_multi_hot_map,
    get_ap
)

def master_params_to_state_dict(model,master_params):
    state_dict = model.state_dict()
    for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
    return state_dict

def main(config,config_1):
    # Create output dir if training
    if not config_1.Dataset.eval_weights:
        os.makedirs(config_1.Dataset.output_dir, exist_ok=True)
    print("Loading dataset")
    source_loader, target_test_loader = get_dataset(config_1) # you get the two data sets 
    source_iter = iter(target_test_loader)
    data_source = next(source_iter)
    (
                s_rgb,
                s_depth,
                s_heads,# the face
                s_masks,
                s_gaze_heatmaps,
                _,
                gaze_points,
                s_gaze_inside,# boolean yes or no
                _,
                _,coordinates_train
    ) = data_source 
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    id= 14
    mask = 1-s_masks[id]
    fig = plt.figure(figsize=(mask.shape[2] / 96, mask.shape[1] / 96), dpi=96)
    ax = fig.add_subplot(111)

    ax.axis("off")
    print(mask.shape)
    ax.imshow((mask.permute(1, 2, 0).cpu().numpy()), cmap='gray')
    plt.savefig('mask.png', dpi=96, bbox_inches='tight')
    ####################################################
    head_image = s_heads[id]
    fig = plt.figure(figsize=(head_image.shape[2] / 96, head_image.shape[1] / 96), dpi=96)
    ax = fig.add_subplot(111)

    ax.axis("off")
    ax.imshow((invTrans(head_image).permute(1, 2, 0).cpu().numpy()))
    plt.savefig('head.png', dpi=96, bbox_inches='tight')
    ###############################################
    image=s_rgb[id]
    fig = plt.figure(figsize=(image.shape[2] / 96, image.shape[1] / 96), dpi=96)
    ax = fig.add_subplot(111)

    ax.axis("off")
    ax.imshow((invTrans(image).permute(1, 2, 0).cpu().numpy()))
    plt.savefig('image.png', dpi=96, bbox_inches='tight')
    ########################################################
    depth=s_depth[id]
    fig = plt.figure(figsize=(depth.shape[2] / 96, depth.shape[1] / 96), dpi=96)
    ax = fig.add_subplot(111)
    ax.imshow((depth.permute(1, 2, 0).cpu().numpy()), cmap='magma')    
    ax.axis("off")
    plt.savefig('depth.png', dpi=96, bbox_inches='tight')
    plt.show()

















if __name__ == "__main__":
    # Make runs repeatable as much as possible
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    # load_dotenv()# I donot get very well what is happening in here.It shouldnot be used to be honest but va bene hahahahahaha
    '''
    The load_dotenv() function is used in Python to load environment variables from a .env file into the operating system's environment variables.
    Once you have installed the package, you can use the load_dotenv() function to load the environment variables from the .env file located in the
    current working directory. The .env file should contain environment variable names and their values, separated by an equal sign (=).
    For example, if you have a .env file with the following contents:
    You can load these variables into your Python script by calling load_dotenv() at the beginning of your script:


    I didnot find anything like that in the github 
    '''
    x1,x2,x3=get_config()
    merged_config = OmegaConf.merge(x2, x3)
    main(x1,merged_config)