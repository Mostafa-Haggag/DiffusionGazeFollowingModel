
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
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    print("AMP is not available")

# from config import get_config
from gaze.datasets import get_dataset,get_dataset_mhug
from diffusion import get_model, load_pretrained
from optimizer import get_optimizer
from schedular import LinearWarmupCosineAnnealingLR
from diffusion.modules.new_ADM_final.script_util import create_gaussian_diffusion
from diffusion.modules.new_ADM_final.resample import create_named_schedule_sampler
from diffusion.modules.new_ADM_final.nn import update_ema
import pandas as pd 
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
## begin config
import argparse

from omegaconf import OmegaConf


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_diffusion", default="diffusion.yaml")
    parser.add_argument("--yaml_gaze", default="gaze.yaml")
    parser.add_argument("--tag", default="default", help="Description of this run")
    parser.add_argument("--check_img", default="", help="Description of this run")
    parser.add_argument("--threshold", type=float, default=0., help="Resnet face inplanes")

    args = parser.parse_args()
    config_2 = OmegaConf.load(args.yaml_gaze)
    config_1 = OmegaConf.load(args.yaml_diffusion)
    # Update output dir
    args.model_id = f"spatial_depth_late_fusion_{config_1.Dataset.source_dataset}"
    args.output_dir = os.path.join(config_1.Dataset.output_dir, args.model_id, args.tag)
    config_1.Dataset.output_dir =os.path.join(config_1.Dataset.output_dir, args.model_id, args.tag)
    # Reverse resume flag to ease my life
    args.resume = not config_1.Dataset.no_resume and config_1.Dataset.eval_weights is None

    # Reverse wandb flag
    args.wandb = not config_1.Dataset.no_wandb

    # Reverse save flag
    args.save = not config_1.Dataset.no_save

    # Check if AMP is set and is available. If not, remove amp flag
    if config_1.Dataset.amp and amp is None:
        config_1.Dataset.amp = None

    # Print configuration
    print(vars(args))
    return args,config_1,config_2

## end config 
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
    # if not config_1.experiment_parameter.Debugging_maps:
    #     matplotlib.use('Agg')
    print("Loading dataset")
    if config_1.Dataset.source_dataset !="mhug":
        source_loader, target_test_loader = get_dataset(config_1) # you get the two data sets 
    else:
        target_test_loader = get_dataset_mhug(config_1)

    device = torch.device(config_1.Dataset.device) # the device that you will be working with . 
    print(f"Running on {device}")
    # Load model
    print("Loading model")
    # YOU  got the model 
    model = get_model(config_1, device=device).to(device,memory_format=get_memory_format(config_1))
    # this the spatiiala transform number 2 contianing everything that I need in here right now.  

    print("The diffusion ")
    diffusion = create_gaussian_diffusion(
                steps=config_1.Diffusion.train_time_steps,
                sample_steps=config_1.Diffusion.sample_time_steps,
                learn_sigma=config_1.Diffusion.adm_learn_sigma,
                noise_schedule='linear',
                use_kl=config_1.Diffusion.adm_use_kl,
                predict_xstart=config_1.Diffusion.adm_predict_xstart,
                rescale_timesteps=config_1.Diffusion.adm_rescale_timesteps,
                rescale_learned_sigmas=config_1.Diffusion.adm_rescale_learned_sigmas,
                timestep_respacing="",
                normalization_value=config_1.experiment_parameter.diff_normalization,
                normalizaiton_std_flag=config_1.experiment_parameter.norm_std_flag,
                noise_changer=config_1.experiment_parameter.noise_changer,
                mse_loss_weight_type=config_1.losses_parameters.mse_loss_weight_type,
                predict_v=config_1.Diffusion.adm_predict_v,
                enforce_snr=config_1.experiment_parameter.fix_snr,
                )
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)# the weights of everything
    is_ddim_sampling = config_1.Diffusion.sample_time_steps < config_1.Diffusion.train_time_steps
    sample_fn = (
                    diffusion.p_sample_loop if not is_ddim_sampling else diffusion.ddim_sample_loop
                    )
    ema_rate = config_1.Dataset.ema_rate
    if config_1.Dataset.eval_weights:
        # evaluation work 
        # Need to be editted
        print("Preparing evaluation")
        pretrained_dict_org = torch.load(config_1.Dataset.eval_weights, map_location=device)
        pretrained_dict = pretrained_dict_org.get("model_state_dict") or pretrained_dict_org.get("model")
        # you have dictionary with everything in the model
        run_id = pretrained_dict_org.get("run_id")
        epoch  = pretrained_dict_org.get("epoch") 

        for index, element in enumerate(list(pretrained_dict_org.get("ema_params").keys())):
            if element.startswith("model."):
                # print(element)
                break
        new_dict = {**{key: pretrained_dict[key] for key in list(pretrained_dict.keys())[:index]},
            **{key: pretrained_dict_org.get("ema_params")[key] for key in list(pretrained_dict_org.get("ema_params").keys())[index:]}}
        # ema_params = [new_dict[name] for name, _ in model.named_parameters()] # not needed
        if config_1.experiment_parameter.ema_on:
            model = load_pretrained(model,new_dict)
            # model = load_pretrained(model,pretrained_dict_org.get("ema_params"))

        else:
            model = load_pretrained(model, pretrained_dict)
        # Get optimizer
        optimizer = get_optimizer(model
                                  , lr=config_1.Dataset.lr
                                  , weight_decay=config_1.Dataset.weight_decay
                                  )
        # incase there is an ema checkpoint for the loaded weights
        optimizer.zero_grad()
        if config_1.Dataset.amp:
                model, optimizer = amp.initialize(model, optimizer, opt_level=config_1.Dataset.amp)
        ## TODO: check the evaluate funciton in here !!!!!!!
        auc, min_dist, avg_dist, min_ang_err, avg_ang_err,avg_ao,wandb_gaze_heatmap_images = \
                                        evaluate(config_1, 
                                                model,
                                                epoch,
                                                device,
                                                target_test_loader,
                                                sample_fn,
                                                config.check_img,
                                                config.threshold
                                                )
# showing pictures 
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img
def validate_images(image,gazemap_gn,gazemap_pred,vector,epoch,sorted_list=None,Title_figure_3="Pred",new_path=None):
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
        wandb_gaze_heatmap_images = []
        for idx in range(len(image)):
            # plot the image
            fig = plt.figure(figsize=((image.shape[3]/10) , image.shape[2]/10 ),dpi=10)
            ax = plt.gca()
            ax.add_patch(
                            patches.Rectangle(
                                (vector[idx][0], vector[idx][1]),
                                vector[idx][2] - vector[idx][0],
                                vector[idx][3] - vector[idx][1],
                                linewidth=5,
                                edgecolor=(1, 0, 0),
                                facecolor="none",
                            )
                        )

            # ### Ploting the predicted
            ax.imshow((invTrans(image[idx]).permute(1, 2, 0).cpu().numpy()),cmap='viridis',vmin=0, vmax=1)    
            ax.axis("off")
            gaze_heatmap_predicted = gazemap_pred[idx].cpu()
            gaze_heatmap_predicted = resize(
                                gaze_heatmap_predicted,  # Add channel dim
                                (image[idx].shape[1] ,image[idx].shape[2] ),  # [h, w]
                            )
            img = ax.imshow(
                            gaze_heatmap_predicted, cmap="viridis", alpha=0.3,vmin=0, vmax=1)
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="3%", pad=0.05)
            # plt.colorbar(img, cax=cax)
            # if sorted_list != None:
            #     data_to_read = "_{:.4f}".format(sorted_list[idx])
            #     ax.set_title(Title_figure_3+data_to_read,fontsize=12)
            # else:
            #     ax.set_title(Title_figure_3,fontsize=12)
            # fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            fig.tight_layout()

            final_image = fig2img(fig)

            # dpi = 200
            # plt.savefig(new_path[idx], dpi=dpi, bbox_inches='tight')

            # plt.show()
            plt.close(fig)
            plt.close("all")
            final_image.save(new_path[idx][:-3]+'png')
            wandb_gaze_heatmap_images.append(wandb.Image(final_image))

        return wandb_gaze_heatmap_images

## evaluate metrics
def evaluate(config, model, epoch,device, loader, sample_fn,check_img,threshold):
    model.eval()

    output_size = config.Dataset.output_size
    print_every = config.Dataset.print_every

    auc_meter = AverageMeter()
    min_dist_meter = AverageMeter()
    avg_dist_meter = AverageMeter()
    min_ang_error_meter = AverageMeter()
    avg_ang_error_meter = AverageMeter()
    ao_meter =  AverageMeter()


    data_list = []
    with torch.no_grad():# you have more than loader
        for batch, data in enumerate(loader):
            if config.Gaze.depth_flag:    
                (
                    images,
                    depth,
                    faces,
                    masks,
                    gazer_mask,
                    eye_coords,
                    gaze_coords,
                    gaze_inout,
                    img_size,
                    path,coordinates_test
                ) = data
                depth = depth.to(device, non_blocking=True, memory_format=get_memory_format(config))
            else:
                (
                    images,
                    faces,
                    masks,
                    gazer_mask,
                    eye_coords,
                    gaze_coords,
                    gaze_inout,
                    img_size,
                    path,coordinates_test
                ) = data
            images_copy= images
            images = images.to(device, non_blocking=True, memory_format=get_memory_format(config))
            gazer_mask = gazer_mask.to(device, non_blocking=True, memory_format=get_memory_format(config))
            faces = faces.to(device, non_blocking=True, memory_format=get_memory_format(config))
            masks = masks.to(device, non_blocking=True, memory_format=get_memory_format(config))
            gaze_inout = gaze_inout.to(device, non_blocking=True).float()
            if check_img != '':
                if check_img not in path:
                    assert (batch + 1) != len(loader), "ERROR you have entered a wrong directory so it cannot be found"
                    continue


            if config.Gaze.depth_flag:    
                micro_cond = {'images':images,
                      'face':faces,
                      'masks':masks,
                      'depth': depth
                      }
            else:
                micro_cond = {'images':images,
                      'face':faces,
                      'masks':masks
                      }
            gaze_heatmap_pred,inout = sample_fn(
            model,
            (images_copy.shape[0], 1, 64, 64),
            model_kwargs=micro_cond,
            progress=True
            ,clip_denoised=False,

            )
            gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1).cpu()#32,64,64
            n_jobs = max(1, min(multiprocessing.cpu_count(), 12, config.Dataset.batch_size))# njobs = 8
            metrics = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_one_item)(
                    gaze_heatmap_pred[b_i], eye_coords[b_i], gaze_coords[b_i], img_size[b_i], output_size
                )
                for b_i in range(len(gaze_coords))
            )
            ao = get_ap(gaze_inout.cpu().numpy(),inout.cpu().numpy())
            ao_meter.update(ao)

            # metrics = list(filter(partial(is_not, None),metrics ))
            auc_list =[]
            auc_list_extractor =[]
            new_path = []
            # for metric in metrics:
            for index, metric in enumerate(metrics):

                if metric is None:
                    continue
                ao_index = get_ap(gaze_inout.cpu().numpy()[index],inout.cpu().numpy()[index])
                auc_score, min_dist, avg_dist, min_ang_err, avg_ang_err,auc_precision_recall = metric
                new_path.append("../normal_diffusion/"+str(batch)+"_"+str(index)+"_"+path[index].split("/")[-1])
                auc_list.append(auc_score)
                auc_list_extractor.append(index)

                data_list.append([str(batch)+"_"+str(index)+"_"+path[index],auc_score.item(), min_dist.item(), avg_dist.item(), min_ang_err.item(), avg_ang_err.item(),ao_index])
                auc_meter.update(auc_score)
                min_dist_meter.update(min_dist)
                min_ang_error_meter.update(min_ang_err)
                avg_dist_meter.update(avg_dist)
                avg_ang_error_meter.update(avg_ang_err)
            sliced_list = [auc_list[i] for i in auc_list_extractor]
            wandb_gaze_heatmap_images= validate_images(
                                                    images_copy[auc_list_extractor,:],
                                                    gazer_mask[auc_list_extractor,:],
                                                    gaze_heatmap_pred[auc_list_extractor,:],
                                                    coordinates_test[auc_list_extractor,:],
                                                    epoch,sliced_list,new_path=new_path
                                                    )
            if check_img != '':
                        break

            if (batch + 1) % print_every == 0 or (batch + 1) == len(loader):
                print(
                    f"Evaluation - BATCH {(batch + 1):04d}/{len(loader)} "
                    f"\t AUC {auc_meter.avg:.3f}"
                    f"\t AVG. DIST. {avg_dist_meter.avg:.3f}"
                    f"\t MIN. DIST. {min_dist_meter.avg:.3f}"
                    f"\t AVG. ANG. ERR. {avg_ang_error_meter.avg:.3f}"
                    f"\t MIN. ANG. ERR. {min_ang_error_meter.avg:.3f}"
                    f"\t MIN. AO. {ao_meter.avg:.3f}"
                )
    columns = [
    "Frame",
    "auc",
    "min_dist",
    "avg_dist",
    "min_ang_err",
    "avg_ang_err",
    "AO_value"
    ]
    data_list.append(['evaluation final',auc_meter.avg,min_dist_meter.avg.item(),avg_dist_meter.avg.item(),min_ang_error_meter.avg.item(),avg_ang_error_meter.avg,ao_meter.avg])
    df = pd.DataFrame(data_list, columns=columns)
    filename = "evaluation_data.csv"
    df.to_csv(filename, header=True, index=False)
    return (
        auc_meter.avg,
        min_dist_meter.avg,
        avg_dist_meter.avg,
        min_ang_error_meter.avg,
        avg_ang_error_meter.avg,
        ao_meter.avg,
        wandb_gaze_heatmap_images
    )

def evaluate_one_item(
    gaze_heatmap_pred,
    eye_coords,
    gaze_coords,
    img_size,
    output_size,
):

    # Remove padding and recover valid ground truth points
    valid_gaze = gaze_coords[gaze_coords != -1].view(-1, 2)
    valid_eyes = eye_coords[eye_coords != -1].view(-1, 2)

    # Skip items that do not have valid gaze coords
    if len(valid_gaze) == 0:
        return

    multi_hot = get_multi_hot_map(valid_gaze, img_size) # ground truth 
    scaled_heatmap = resize(gaze_heatmap_pred, (img_size[1], img_size[0]))# resize it 
    #  np.reshape(onehot_im, onehot_im.size)
    # print(multi_hot.shape)
    # print(np.reshape(multi_hot, multi_hot.size).shape)
    precision, recall, thresholds = precision_recall_curve(np.reshape(multi_hot, multi_hot.size), np.reshape(scaled_heatmap, scaled_heatmap.size))
    # auc_precision_recall = auc(recall, precision)
    auc_precision_recall = get_ap(np.reshape(multi_hot, multi_hot.size), np.reshape(scaled_heatmap, scaled_heatmap.size))

    auc_score = get_auc(scaled_heatmap, multi_hot)

    pred_x, pred_y = get_heatmap_peak_coords(gaze_heatmap_pred)#predicted points 
    norm_p = torch.tensor([pred_x / float(output_size), pred_y / float(output_size)])#normalzie predictied points 
    all_distances = []
    all_angular_errors = []
    for index, gt_gaze in enumerate(valid_gaze):
        all_distances.append(get_l2_dist(gt_gaze, norm_p))
        all_angular_errors.append(get_angular_error(gt_gaze - valid_eyes[index], norm_p - valid_eyes[index]))

    mean_gt_gaze = torch.mean(valid_gaze, 0)
    avg_distance = get_l2_dist(mean_gt_gaze, norm_p)
    return auc_score, min(all_distances), avg_distance, min(all_angular_errors), np.mean(all_angular_errors),auc_precision_recall*100

if __name__ == "__main__":
    # Make runs repeatable as much as possible
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    x1,x2,x3=get_config()
    merged_config = OmegaConf.merge(x2, x3)
    main(x1,merged_config)