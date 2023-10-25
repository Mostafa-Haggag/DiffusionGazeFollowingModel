import os
import multiprocessing
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
import pandas as pd

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
    if not config_1.experiment_parameter.Debugging_maps:
        matplotlib.use('Agg')
    print("Loading dataset")
    source_loader, target_test_loader = get_dataset(config_1) # you get the two data sets 
    device = torch.device(config_1.Dataset.device) # the device that you will be working with . 
    print(f"Running on {device}")
    # Load model
    print("Loading model")
    # YOU  got the model 
    model = get_model(config_1, device=device).to(device,memory_format=get_memory_format(config_1))
    # this the spatiiala transform number 2 contianing everything that I need in here right now.  
    # schedule_sampler = create_named_schedule_sampler('uniform', diffusion)# the weights of everything
    steps = [5,25,75,100,250,500,750]
    results_df = pd.DataFrame(columns=['Step', 'AUC', 'Min_Distance', 'Avg_Distance', 'Min_Angular_Error', 'Avg_Angular_Error'])

    for step in steps:
        config_1.Diffusion.sample_time_steps = step
        print("The number of steps are :",config_1.Diffusion.sample_time_steps)
        print("The diffusion ")
        diffusion = create_gaussian_diffusion(
                steps=config_1.Diffusion.train_time_steps,
                sample_steps=config_1.Diffusion.sample_time_steps,
                learn_sigma=config_1.Diffusion.adm_learn_sigma,
                noise_schedule=config_1.Diffusion.schedule_type,
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
                factor_multiplier=config_1.experiment_parameter.factor_multiplier,
                )
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
            print(f"We are evaluating run_id: {run_id}") 
            print(f"We are evaluating epoch: {epoch}")
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
            auc, min_dist, avg_dist, min_ang_err, avg_ang_err,avg_ao,avg_heatmpap_ao,wandb_gaze_heatmap_images= \
                                            evaluate(config_1, 
                                                    model,
                                                    epoch,
                                                    device,
                                                    target_test_loader,
                                                    sample_fn
                                                    ) 
                # Append the results to the DataFrame
            results_df = results_df.append({
                'Step': step,
                'AUC': auc,
                'Min_Distance': min_dist.item(),
                'Avg_Distance': avg_dist.item(),
                'Min_Angular_Error': min_ang_err.item(),
                'Avg_Angular_Error': avg_ang_err
            }, ignore_index=True)
            # Print summary
            print("\nEval summary")
            print(f"AUC: {auc:.3f}")
            print(f"Minimum distance: {min_dist:.3f}")
            print(f"Average distance: {avg_dist:.3f}")
            print(f"Minimum angular error: {min_ang_err:.3f}")
            print(f"Average angular error: {avg_ang_err:.3f}")
    print("\nEval summary")
    print(results_df)
    results_df.to_csv('evaluation_results.csv', index=False)

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img
def validate_images(image,gazemap_gn,gazemap_pred,vector,epoch,debug=False,sorted_list=None,Title_figure_3="Pred"):
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
        wandb_gaze_heatmap_images = []
        for idx in range(len(image)):
            # plot the image
            fig, (
                    ax_bbox,
                    ax_heatmap_gn,
                    ax_heatmap_pred,
                ) = plt.subplots(
                    nrows=1,
                    ncols=3,
                    figsize=((image.shape[3]*3)/96 , image.shape[2]/96 ),
                    dpi=96,
                )
            ax_bbox.axis("off")
            ax_bbox.imshow((invTrans(image[idx]).permute(1, 2, 0).cpu().numpy()),vmin=0, vmax=1)    
            ax_bbox.add_patch(
                            patches.Rectangle(
                                (vector[idx][0], vector[idx][1]),
                                vector[idx][2] - vector[idx][0],
                                vector[idx][3] - vector[idx][1],
                                linewidth=5,
                                edgecolor=(1, 0, 0),
                                facecolor="none",
                            )
                        )
            ax_bbox.set_title("BBox",fontsize=12)
            
            ## heat map ground truth
            ax_heatmap_gn.imshow((invTrans(image[idx]).permute(1, 2, 0).cpu().numpy()),vmin=0, vmax=1)    
            ax_heatmap_gn.axis("off")
            gaze_heatmap = gazemap_gn[idx].cpu()
            
            gaze_heatmap = resize(
                                gaze_heatmap,  # Add channel dim
                                (image[idx].shape[1] ,image[idx].shape[2] ),  # [h, w]
                            )
            
            im = ax_heatmap_gn.imshow(
                            gaze_heatmap, cmap="jet", alpha=0.3, vmin=0,vmax=1
                        )
            divider = make_axes_locatable(ax_heatmap_gn)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax_heatmap_gn.set_title("GN",fontsize=12)

            ### Ploting the predicted
            ax_heatmap_pred.imshow((invTrans(image[idx]).permute(1, 2, 0).cpu().numpy()),vmin=0, vmax=1)    
            ax_heatmap_pred.axis("off")
            gaze_heatmap_predicted = gazemap_pred[idx].cpu()
            gaze_heatmap_predicted = resize(
                                gaze_heatmap_predicted,  # Add channel dim
                                (image[idx].shape[1] ,image[idx].shape[2] ),  # [h, w]
                            )
            img = ax_heatmap_pred.imshow(
                            gaze_heatmap_predicted, cmap="jet", alpha=0.3,vmin=0)
            divider = make_axes_locatable(ax_heatmap_pred)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            plt.colorbar(img, cax=cax)
            if sorted_list != None:
                data_to_read = "_{:.2f}".format(sorted_list[idx][0])
                ax_heatmap_pred.set_title(Title_figure_3+data_to_read,fontsize=12)
            else:
                ax_heatmap_pred.set_title(Title_figure_3,fontsize=12)

            final_image = fig2img(fig)
            if debug:
                plt.show()
            plt.close(fig)
            plt.close("all")
            wandb_gaze_heatmap_images.append(wandb.Image(final_image))

        return wandb_gaze_heatmap_images

def evaluate(config, model, epoch,device, loader, sample_fn):
    model.eval()

    output_size = config.Dataset.output_size
    print_every = config.Dataset.print_every

    auc_meter = AverageMeter()
    min_dist_meter = AverageMeter()
    avg_dist_meter = AverageMeter()
    min_ang_error_meter = AverageMeter()
    avg_ang_error_meter = AverageMeter()
    ao_heat_map_meter =  AverageMeter()
    gaze_to_save = 4
    previous_sorted_list = []
    #### meow meoww
    new_image = [] 
    new_gazer_mask = []
    new_gaze_heatmap_pred = []
    new_coordinate_test = []
    gaze_inside_all = []
    gaze_inside_pred_all = []
    #
    # original_indices = []
    # loader has size of 150
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
                    _,coordinates_test
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
                    _,coordinates_test
                ) = data
            images_copy= images
            # images = torch.cat((images, masks), dim=1)
            images = images.to(device, non_blocking=True, memory_format=get_memory_format(config))
            gazer_mask = gazer_mask.to(device, non_blocking=True, memory_format=get_memory_format(config))
            faces = faces.to(device, non_blocking=True, memory_format=get_memory_format(config))
            masks = masks.to(device, non_blocking=True, memory_format=get_memory_format(config))
            gaze_inout = gaze_inout.to(device, non_blocking=True).float()
            # print(gaze_inout.shape,images.shape)
            # print(gaze_inout[0].shape)
            # print(gaze_coords[0].shape)
            # print(coordinates_test.shape)
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
            clip_denoised=True,
            progress=True,
            sigma_hm=config.experiment_parameter.random_sigma,
            eta=0
            )
            # print(gaze_heatmap_pred.shape)
            # gaze_heatmap_pred = model.sample(images,masks,faces,gazer_mask,config.eval_from_picture,config.time_noise,images.shape[0])
            # sample = torch.cat((gaze_heatmap_pred[:gaze_to_save],gazer_mask.unsqueeze(1)[:gaze_to_save]), -2)
            # you sample for everything which is a problem.

            # slicing to remove the uneeded stuff from here. 
            gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1).cpu()#32,64,64
            gaze_inside_all.extend(gaze_inout.cpu().tolist())
            gaze_inside_pred_all.extend(inout.cpu().tolist())
            # Sets the number of jobs according to batch size and cpu counts. In any case, no less than 1 and more than
            # 8 jobs are allocated.
            n_jobs = max(1, min(multiprocessing.cpu_count(), 12, config.Dataset.batch_size))# njobs = 8
            metrics = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_one_item)(
                    gaze_heatmap_pred[b_i], eye_coords[b_i], gaze_coords[b_i], img_size[b_i], output_size
                )
                for b_i in range(len(gaze_coords))
            )
            # print(gaze_inout.cpu().numpy())
            # print(inout.cpu().numpy())
            # ao = get_ap(gaze_inout.cpu().numpy(),inout.cpu().numpy())
            # print(ao,ao.shape)
            # print(ao.shape)
            # ao_meter.update(ao)

            metrics = list(filter(partial(is_not, None),metrics ))
            # len gaze coordinates = 32 which mean that i am looping on the batch size
            # eye coordinates is 32 by 20 by2
            # gaze coordinates is 32 by 20 by 2 
            ## image size is 32 by 2 
            ## output size 64
            sorted_tuples = sorted(enumerate(metrics), key=lambda x: x[1][0],reverse=True)# I sort the metrics
            previous_sorted_list.extend([x[1] for x in sorted_tuples[:gaze_to_save]]) # extract the values auc of the worst 4
            original_indices=[x[0] for x in sorted_tuples[:gaze_to_save]]# index auc of worst 4

            #########################################################################
            # adding the indices

            # new_image.extend([x[0] for x in images[original_indices]]) 
            # new_gazer_mask.extend([x[0] for x in gazer_mask[original_indices]])
            # new_gaze_heatmap_pred.extend([x[0] for x in gaze_heatmap_pred[original_indices]])
            # new_coordinate_test.extend([x[0] for x in coordinates_test[original_indices]])
            #########################################################################
            #[3,2,7,5]
            # [8,9,10,4]
            if batch == 0:
                new_image = images[original_indices].clone()
                new_gazer_mask = gazer_mask[original_indices].clone()
                new_gaze_heatmap_pred = gaze_heatmap_pred[original_indices].clone()
                new_coordinate_test = coordinates_test[original_indices].clone()

            else:
                sorted_tuples_2 = sorted(enumerate(previous_sorted_list), key=lambda x: x[1][0],reverse=True)
                new_sorted_list = [x[1] for x in sorted_tuples_2[:gaze_to_save]]
                final_indices = [x[0] for x in sorted_tuples_2[:gaze_to_save]] # index of the current batch so we can choose witch to remove
                previous_sorted_list = new_sorted_list

                ################################
                new_image = torch.concat([new_image,images[original_indices].clone()],0)
                new_gazer_mask = torch.concat([new_gazer_mask,gazer_mask[original_indices].clone()],0)
                new_gaze_heatmap_pred = torch.concat([new_gaze_heatmap_pred,gaze_heatmap_pred[original_indices].clone()],0)
                new_coordinate_test = torch.concat([new_coordinate_test,coordinates_test[original_indices].clone()],0)
                ################################
                new_image = new_image[final_indices]
                new_gazer_mask = new_gazer_mask[final_indices]
                new_gaze_heatmap_pred = new_gaze_heatmap_pred[final_indices]
                new_coordinate_test = new_coordinate_test[final_indices]

            # original_indices = original_indices[:gaze_to_save]
            for metric in metrics:
                if metric is None:
                    continue

                auc_score, min_dist, avg_dist, min_ang_err, avg_ang_err,ao_heatmap = metric
                ao_heat_map_meter.update(ao_heatmap)
                auc_meter.update(auc_score)
                min_dist_meter.update(min_dist)
                min_ang_error_meter.update(min_ang_err)
                avg_dist_meter.update(avg_dist)
                avg_ang_error_meter.update(avg_ang_err)
            if (batch + 1) % print_every == 0 or (batch + 1) == len(loader):
                print(
                    f"Evaluation - BATCH {(batch + 1):04d}/{len(loader)} "
                    f"\t AUC {auc_meter.avg:.3f}"
                    f"\t AVG. DIST. {avg_dist_meter.avg:.3f}"
                    f"\t MIN. DIST. {min_dist_meter.avg:.3f}"
                    f"\t AVG. ANG. ERR. {avg_ang_error_meter.avg:.3f}"
                    f"\t MIN. ANG. ERR. {min_ang_error_meter.avg:.3f}"
                    f"\t MIN. AO. {get_ap(gaze_inside_all, gaze_inside_pred_all):.3f}"
                    f"\t MIN. HM AO. {ao_heat_map_meter.avg:.3f}"

                )
                if (batch + 1) == len(loader) or config.experiment_parameter.Debugging_maps:
                    wandb_gaze_heatmap_images= validate_images(
                                                    new_image,
                                                    new_gazer_mask,
                                                    new_gaze_heatmap_pred,
                                                    new_coordinate_test,
                                                    epoch,config.experiment_parameter.Debugging_maps,sorted_list=previous_sorted_list
                                                    )
    gaze_inside_ap = get_ap(gaze_inside_all, gaze_inside_pred_all)

    return (
        auc_meter.avg,
        min_dist_meter.avg,
        avg_dist_meter.avg,
        min_ang_error_meter.avg,
        avg_ang_error_meter.avg,
        gaze_inside_ap,
        ao_heat_map_meter.avg,
        wandb_gaze_heatmap_images,
    )


def evaluate_one_item(
    gaze_heatmap_pred,
    eye_coords,
    gaze_coords,
    img_size,
    output_size,
):
    #gaze_heatmap_pred 64by 64
    # eye_coords 20 by2 
    # gaze coordinate 20 by 2 
    # image__size 2
    # Remove padding and recover valid ground truth points
    valid_gaze = gaze_coords[gaze_coords != -1].view(-1, 2)
    valid_eyes = eye_coords[eye_coords != -1].view(-1, 2)

    # Skip items that do not have valid gaze coords
    if len(valid_gaze) == 0:
        return
    org_hot = get_multi_hot_map(valid_gaze, torch.full((2,), 64,dtype=torch.int32)) # ground truth 
    unscaled_heatmap = resize(gaze_heatmap_pred, (64, 64))
    auc_precision_recall = get_ap(np.reshape(org_hot, org_hot.size), np.reshape(unscaled_heatmap, unscaled_heatmap.size))
    # AUC: area under curve of ROC
    multi_hot = get_multi_hot_map(valid_gaze, img_size) # ground truth 
    scaled_heatmap = resize(gaze_heatmap_pred, (img_size[1], img_size[0]))# resize it 
    auc_score = get_auc(scaled_heatmap, multi_hot)

    # Min distance: minimum among all possible pairs of <ground truth point, predicted point>
    pred_x, pred_y = get_heatmap_peak_coords(gaze_heatmap_pred)#predicted points 
    norm_p = torch.tensor([pred_x / float(output_size), pred_y / float(output_size)])#normalzie predictied points 
    all_distances = []
    all_angular_errors = []
    for index, gt_gaze in enumerate(valid_gaze):
        all_distances.append(get_l2_dist(gt_gaze, norm_p))
        all_angular_errors.append(get_angular_error(gt_gaze - valid_eyes[index], norm_p - valid_eyes[index]))

    # Average distance: distance between the predicted point and human average point
    mean_gt_gaze = torch.mean(valid_gaze, 0)
    avg_distance = get_l2_dist(mean_gt_gaze, norm_p)
    return auc_score, min(all_distances), avg_distance, min(all_angular_errors), np.mean(all_angular_errors),auc_precision_recall


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
