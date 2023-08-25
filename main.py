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
from gaze.datasets import get_dataset,get_dataset_mhug
from diffusion import get_model, load_pretrained
from optimizer import get_optimizer
from schedular import LinearWarmupCosineAnnealingLR
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

        # ema_params = copy.deepcopy(list(model.parameters()))
        # this line of code doesnot make any sense we could remove it
        # ema_params = [pretrained_dict_org.get("ema_params")[name] for name, _ in model.named_parameters()]
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
                                                sample_fn
                                                )
        if run_id is not None and config.wandb:
            print(f"Resuming wandb run with id {run_id}")
            if config_1.Dataset.source_dataset=="videoattentiontarget":
                wandb.init(id=run_id,
                            resume="must",
                            save_code=True)
                wandb.watch(model,log="gradients", log_freq=1000)
                wandb.log(
                            {
                                "epoch": epoch,
                                "video/auc": auc,
                                "video/min_dist": min_dist,
                                "video/avg_dist": avg_dist,
                                "video/min_ang_err": min_ang_err,
                                "video/avg_ang_err": avg_ang_err,
                                "video/avg_ao": avg_ao,
                                "video/images":wandb_gaze_heatmap_images,
                            }
                )
            if config_1.Dataset.source_dataset=="mhug":
                wandb.init(id=run_id,
                            resume="must",
                            save_code=True)
                wandb.watch(model,log="gradients", log_freq=1000)
                wandb.log(
                            {
                                "epoch": epoch,
                                "mhug/auc": auc,
                                "mhug/min_dist": min_dist,
                                "mhug/avg_dist": avg_dist,
                                "mhug/min_ang_err": min_ang_err,
                                "mhug/avg_ang_err": avg_ang_err,
                                "mhug/avg_ao": avg_ao,
                                "mhug/images":wandb_gaze_heatmap_images,
                            }
                )
            else:
                wandb.init(id=run_id,
                            resume="must",
                            save_code=True)
                wandb.watch(model,log="gradients", log_freq=1000)
                wandb.log(
                            {
                                "epoch": epoch,
                                "val/auc": auc,
                                "val/min_dist": min_dist,
                                "val/avg_dist": avg_dist,
                                "val/min_ang_err": min_ang_err,
                                "val/avg_ang_err": avg_ang_err,
                                "val/avg_ao": avg_ao,
                                "val/images":wandb_gaze_heatmap_images,
                            }
                )
        # Print summary
        print("\nEval summary")
        print(f"AUC: {auc:.3f}")
        print(f"Minimum distance: {min_dist:.3f}")
        print(f"Average distance: {avg_dist:.3f}")
        print(f"Minimum angular error: {min_ang_err:.3f}")
        print(f"Average angular error: {avg_ang_err:.3f}")
    else:
        print("Preparing training")

        # Select best kernel for convolutions
         # this is used to fix everything 
        torch.backends.cudnn.benchmark = True

        # Allows to resume a run from a given epoch
        next_epoch = 0

        # This value is filled by checkpoint if resuming
        # Once assigned, run_id equals the id of a wandb run
        run_id = None

        # Check and resume previous run if existing
        if config.resume:# you are resuming from specific place 
            checkpoints = os.listdir(config_1.Dataset.output_dir)
            checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith("pth")]
            if len(checkpoints) > 0:
                latest_checkpoint = max(
                    [os.path.join(config.output_dir, d) for d in checkpoints],
                    key=os.path.getmtime,
                )
                print(f"Latest checkpoint found: {latest_checkpoint}")
                print(f"Loading weights, optimizer and losses from {latest_checkpoint} run. This may take a while")

                checkpoint = torch.load(latest_checkpoint)
                # The torch.load() function in PyTorch is used to load saved objects from a file, including models and tensors.
                # It is a general-purpose function that can be used to load various types of objects that were previously 
                # saved using torch.save() or similar methods.
                model = load_pretrained(model, checkpoint["model"])
                # # Get optimizer

                optimizer = get_optimizer(model
                                        , lr=config_1.Dataset.lr
                                        , weight_decay=config_1.Dataset.weight_decay
                                    )

                ema_params = [checkpoint.get("ema_params")[name] for name, _ in model.named_parameters()]

                
                optimizer.zero_grad()
                #
                if config_1.experiment_parameter.lr_schedular:
                    scheduler = LinearWarmupCosineAnnealingLR(optimizer,warmup_epochs=int(config_1.Dataset.epochs*0.1), max_epochs=config_1.Dataset.epochs)
                    scheduler.load_state_dict(checkpoint["scheduler"])
                optimizer.load_state_dict(checkpoint["optimizer"])# loading the state of the optimizer
                next_epoch = checkpoint["epoch"] + 1
                mse_loss = checkpoint["mse_loss"]
                run_id = checkpoint["run_id"]
                del checkpoint

        # The next_epoch check makes sure that we start with init_weights even when resume is set to True but no
        # checkpoints are not found and i am starting from 0 and I must set the intial weightss
        # the one the 3amadak allak 3alehom 
        if config_1.Dataset.init_weights and next_epoch == 0:
            # you are putting some weights in the begining only
            print("Loading init weights")

            pretrained_dict = torch.load(config_1.Dataset.init_weights, map_location=device)
            pretrained_dict = pretrained_dict.get("model_state_dict") or pretrained_dict.get("model")

            model = load_pretrained(model, pretrained_dict)
            # optimizer = get_optimizer(model.parameters()
            #                             , lr=config.lr
            #                             , weight_decay=config.weight_decay
            #                         )
            optimizer = get_optimizer(model
                                        , lr=config_1.Dataset.lr
                                        , weight_decay=config_1.Dataset.weight_decay
                                    )
            ema_params = copy.deepcopy(list(model.parameters()))
            if config_1.experiment_parameter.lr_schedular:
                    scheduler = LinearWarmupCosineAnnealingLR(optimizer,warmup_epochs=int(config_1.Dataset.epochs*0.1), max_epochs=config_1.Dataset.epochs)
            # # Get optimizer
            optimizer.zero_grad()
            del pretrained_dict
        elif len(checkpoints) == 0:
            # you are putting no weights at all
            # optimizer = get_optimizer(model.parameters()
            #                             , lr=config.lr
            #                             , weight_decay=config.weight_decay
            #                         )
            optimizer = get_optimizer(model
                                        , lr=config.lr
                                        , weight_decay=config.weight_decay
                                    )

            ema_params = copy.deepcopy(list(model.parameters()))
            optimizer.zero_grad()
            if config_1.experiment_parameter.lr_schedular:
                    scheduler = LinearWarmupCosineAnnealingLR(optimizer,warmup_epochs=int(config_1.Dataset.epochs*0.1), max_epochs=config_1.Dataset.epochs)
        # turning it on fucks everything
        # print(len(optimizer.param_groups))

        for param_group in optimizer.param_groups:
            print("Learning rate:", param_group['lr'])
        print(optimizer.param_groups[0]['lr'])
        # Initialize wandb
        # it is realted to weight and bias 
        if run_id is not None and config.wandb:
                print(f"Resuming wandb run with id {run_id}")

                wandb.init(id=run_id,
                           resume="must",
                           save_code=True)
                if config_1.Diffusion.watch_wandb:
                    wandb.watch(model,log="gradients", log_freq=2000)

        elif config.wandb:
                run_id = wandb.util.generate_id()
                print(f"Starting a new wandb run with id {run_id}")
                config_dict = OmegaConf.to_container(config_1, resolve=True)
                config_dict.update(vars(config))
                wandb.init(
                    id=run_id,
                    config=config_dict,
                    tags=["spatial_depth_late_fusion", config_1.Dataset.source_dataset],
                    save_code=True
                )
                wandb.run.name = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}_{config.model_id}_{config.tag}'
                
                if config_1.Diffusion.watch_wandb:
                    wandb.watch(model,log="gradients", log_freq=2000)

        else:
                # We force the run_id to a random string
                # I shouldnot watch the model in here because it mean that  I donot have WANDB turnied on 
                run_id = "0118999881999111725 3"
        # Init AMP if enabled and available
        # we can ignore this part as it is not needed 
        if config_1.Dataset.amp:
            model, optimizer = amp.initialize(model, optimizer, opt_level=config_1.Dataset.amp)
        # turning on the DDIM Sampler from here!!!!!
        # the function  that we will be calling aaccording to the number of smaplign 
        # time steps relative to the train time steps !!!!!
        mse_loss = nn.MSELoss() # not reducing in order to ignore outside cases
        bg_loss =  nn.BCEWithLogitsLoss()

        print(f"Training from epoch {next_epoch + 1} to {config_1.Dataset.epochs}. {len(source_loader)} batches per epoch")
        for ep in range(next_epoch, config_1.Dataset.epochs):
            start = datetime.now()
            model.train()
            train_one_epoch(
                config_1,
                ep,
                model,
                device,
                source_loader,
                optimizer,
                ema_params,
                diffusion,
                schedule_sampler,
                ema_rate,mse_loss,bg_loss,config.wandb
                )
            print(f"Epoch {ep + 1} took {datetime.now() - start}")
            
            if config_1.experiment_parameter.lr_schedular:
                scheduler.step()
                checkpoint = {
                    "run_id": run_id,
                    "epoch": ep,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "mse_loss": F.mse_loss,
                    'scheduler': scheduler.state_dict(),
                    "ema_params":master_params_to_state_dict(model,ema_params) # setting it to the correct values
                }                
            else:
                checkpoint = {
                    "run_id": run_id,
                    "epoch": ep,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "mse_loss": F.mse_loss,
                    "ema_params":master_params_to_state_dict(model,ema_params) # setting it to the correct values
                }

            # Save the model
            # We want to save the checkpoint of the last epoch, so that we can resume training later
            save_path = os.path.join(config_1.Dataset.output_dir, "ckpt_last.pth")

            # Keep previous checkpoint until we are sure that this saving goes through successfully.
            backup_path = os.path.join(config_1.Dataset.output_dir, "ckpt_last.backup.pth")
            if os.path.exists(save_path):
                os.rename(save_path, backup_path)

            # Try to save and load the latest checkpoint. If no exception, delete backup file. Otherwise, stop.
            try:
                torch.save(checkpoint, save_path)
                _ = torch.load(save_path, map_location=torch.device("cpu"))

                print(f"Checkpoint saved at {save_path}")
            except Exception as e:
                print(e)
                print("Unable to save or verify last checkpoint. Restoring previous checkpoint.")

                os.remove(save_path)
                os.rename(backup_path, save_path)

                exit(1)

            # Remove backup file
            if os.path.exists(backup_path):
                os.remove(backup_path)

            if config.save and ((ep + 1) % config_1.Dataset.save_every == 0 or (ep + 1) == config_1.Dataset.epochs):
                save_path = os.path.join(config_1.Dataset.output_dir, f"ckpt_epoch_{ep + 1}.pth")
                torch.save(checkpoint, save_path)
                # I should also save the ema checkpoint this is missing

                print(f"Checkpoint saved at {save_path}")

            if (ep + 1) % config_1.Dataset.evaluate_every == 0 or (ep + 1) == config_1.Dataset.epochs:
                print("Starting evaluation")
                
                auc, min_dist, avg_dist, min_ang_err, avg_ang_err,avg_ao,wandb_gaze_heatmap_images = \
                                        evaluate(config_1, 
                                                model,
                                                ep+1,
                                                device,
                                                target_test_loader,
                                                sample_fn
                                                )

                # target test loader is passed 
                if config.wandb:
                    wandb.log(
                    {
                            "epoch": ep + 1,
                            "val/auc": auc,
                            "val/min_dist": min_dist,
                            "val/avg_dist": avg_dist,
                            "val/min_ang_err": min_ang_err,
                            "val/avg_ang_err": avg_ang_err,
                            "val/avg_ao":avg_ao,
                            "val/images":wandb_gaze_heatmap_images,
                    }
                    )
def train_one_epoch(
    config,
    epoch,
    model,
    device,
    source_loader,
    optimizer,
    ema_params,
    diffusion,
    schedule_sampler,
    ema_rate,
    mse_loss,bg_loss,wandb_flag
):
    gaze_to_save = 5
    print_every = config.Dataset.print_every
    source_iter = iter(source_loader)
    n_iter = len(source_loader)
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    for batch in range(n_iter):# iterate for number of things
        if config.Gaze.depth_flag:
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
            s_depth = s_depth.to(device, non_blocking=True, memory_format=get_memory_format(config))
        else:
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
        # the part of the samplar is ready in here. 
        # sampler is outside 
        t, weights = schedule_sampler.sample(s_rgb.shape[0], device)
        # s_rgb = torch.cat((s_rgb, s_masks), dim=1)
        s_rgb = s_rgb.to(device, non_blocking=True, memory_format=get_memory_format(config))
        s_heads = s_heads.to(device, non_blocking=True, memory_format=get_memory_format(config))
        s_gaze_heatmaps = s_gaze_heatmaps.unsqueeze(1).to(device, non_blocking=True)
        s_gaze_inside = s_gaze_inside.to(device, non_blocking=True).float()
        gaze_points = gaze_points.to(device, non_blocking=True, memory_format=get_memory_format(config))
        s_masks = s_masks.to(device, non_blocking=True, memory_format=get_memory_format(config))
        if config.Gaze.depth_flag:
            micro_cond = {'images':s_rgb,
                      'face':s_heads,
                      'masks':s_masks,
                      'noise_strength':config.experiment_parameter.offset_noise_strength,
                      'depth':s_depth
                      }
        else:
            micro_cond = {'images':s_rgb,
                'face':s_heads,
                'masks':s_masks,
                'noise_strength':config.experiment_parameter.offset_noise_strength,
                }
        # return from the model
        # you didnot normalize here at all 
        compute_losses = functools.partial(
                        diffusion.training_losses,
                        model,
                        s_gaze_heatmaps,
                        t,
                        model_kwargs=micro_cond,
                    )
        losses = compute_losses()
        Xent_loss = bg_loss(losses["inout"].squeeze(), s_gaze_inside.squeeze())

        if config.experiment_parameter.Debugging_maps:
            validate_images(
                                                    
                                                    s_rgb[:gaze_to_save].detach(),
                                                    s_gaze_heatmaps[:gaze_to_save].squeeze(1).detach(),
                                                    losses["output"][:gaze_to_save].detach(),
                                                    coordinates_train[:gaze_to_save].detach(),
                                                    epoch,config.experiment_parameter.Debugging_maps
                                                    )
        s_rec_loss = torch.mul(losses["loss"] * weights, s_gaze_inside.mean(axis=1))
        s_rec_loss = torch.sum(losses["loss"] * weights) / torch.sum(s_gaze_inside.mean(axis=1))
        if config.losses_parameters.other_loss:
            output_loss = mse_loss(losses["location"], torch.flip(gaze_points.squeeze(1), [1]))
            if config.losses_parameters.kl_div_loss:
                the_input = F.log_softmax(s_gaze_heatmaps.squeeze(1).view(s_gaze_heatmaps.shape[0],-1), dim=1)
                the_target =F.softmax(losses["output"].view(s_gaze_heatmaps.shape[0],-1) , dim=1)
                the_kl_loss = kl_loss(the_input,the_target)
                total_loss = s_rec_loss +  10*the_kl_loss
            else:
                total_loss = 100000*s_rec_loss + 10000*output_loss
        else:
            if config.losses_parameters.x_loss:
                total_loss = s_rec_loss + 0.01*Xent_loss
            else:
                total_loss = s_rec_loss

        if config.Dataset.amp:
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()
        # # Gradient clipping 
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        # the idea here is that you are not evaluating the ema and this is the problem 
        # do you understand the idea in here? you could choose to evaluate using what weights
        # if you use ema weights you will be evaluating something else.
        # for rate, params in zip(ema_rate, ema_params):
        update_ema(ema_params, list(model.parameters()), rate=ema_rate)
        optimizer.zero_grad()

        if (batch + 1) % print_every == 0 or (batch + 1) == n_iter:
            log = f"Training - EPOCH {(epoch + 1):02d}/{config.Dataset.epochs:02d} BATCH {(batch + 1):04d}/{n_iter} "
            log += f"\t TASK LOSS (L2) {s_rec_loss:.6f}"
            print(log)

        if wandb_flag:
            if config.losses_parameters.other_loss:
                if config.losses_parameters.kl_div_loss:
                    log = {
                        "epoch": epoch + 1,
                        "train/batch": batch,
                        "train/KL_loss": the_kl_loss.item(),
                        "train/outputloss": s_rec_loss.item(),
                        "train/loss": total_loss.item(),
                        "lr_dm":optimizer.param_groups[0]['lr'],
                        "lr_resnet":optimizer.param_groups[1]['lr'],
                        }
                else:
                        log = {
                        "epoch": epoch + 1,
                        "train/batch": batch,
                        "train/point_loss": output_loss.item(),
                        "train/outputloss": s_rec_loss.item(),
                        "train/loss": total_loss.item(),
                        "lr_dm":optimizer.param_groups[0]['lr'],
                        "lr_resnet":optimizer.param_groups[1]['lr'],
                        }
            else:
                    if config.losses_parameters.x_loss:
                        log = {
                        "epoch": epoch + 1,
                        "train/batch": batch,
                        "train/outputloss": s_rec_loss.item(),
                        "train/Xent_loss": Xent_loss.item(),
                        "train/loss": total_loss.item(),
                        "lr_dm":optimizer.param_groups[0]['lr'],
                        "lr_resnet":optimizer.param_groups[1]['lr'],
                        }
                    else:
                        log = {
                        "epoch": epoch + 1,
                        "train/batch": batch,
                        "train/outputloss": s_rec_loss.item(),
                        "train/loss": total_loss.item(),
                        "lr_dm":optimizer.param_groups[0]['lr'],
                        "lr_resnet":optimizer.param_groups[1]['lr'],
                        }
            wandb.log(log)
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
                            gaze_heatmap_predicted, cmap="jet", alpha=0.3,vmin=0, vmax=1)
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
    ao_meter =  AverageMeter()
    gaze_to_save = 4
    previous_sorted_list = []
    #### meow meoww
    new_image = [] 
    new_gazer_mask = []
    new_gaze_heatmap_pred = []
    new_coordinate_test = []
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
            progress=True
            )
            # print(gaze_heatmap_pred.shape)
            # gaze_heatmap_pred = model.sample(images,masks,faces,gazer_mask,config.eval_from_picture,config.time_noise,images.shape[0])
            # sample = torch.cat((gaze_heatmap_pred[:gaze_to_save],gazer_mask.unsqueeze(1)[:gaze_to_save]), -2)
            # you sample for everything which is a problem.

            # slicing to remove the uneeded stuff from here. 
            gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1).cpu()#32,64,64

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
            ao = get_ap(gaze_inout.cpu().numpy(),inout.cpu().numpy())
            # print(ao,ao.shape)
            # print(ao.shape)
            ao_meter.update(ao)

            metrics = list(filter(partial(is_not, None),metrics ))
            # len gaze coordinates = 32 which mean that i am looping on the batch size
            # eye coordinates is 32 by 20 by2
            # gaze coordinates is 32 by 20 by 2 
            ## image size is 32 by 2 
            ## output size 64
            sorted_tuples = sorted(enumerate(metrics), key=lambda x: x[1][0])# I sort the metrics
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
                sorted_tuples_2 = sorted(enumerate(previous_sorted_list), key=lambda x: x[1][0])
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

                auc_score, min_dist, avg_dist, min_ang_err, avg_ang_err = metric

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
                    f"\t MIN. AO. {ao_meter.avg:.3f}"

                )
                if (batch + 1) == len(loader) or config.experiment_parameter.Debugging_maps:
                    wandb_gaze_heatmap_images= validate_images(
                                                    new_image,
                                                    new_gazer_mask,
                                                    new_gaze_heatmap_pred,
                                                    new_coordinate_test,
                                                    epoch,config.experiment_parameter.Debugging_maps,sorted_list=previous_sorted_list
                                                    )
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
    return auc_score, min(all_distances), avg_distance, min(all_angular_errors), np.mean(all_angular_errors)


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
