import argparse
import os
# from datetime import datetime

#datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def get_config():
    parser = argparse.ArgumentParser()

    # Run metadata
    parser.add_argument("--tag", default="default", help="Description of this run")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])

    # Dataset args
    parser.add_argument("--input_size", type=int, default=224, help="input size")
    parser.add_argument("--output_size", type=int, default=64, help="output size")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--source_dataset_dir",
        type=str,
        default="datasets/gazefollow_extended",
        help="directory where the source dataset is located",
    )
    # datasets/gazefollow_extended
    # datasets/video_dataset
    parser.add_argument(
        "--source_dataset",
        type=str,
        default="gazefollow",
        choices=["gazefollow", "videoattentiontarget", "goo"],
    )
    # gazefollow
    # videoattentiontarget
    parser.add_argument(
        "--target_dataset_dir",
        type=str,
        default="datasets/gazefollow_extended",
        help="directory where the target dataset is located",
    )
    # datasets/gazefollow_extended
    # datasets/video_dataset
    parser.add_argument(
        "--target_dataset",
        type=str,
        default="gazefollow",
        choices=["gazefollow", "videoattentiontarget", "goo"],
    )
    # gazefollow
    # videoattentiontarget
    parser.add_argument("--num_workers", type=int, default=max(12, os.cpu_count()))
    parser.add_argument("--gaze_point_threshold", type=int, default=0)

    # Model args
    parser.add_argument("--init_weights", type=str, help="initial weights")
    parser.add_argument("--eval_weights", type=str, help="If set, performs evaluation only")
    # arguments added by mostafa
    parser.add_argument("--list_resnet_scene_layers", nargs='*', help='Please enter a list for the resnet scene layers', default=[3, 4, 6, 3, 2])
    parser.add_argument("--list_resnet_face_layers", nargs='*', help='Please enter a list for the resnet face layers', default=[3, 4, 6, 3, 2])
    parser.add_argument("--resnet_scene_inplanes", type=float, default=64, help="Resnet scene inplanes")
    parser.add_argument("--resnet_face_inplanes", type=float, default=64, help="Resnet face inplanes")
    parser.add_argument("--unet_inout_channels", type=float, default=1, help="Unet input output channels")
    parser.add_argument("--unet_inplanes", type=int, default=8, help="Unet inplanes")
    parser.add_argument("--unet_residual", type=int, default=1, help="Unet reisdual connections blocks")
    parser.add_argument("--list_unet_inplanes_multipliers", nargs='*', help='Please enter a list for the inplanes multipliers for unet', default=[1,2,3,4])
    parser.add_argument("--list_unet_attention_levels", nargs='*', help='Please enter a list for the attention levels for unet', default=[2,4])
    parser.add_argument("--unet_spatial_tf_heads", type=int, default=4, help="number of transfoermer heads")
    parser.add_argument("--unet_spatial_tf_layers", type=int, default=1, help="number of transformers")
    parser.add_argument("--unet_context_vector", type=int, default=1024, help="Unet context vector of spefic size ")
    parser.add_argument("--is_subsample_test_set", default=True, action="store_false", help="This flag is used to subsample the test set, set \
                        it to false so that you have access to the full test set")
    parser.add_argument("--Debugging_maps", default=False, action="store_true", help="This flag is used to output maps during training as figures")
    parser.add_argument("--watch_wandb", default=False, action="store_true", help="This flag is used to turn on the wandb watching feature")
    parser.add_argument("--schedular", default="linear", choices=["linear", "cosine", "sigmoid"])
    parser.add_argument("--diff_normalization", type=float, default=1, help="This is used to control the normalizaiton range of the diffusion model")
    parser.add_argument("--norm_std_flag", default=False, action="store_true", help="This flag is used to turn on the std during normalization")
    parser.add_argument("--predict_sthg", default="linear", choices=["pred_noise", "pred_x0"])
    parser.add_argument("--attention_layout", default=False, action="store_true", help="This flag will switch into a different design for the resnets including the enocder")
    parser.add_argument("--random_flag", default=False, action="store_true", help="This flag is used to make the heatmaps have different radius or even higher values ")
    parser.add_argument("--train_time_steps", type=int, default=1000, help="time steps for trainng the diffusion model")
    parser.add_argument("--sample_time_steps", type=int, default=250, help="time steps for evaluating the diffusion model ")
    parser.add_argument("--unet_dropout", type=float, default=0.0, help="This is used to control the drop of the unet")
    parser.add_argument("--offset_noise_strength", type=float, default=0.1, help="This is used to control the drop of the unet")
    parser.add_argument("--noise_changer", default=False, action="store_true", help="This is the noise that we change the noise ")
    # parser.add_argument("--cond_scale", type=float, default=1., help="When we set it to 1, we are not using classfier free guidance at sampling but when we set it for \
    #                     any value between 1 to 9 you are going more towards the classifier guidance stuffff  ")
    # parser.add_argument("--cond_drop_prob", type=float, default=0., help="By setting this value to 0 you are not using classifer free guidance at trainging and when you set it to more \
    #                     than 0 you start working with classfieir free guidance")
    # parser.add_argument("--rescaled_phi", type=float, default=0., help="It is used to set some specfic parameters ")
    parser.add_argument("--mse_loss_weight_type", default="constant")
    # the options can be (constant,min_snr_,max_snr_,trunc_snr,snr,inv_snr,)
    # ADM model paramters to be removed if not needed
    parser.add_argument("--adm", default=False, action="store_true", help="This is flag to swtich to the formulation and Unet of ADM model")
    parser.add_argument("--adm_use_kl", default=False, action="store_true", help="This is flag to swtich to the formulation and Unet of ADM model")
    parser.add_argument("--adm_predict_xstart", default=False, action="store_true", help="This is flag to swtich to the formulation and Unet of ADM model")
    parser.add_argument("--adm_predict_v", default=False, action="store_true", help="This is flag to swtich to the formulation v")
    parser.add_argument("--lr_schedular", default=False, action="store_true", help="This is flag to swtich to the lrschedular")
    parser.add_argument("--fix_snr", default=False, action="store_true", help="this is used to fix the SNR ratio of the model ")

    parser.add_argument("--adm_rescale_timesteps", default=False, action="store_true", help="This is flag to swtich to the formulation and Unet of ADM model")
    parser.add_argument("--adm_rescale_learned_sigmas", default=False, action="store_true", help="This is flag to swtich to the formulation and Unet of ADM model")
    parser.add_argument("--adm_learn_sigma", default=False, action="store_true", help="This is flag to swtich to the formulation and Unet of ADM model")
    parser.add_argument("--clip_denoised", default=False, action="store_true", help="This is flag to clip denosied of the model")
    parser.add_argument("--adm_attention_module", default=False, action="store_true", help="THis flag is used to change the attention to different path")

    parser.add_argument("--other_loss", default=False, action="store_true", help="THis flag is used activate another part of the loss that I am not sure if it is useful or not ")
    # for the schedulers
    parser.add_argument("--linear_noise_multiplier", type=float, default=1, help="This is used to control the noise in the linear schedular.")
    parser.add_argument("--cosine_s", type=float, default=0.008, help="This is used to control the s parameter in cosine schedule.")
    parser.add_argument("--sigmoid_start", type=float, default=-3, help="This is used to control the start parameter in sigmoid schedule.")
    parser.add_argument("--sigmoid_end", type=float, default=3, help="This is used to control the end parameter in sigmoid schedule.")
    parser.add_argument("--sigmoid_tau", type=float, default=1, help="This is used to control the tau parameter in sigmoid schedule.")
    parser.add_argument("--sigmoid_clamp_min", type=float, default=1e-5, help="This is used to control the clamp parameter in sigmoid schedule.")
    parser.add_argument("--eval_from_picture", default=False, action="store_true", help="When set to true , you are not starting from noise")
    parser.add_argument("--time_noise", type=int, default=-500, help="You noise till certain point .")
    parser.add_argument("--kl_div_loss", default=False, action="store_true", help="This is used to try to run KL divergence loss to turn it on")
    parser.add_argument("--x_loss", default=False, action="store_true", help="This is used to try to turn on the classes losse")
    # Training args
    parser.add_argument("--lr", type=float, default=2.5e-4, help="learning rate")# changed from 2.5e-4 to 8e-5
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--evaluate_every", type=int, default=5, help="evaluate every N epochs")
    parser.add_argument("--save_every", type=int, default=1, help="save model every N epochs")
    parser.add_argument("--print_every", type=int, default=10, help="print training stats every N batches")
    parser.add_argument("--no_resume", default=False, action="store_true", help="Resume from a stopped run if exists")
    parser.add_argument("--output_dir", type=str, default="output", help="Path to output folder")
    parser.add_argument("--amp", type=str, default=None, help="AMP optimization level")
    parser.add_argument("--channels_last", default=False, action="store_true")
    parser.add_argument("--freeze_scene", default=False, action="store_true", help="Freeze the scene backbone")
    parser.add_argument("--freeze_face", default=False, action="store_true", help="Freeze the head backbone")
    # some extra paramters 
    parser.add_argument("--ema_on", default=False, action="store_true", help="Work with the EMA weights")
    # parser.add_argument("--ema_rate", action='store', default="0.9999", help="This is the ema rate that you should enter, can be passed as string.")
    parser.add_argument("--ema_rate", type=float, default=0.9999, help="the ema rate value that you choose in here. ")
    parser.add_argument("--use_fp16", default=False, action="store_true", help="This is used to turn on fp 16 mixed precision")
    parser.add_argument("--fp16_scale_growth", type=float, default=1e-3, help="floating point scale growth")
    parser.add_argument("--lr_anneal_steps", type=float, default=0, help="the learning rate anenel steps")
    parser.add_argument("--weight_decay", type=float, default=0, help="the weighting decay value ")

    parser.add_argument("--no_wandb", default=False, action="store_true", help="Disables wandb")
    parser.add_argument(
        "--no_save",
        default=False,
        action="store_true",
        help="Do not save checkpoint every {save_every}. Stores last checkpoint only to allow resuming",
    )

    args = parser.parse_args()

    # Update output dir
    args.model_id = f"spatial_depth_late_fusion_{args.source_dataset}_{args.target_dataset}"
    args.output_dir = os.path.join(args.output_dir, args.model_id, args.tag)

    # Reverse resume flag to ease my life
    args.resume = not args.no_resume and args.eval_weights is None
    del args.no_resume

    # Reverse wandb flag
    args.wandb = not args.no_wandb
    del args.no_wandb

    # Reverse save flag
    args.save = not args.no_save
    del args.no_save

    # Check if AMP is set and is available. If not, remove amp flag
    if args.amp and amp is None:
        args.amp = None

    # Print configuration
    print(vars(args))
    ## Python vars() Function
    ## The vars() function returns the __dict__ attribute of an object
    ##
    print()

    return args
