import argparse
import os
# from datetime import datetime

#datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None
from omegaconf import OmegaConf


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--yaml_diffusion", default="diffusion.yaml")
    parser.add_argument("--yaml_gaze", default="gaze.yaml")
    parser.add_argument("--tag", default="default", help="Description of this run")

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
