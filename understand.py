from omegaconf import OmegaConf

# Load the configuration file
config = OmegaConf.load("diffusion.yaml")
use_augmentation = config.Diffusion.list_unet_inplanes_multipliers
shuffle_data = config.Dataset.source_dataset_dir
print(type(config))
print(f"Use Augmentation: {use_augmentation}")
print(f"Shuffle Data: {shuffle_data}")