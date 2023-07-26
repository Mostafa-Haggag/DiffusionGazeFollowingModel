import os

from torch.utils.data import DataLoader

from .GazeFollow import GazeFollow
from .VideoAttentionTarget import VideoAttentionTargetImages


def get_loader(name: str, root_dir: str, random_flag=False,input_size=224, output_size=64, batch_size=48, num_workers=6, is_train=True,is_subsample_test_set=True,gaze_point_threshold=0):
    if name == "gazefollow":# we enter into here
        labels = os.path.join(root_dir, "train_annotations_release.txt" if is_train else "test_annotations_release.txt")
        # root_dir datasets/gazefollow_extended
        # labels they are coming from file 
        dataset = GazeFollow(root_dir, labels, random_size=random_flag,input_size=input_size, output_size=output_size, is_test_set=not is_train,is_subsample_test_set=is_subsample_test_set,gaze_point_threshold=gaze_point_threshold)
        loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers, pin_memory=True
        )
    elif name == "videoattentiontarget":
        labels = os.path.join(root_dir, "annotations/train" if is_train else "annotations/test")
        dataset = VideoAttentionTargetImages(
            root_dir, labels, random_size=random_flag,input_size=input_size, output_size=output_size, is_test_set=not is_train
        )
        loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers, pin_memory=True
        )
    else:
        raise ValueError(f"Invalid dataset: {name}")

    return loader

### Function to get all of the datasetts 
def get_dataset(config):
    # source dataset "datasets/gazefollow_extended"
    source_loader = get_loader(
        name=config.source_dataset,
        root_dir=config.source_dataset_dir,
        random_flag=config.random_flag,
        input_size=config.input_size,
        output_size=config.output_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        is_train=True,
    )

    target_test_loader = get_loader(
        name=config.source_dataset,
        root_dir=config.source_dataset_dir,
        random_flag=config.random_flag,
        input_size=config.input_size,
        output_size=config.output_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        is_train=False,
        is_subsample_test_set=config.is_subsample_test_set,
        gaze_point_threshold=config.gaze_point_threshold,
    )

    return source_loader, target_test_loader
