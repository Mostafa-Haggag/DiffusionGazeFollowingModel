import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
import random 
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    crop,
)

from gaze.datasets.transforms.ToColorMap import ToColorMap
from utils import get_head_mask, get_label_map


class MHUGImages(Dataset):
    def __init__(self, data_dir, labels_dir,random_size=False,depth_on=False, input_size=224, output_size=64,):
        self.data_dir = data_dir
        self.input_size = input_size
        self.output_size = output_size
        self.depth_on = depth_on

        self.head_bbox_overflow_coeff = 0.1  # Will increase/decrease the bbox of the head by this value (%)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.random_size=random_size
        self.depth_transform = transforms.Compose(
            [ToColorMap(plt.get_cmap("magma")), transforms.Resize((input_size, input_size)), transforms.ToTensor()]
        )


        self.X = []
        # all_dfs = []

        for show_dir in glob.glob(os.path.join(labels_dir, "*")):
            for sequence_path in glob.glob(os.path.join(show_dir, "*.csv")):
                df = pd.read_csv(
                    sequence_path,
                    header=None,
                    index_col=False,
                    names=["path","person_index" ,"xmin", "ymin", "xmax", "ymax", "gazex", "gazey"],
                )
                # sampling 20% from the full data set
                # df = df.sample(frac=0.2)

                show_name = sequence_path.split("/")[-3]
                clip = sequence_path.split("/")[-2]

                df["path"] = df["path"].apply(lambda path: os.path.join(clip,"images",path))
                # all_dfs.append(df)  # Append each DataFrame to the list
                self.X.extend(df.values.tolist())
        self.length = len(self.X)
        # final_df = pd.concat(all_dfs, axis=0, ignore_index=True)
        # final_df.to_csv('meow.csv')
        print(f"Total images: {self.length}")

    def __getitem__(self, index):
        return self.__get_test_item__(index)


    def __len__(self):
        return self.length

    def __get_test_item__(self, index):
        (path,person_index ,x_min, y_min, x_max, y_max, gaze_x, gaze_y) = self.X[index]

        img = Image.open(os.path.join(self.data_dir, "images", path))
        img = img.convert("RGB")
        width, height = img.size
        x_min, y_min, x_max, y_max, gaze_x, gaze_y = map(float, [x_min, y_min, x_max, y_max, gaze_x, gaze_y])

        if gaze_x == -1 and gaze_y == -1:
            gaze_inside = False
        else:
            if gaze_x < 0:  # move gaze point that was slightly outside the image back in
                gaze_x = 0
            if gaze_y < 0:
                gaze_y = 0
            gaze_inside = True

        # Crop the face
        face = img.copy().crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        eye_x = np.mean([x_max, x_min]) / width
        eye_y = np.mean([y_max, y_min]) / height

        head = get_head_mask(x_min, y_min, x_max, y_max, width, height, resolution=self.input_size).unsqueeze(0)

        if self.depth_on:
            # Load depth image
            depth = Image.open(os.path.join(self.data_dir, "depth/images", path))
            depth = depth.convert("L")
        # ... and depth
            if self.depth_transform is not None:
                depth = self.depth_transform(depth)

        # Apply transformation to images...
        if self.image_transform is not None:
            img = self.image_transform(img)
            face = self.image_transform(face)


        if gaze_inside:
            gaze_x /= float(width)
            gaze_y /= float(height)
            gaze_heatmap = torch.zeros(self.output_size, self.output_size)
            if self.random_size:
                    sigma = random.randint(7, 10)
            else: 
                    sigma= 3
            # set the size of the output
            gaze_heatmap = get_label_map(
                gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size], sigma, pdf="Gaussian"
            )
        else:
            gaze_heatmap = torch.zeros(self.output_size, self.output_size)

        eye_coords = (eye_x, eye_y)
        gaze_coords = (gaze_x, gaze_y)
        information_tensor =  torch.tensor([x_min*self.input_size/width,y_min*self.input_size/height,
                                            x_max*self.input_size/width,y_max*self.input_size/height]).numpy()
        information_tensor = information_tensor.astype(int)
        information_tensor = np.clip(information_tensor, 0, self.input_size - 1)
        if self.depth_on:
            return (
                img,
                depth,
                face,
                head,
                gaze_heatmap,
                torch.FloatTensor([eye_coords]),
                torch.FloatTensor([gaze_coords]),
                torch.IntTensor([gaze_inside]),
                torch.IntTensor([width, height]),
                path,information_tensor
            )
        else:
            return (
                img,
                # depth,
                face,
                head,
                gaze_heatmap,
                torch.FloatTensor([eye_coords]),
                torch.FloatTensor([gaze_coords]),
                torch.IntTensor([gaze_inside]),
                torch.IntTensor([width, height]),
                path,information_tensor
            )