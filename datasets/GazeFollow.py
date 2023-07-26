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

from datasets.transforms.ToColorMap import ToColorMap
from utils import get_head_mask, get_label_map

class GazeFollow(Dataset):
    def __init__(self, data_dir, labels_path,random_size=False, input_size=224, output_size=64, is_test_set=False,is_subsample_test_set=True,
                 gaze_point_threshold=0):
        self.data_dir = data_dir
        self.input_size = input_size
        self.output_size = output_size
        self.is_test_set = is_test_set
        self.gaze_point_threshold=gaze_point_threshold
        self.head_bbox_overflow_coeff = 0.1  
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.random_size=random_size

    
        column_names = [
            "path",
            "idx",
            "body_bbox_x",#bounding box body 
            "body_bbox_y",
            "body_bbox_w",
            "body_bbox_h",
            "eye_x",
            "eye_y",
            "gaze_x",
            "gaze_y",
            "bbox_x_min",#head bounding box
            "bbox_y_min",
            "bbox_x_max",
            "bbox_y_max",
        ]
        if is_test_set is False:
            column_names.append("inout")

        df = pd.read_csv(labels_path, sep=",", names=column_names, usecols=column_names, index_col=False)
        if is_test_set:
            # to allow for subsampling of the test set
            if self.gaze_point_threshold > 0:
                original_size = len(df)
                # Copy df
                variance_df = df.copy()
                variance_df = variance_df.groupby(["path", "eye_x"])
                dists = []
                # Iterate each group
                for _, group in variance_df:
                    # Get gaze_x and gaze_y
                    gaze_x = torch.tensor(group["gaze_x"].values)
                    gaze_y = torch.tensor(group["gaze_y"].values)

                    gaze_points = torch.stack((gaze_x, gaze_y), dim=1)
                    # Calculate average point
                    avg_point = gaze_points.mean(dim=0)

                    # Calculate distance from average point
                    dist = torch.norm(gaze_points - avg_point, dim=1)
                    dists += dist.tolist()

                # Sort distances
                dists.sort()
                # Get distance value that is 10th percentile
                threshold = dists[int(len(dists) * self.gaze_point_threshold)]
                drop_list = []
                # Iterate each group
                for _, group in variance_df:
                    # Get gaze_x and gaze_y
                    gaze_x = torch.tensor(group["gaze_x"].values)
                    gaze_y = torch.tensor(group["gaze_y"].values)

                    gaze_points = torch.stack((gaze_x, gaze_y), dim=1)

                    # Calculate average point
                    avg_point = gaze_points.mean(dim=0)

                    # Calculate distance from average point
                    dist = torch.norm(gaze_points - avg_point, dim=1)
                    drop_list += group.iloc[
                        (dist > threshold).nonzero().flatten()
                    ].index.tolist()
                df.drop(df.index[drop_list], inplace=True)
                after_size = len(df)
                print("The size of data set was {0} then we became {1}, so we removed {2} sample in the run".format(original_size,after_size,original_size-after_size))
            if is_subsample_test_set:
                df = df.sample(frac=0.1)
                df = df[
                    ["path", "eye_x", "eye_y", "gaze_x", "gaze_y", "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max"]
                ].groupby(["path", "eye_x"])
                self.keys = list(df.groups.keys())
                self.X = df
                self.length = len(self.keys)
            else:
                df = df[
                    ["path", "eye_x", "eye_y", "gaze_x", "gaze_y", "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max"]
                ].groupby(["path", "eye_x"])
                self.keys = list(df.groups.keys())
                self.X = df
                self.length = len(self.keys)

        else:
            '''
            Index(['index', 'path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w',
            'body_bbox_h', 'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'bbox_x_min',
            'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'inout'],
            dtype='object')
            '''
            # only use "in" or "out "gaze (-1 is invalid, 0 is out gaze)
            # modifiyed by the instruction of fernando
            df = df[df["inout"] != -1]  
            df.reset_index(inplace=True)
            # path of all of the pictures
            self.X = df["path"] 
            self.y = df[
                ["bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max", "eye_x", "eye_y", "gaze_x", "gaze_y", "inout"]
            ]# coordinates
            self.length = len(df)# df has shape of 124095 by 16 len
            # length -= 124095
    def __getitem__(self, index):
        if self.is_test_set:
            return self.__get_test_item__(index)
        else:
            return self.__get_train_item__(index)

    def __len__(self):
        return self.length

    def __get_train_item__(self, index):
        # get a specific image can eb turned into ''train/00000023/00023976.jpg''
        path = self.X.iloc[index]
        # get teh values of this index
        x_min, y_min, x_max, y_max, eye_x, eye_y, gaze_x, gaze_y, gaze_inside = self.y.iloc[index] 

        x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)# you are subtracting
        y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
        x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)# you are adding 
        y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)

        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert("RGB")
        width, height = img.size
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

        # Data augmentation
        # Jitter (expansion-only) bounding box size
        if np.random.random_sample() <= 0.5:
            self.head_bbox_overflow_coeff = np.random.random_sample() * 0.2
            x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)
            y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
            x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)
            y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)

        
        # Random Crop

        if np.random.random_sample() <= 0.5:
            crop_x_min = np.min([gaze_x * width, x_min, x_max])#120.3
            crop_y_min = np.min([gaze_y * height, y_min, y_max])# 178.11
            crop_x_max = np.max([gaze_x * width, x_min, x_max])
            crop_y_max = np.max([gaze_y * height, y_min, y_max])

            # Randomly select a random top left corner
            if crop_x_min >= 0:
                crop_x_min = np.random.uniform(0, crop_x_min)
            if crop_y_min >= 0:
                crop_y_min = np.random.uniform(0, crop_y_min)# make sure that there is no negative value set 

            # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
            crop_width_min = crop_x_max - crop_x_min#199.28
            crop_height_min = crop_y_max - crop_y_min# 564

            
            
            # width 512
            # height 768
            crop_width_max = width - crop_x_min
            crop_height_max = height - crop_y_min

            
            # Randomly select a width and a height
            crop_width = np.random.uniform(crop_width_min, crop_width_max)
  
            crop_height = np.random.uniform(crop_height_min, crop_height_max)

            # Crop it
            img = crop(img, crop_y_min, crop_x_min, crop_height, crop_width)

            # Record the crop's (x, y) offset
            offset_x, offset_y = crop_x_min, crop_y_min

            # Convert coordinates into the cropped frame
            x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y
            gaze_x, gaze_y = (gaze_x * width - offset_x) / float(crop_width), (gaze_y * height - offset_y) / float(
                crop_height
            )

            width, height = crop_width, crop_height

        # Random flip
        if np.random.random_sample() <= 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            x_max_2 = width - x_min
            x_min_2 = width - x_max
            x_max = x_max_2
            x_min = x_min_2
            gaze_x = 1 - gaze_x

        # Random color change
        if np.random.random_sample() <= 0.5:
            img = adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))

            img = adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))

            img = adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        head = get_head_mask(x_min, y_min, x_max, y_max, width, height, resolution=self.input_size).unsqueeze(0)
        # get the mask black and white for the head 
        
        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # Apply transformation to images...
        if self.image_transform is not None:
            img = self.image_transform(img)
            face = self.image_transform(face)



        # Generate the heat map used for deconv prediction
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)
        if self.random_size:
                sigma = random.randint(7, 10)
        else: 
            sigma= 3
        gaze_heatmap = get_label_map(
            gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size], sigma, pdf="Gaussian"
        )

        gaze_coords = (gaze_x, gaze_y)# create a tuple 
        eye_coords = (eye_x, eye_y)
        information_tensor =  torch.tensor([x_min*self.input_size/width,y_min*self.input_size/height,
                                            x_max*self.input_size/width,y_max*self.input_size/height]).numpy()
        information_tensor = information_tensor.astype(int)
        information_tensor = np.clip(information_tensor, 0, self.input_size - 1)
        return (
            img,
            # depth,
            face,
            head,
            gaze_heatmap,
            torch.FloatTensor([eye_coords]),
            torch.FloatTensor([gaze_coords]),
            torch.IntTensor([bool(gaze_inside)]),
            torch.IntTensor([width, height]),
            path,
            information_tensor
        )

    def __get_test_item__(self, index):
        eye_coords = []
        gaze_coords = []
        gaze_inside = []
        for _, row in self.X.get_group(self.keys[index]).iterrows():
            path = row["path"]
            x_min = row["bbox_x_min"]
            y_min = row["bbox_y_min"]
            x_max = row["bbox_x_max"]
            y_max = row["bbox_y_max"]
            gaze_x = row["gaze_x"]
            gaze_y = row["gaze_y"]
            eye_x = row["eye_x"]
            eye_y = row["eye_y"]
            # All ground truth gaze are stacked up
            eye_coords.append([eye_x, eye_y])
            gaze_coords.append([gaze_x, gaze_y])
            gaze_inside.append(True)

        for _ in range(len(gaze_coords), 20):
            # Pad dummy gaze to match size for batch processing
            eye_coords.append([-1, -1])
            gaze_coords.append([-1, -1])
            gaze_inside.append(False)
        eye_coords = torch.FloatTensor(eye_coords)
        gaze_coords = torch.FloatTensor(gaze_coords)
        gaze_inside = torch.IntTensor(gaze_inside)

        # Expand face bbox a bit
        x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
        x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)

        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert("RGB")
        width, height = img.size
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

        head = get_head_mask(x_min, y_min, x_max, y_max, width, height, resolution=self.input_size).unsqueeze(0)

        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))


        # Apply transformation to images...
        if self.image_transform is not None:
            img = self.image_transform(img)
            face = self.image_transform(face)


        # Generate the heat map used for deconv prediction
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)
        num_valid = 0
        for gaze_x, gaze_y in gaze_coords:
            if gaze_x == -1:
                continue

            num_valid += 1
            if self.random_size:
                sigma = random.randint(7, 10)
            else: 
                sigma= 3
            gaze_heatmap = get_label_map(
                gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size], sigma, pdf="Gaussian"
            )
        gaze_heatmap /= num_valid

        gaze_heatmap/=torch.max(gaze_heatmap)
        information_tensor =  torch.tensor([x_min*self.input_size/width,y_min*self.input_size/height,
                                            x_max*self.input_size/width,y_max*self.input_size/height]).numpy()
        information_tensor = information_tensor.astype(int)
        information_tensor = np.clip(information_tensor, 0, self.input_size - 1)
        return (
            img,
            # depth,
            face,
            head,
            gaze_heatmap,
            eye_coords,
            gaze_coords,
            gaze_inside,
            torch.IntTensor([width, height]),
            path,
            information_tensor
        )

    def get_head_coords(self, path):
        if not self.is_test_set:
            raise NotImplementedError("This method is not implemented for training set")

        # NOTE: this is not 100% accurate. I should also condition by eye_x
        # However, for the application of this method it should be enough
        key_index = next((key for key in self.keys if key[0] == path), -1)
        if key_index == -1:
            raise RuntimeError("Path not found")

        for _, row in self.X.get_group(key_index).iterrows():
            x_min = row["bbox_x_min"]
            y_min = row["bbox_y_min"]
            x_max = row["bbox_x_max"]
            y_max = row["bbox_y_max"]

        # Expand face bbox a bit
        x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
        x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)

        return x_min, y_min, x_max, y_max


