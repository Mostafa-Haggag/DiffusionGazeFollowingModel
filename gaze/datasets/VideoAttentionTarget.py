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


class VideoAttentionTargetImages(Dataset):
    def __init__(self, data_dir, labels_dir,random_size=False,depth_on=False, input_size=224, output_size=64, is_test_set=False,sigma=3):
        self.data_dir = data_dir
        self.input_size = input_size
        self.output_size = output_size
        self.is_test_set = is_test_set
        self.depth_on = depth_on
        self.sigma=sigma
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
        for show_dir in glob.glob(os.path.join(labels_dir, "*")):
            for sequence_path in glob.glob(os.path.join(show_dir, "*", "*.txt")):
                df = pd.read_csv(
                    sequence_path,
                    header=None,
                    index_col=False,
                    names=["path", "xmin", "ymin", "xmax", "ymax", "gazex", "gazey"],
                )
                # sampling 20% from the full data set
                df = df.sample(frac=0.2,random_state=42)

                show_name = sequence_path.split("/")[-3]
                clip = sequence_path.split("/")[-2]

                df["path"] = df["path"].apply(lambda path: os.path.join(show_name, clip, path))

                self.X.extend(df.values.tolist())

        self.length = len(self.X)

        print(f"Total images: {self.length} (is test set? {is_test_set})")

    def __getitem__(self, index):
        if self.is_test_set:
            return self.__get_test_item__(index)
        else:
            return self.__get_train_item__(index)

    def __len__(self):
        return self.length

    def __get_train_item__(self, index):
        path, x_min, y_min, x_max, y_max, gaze_x, gaze_y = self.X[index]

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

        # cond for data augmentation
        cond_jitter = np.random.random_sample()
        cond_flip = np.random.random_sample()
        cond_color = np.random.random_sample()
        if cond_color < 0.5:
            n1 = np.random.uniform(0.5, 1.5)
            n2 = np.random.uniform(0.5, 1.5)
            n3 = np.random.uniform(0.5, 1.5)
        cond_crop = np.random.random_sample()

        if cond_crop < 0.5:
            sliced_x_min = x_min
            sliced_x_max = x_max
            sliced_y_min = y_min
            sliced_y_max = y_max

            sliced_gaze_x = gaze_x
            sliced_gaze_y = gaze_y

            # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
            if not gaze_inside:
                crop_x_min = np.min([sliced_x_min, sliced_x_max])
                crop_y_min = np.min([sliced_y_min, sliced_y_max])
                crop_x_max = np.max([sliced_x_min, sliced_x_max])
                crop_y_max = np.max([sliced_y_min, sliced_y_max])
            else:
                crop_x_min = np.min([sliced_gaze_x, sliced_x_min, sliced_x_max])
                crop_y_min = np.min([sliced_gaze_y, sliced_y_min, sliced_y_max])
                crop_x_max = np.max([sliced_gaze_x, sliced_x_min, sliced_x_max])
                crop_y_max = np.max([sliced_gaze_y, sliced_y_min, sliced_y_max])

            # Randomly select a random top left corner
            if crop_x_min >= 0:
                crop_x_min = np.random.uniform(0, crop_x_min)
            if crop_y_min >= 0:
                crop_y_min = np.random.uniform(0, crop_y_min)

            # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
            crop_width_min = crop_x_max - crop_x_min
            crop_height_min = crop_y_max - crop_y_min
            crop_width_max = width - crop_x_min
            crop_height_max = height - crop_y_min

            # Randomly select a width and a height
            crop_width = np.random.uniform(crop_width_min, crop_width_max)
            crop_height = np.random.uniform(crop_height_min, crop_height_max)

        face_x1 = x_min
        face_y1 = y_min
        face_x2 = x_max
        face_y2 = y_max

        face_x1, face_y1, face_x2, face_y2 = map(float, [face_x1, face_y1, face_x2, face_y2])
        gaze_x, gaze_y = map(float, [gaze_x, gaze_y])

        # Data augmentation
        # Jitter (expansion-only) bounding box size.
        if cond_jitter < 0.5:
            k = cond_jitter * 0.1
            face_x1 -= k * abs(face_x2 - face_x1)
            face_y1 -= k * abs(face_y2 - face_y1)
            face_x2 += k * abs(face_x2 - face_x1)
            face_y2 += k * abs(face_y2 - face_y1)
            face_x1 = np.clip(face_x1, 0, width)
            face_x2 = np.clip(face_x2, 0, width)
            face_y1 = np.clip(face_y1, 0, height)
            face_y2 = np.clip(face_y2, 0, height)

        # Random Crop
        if cond_crop < 0.5:
            # Crop it
            img = crop(img, crop_y_min, crop_x_min, crop_height, crop_width)

            # Record the crop's (x, y) offset
            offset_x, offset_y = crop_x_min, crop_y_min

            # Convert coordinates into the cropped frame
            face_x1, face_y1, face_x2, face_y2 = (
                face_x1 - offset_x,
                face_y1 - offset_y,
                face_x2 - offset_x,
                face_y2 - offset_y,
            )
            if gaze_inside:
                gaze_x, gaze_y = (gaze_x - offset_x), (gaze_y - offset_y)
            else:
                gaze_x = -1
                gaze_y = -1

            width, height = crop_width, crop_height

        # Flip?
        if cond_flip < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            x_max_2 = width - face_x1
            x_min_2 = width - face_x2
            face_x2 = x_max_2
            face_x1 = x_min_2
            if gaze_x != -1 and gaze_y != -1:
                gaze_x = width - gaze_x

        # Random color change
        if cond_color < 0.5:
            img = adjust_brightness(img, brightness_factor=n1)
            img = adjust_contrast(img, contrast_factor=n2)
            img = adjust_saturation(img, saturation_factor=n3)

        # Face crop
        face = img.copy().crop((int(face_x1), int(face_y1), int(face_x2), int(face_y2)))
        eye_x = np.mean([face_x1, face_x2]) / width
        eye_y = np.mean([face_y1, face_y2]) / height

        # Head channel image
        head = get_head_mask(face_x1, face_y1, face_x2, face_y2, width, height, resolution=self.input_size).unsqueeze(0)
        # Load depth image
        if self.depth_on:
            depth = Image.open(os.path.join(self.data_dir, "depths", path))
            depth = depth.convert("L")
            # ... and depth
            if self.depth_transform is not None:
                depth = self.depth_transform(depth)

        # Apply transformation to image, face...
        if self.image_transform is not None:
            img = self.image_transform(img)
            face = self.image_transform(face)



        # Deconv output
        if gaze_inside:
            gaze_x /= float(width)  # fractional gaze
            gaze_y /= float(height)
            gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
            if self.random_size:
                    sigma = random.randint(7, 10)
            else: 
                    sigma= self.sigma
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
                path,
                information_tensor
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
                path,
                information_tensor
            )

    def __get_test_item__(self, index):
        (path, x_min, y_min, x_max, y_max, gaze_x, gaze_y) = self.X[index]

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
            depth = Image.open(os.path.join(self.data_dir, "depths", path))
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
                    sigma= self.sigma
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