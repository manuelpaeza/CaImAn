"""
Mask R-CNN
Train on the segmentation of neurons.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Revised by Eric Thompson, Chanjia Cai, and Manuel Paez 
"""

import os
import sys
import json
import datetime
import numpy as np
import torch 
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import torchvision.transforms.v2 as T
from torchvision.ops.boxes import masks_to_boxes

from .utils import ScaleImage, create_mask


# Root directory of the project
# ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
#sys.path.append(ROOT_DIR)  # To find local version of the library
# from ..mrcnn.config import Config
# Path to trained weights file
# COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

#  Dataset
class NeuronsDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.image_filenames = list(sorted(os.listdir(os.path.join(self.root, "images"))))
        self.mask_filenames = list(sorted(os.listdir(os.path.join(self.root, "masks"))))

    def __getitem__(self, idx):
        image_id = idx

        image_scaler = ScaleImage()

        # Image: (C x H x W)
        image_path = os.path.join(self.root, "images", self.image_filenames[idx])
        image = np.load(image_path)['img'] # mean/mean/corr channels  (h w c)
        image = torch.from_numpy(image).permute(2,0,1) # convert to tensor and get into pytorch order C x H x W
        image = image_scaler(image)   # scale so it is in 0,1 range
        image = tv_tensors.Image(image)

        # Masks: N x H x W mask array (N masks)
        mask_path = os.path.join(self.root, "masks", self.mask_filenames[idx])
        masks_loaded = np.load(mask_path, allow_pickle=True)
        masks = masks_loaded['mask']
        # first create boolean mask stack
        all_masks = []
        for mask_ind, mask_dict in enumerate(masks): # [mask_ind]
            mask = create_mask(image[1].shape, mask_dict)
            all_masks.append(mask)
        all_masks = np.array(all_masks)
        # then convert to binary uint8 tensor stack
        all_masks = torch.from_numpy(all_masks.astype(np.uint8))
                
        boxes = masks_to_boxes(all_masks)
        box_areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # tensor of areas

        # there is only one class, so labels are all ones
        num_objs = len(masks)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # let's just say nstances are not crowd: all instances will be used for evaluation
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap up everything into a dictionary describing target
        target = {}
        target["image_id"] = image_id
        target["masks"] = tv_tensors.Mask(all_masks)
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(image))
        target["labels"] = labels
        target["area"] = box_areas
        target["iscrowd"] = iscrowd

        # run augmentation, if transforms exist
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        return image, target
        
    def __len__(self):
        return len(self.image_filenames)
        
    def print_image_filenames(self):
        for image_filename in self.image_filenames:
            print(image_filename)

    def print_mask_filenames(self):
        for mask_filename in self.mask_filenames:
            print(mask_filename)

def data_transform(train=False):
    transform_pipeline = []
    if train:
        transform_pipeline.append(T.ColorJitter(brightness=0.5,
                                                contrast=0.5,
                                                saturation=0.5,
                                                hue=0))  # set as 0 as it mixes three channels
        transform_pipeline.append(T.GaussianBlur(kernel_size=(5,5), 
                                                         sigma=(0.001, 0.3))) # sigma min, max 
        transform_pipeline.append(T.RandomHorizontalFlip(p=0.5))
        transform_pipeline.append(T.RandomVerticalFlip(p=0.5))
        #transform_pipeline.append(T.RandomRotation(4, fill=0, expand=False))  
        transform_pipeline.append(T.SanitizeBoundingBoxes(min_size=2)) 
        
    # Convert to proper type and compose
    transform_pipeline.append(T.ToDtype(torch.float32, scale=True))
    transform_pipeline.append(T.ToPureTensor())
    return T.Compose(transform_pipeline)
