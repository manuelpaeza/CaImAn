#!/usr/bin/env python
"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)

Written by Waleed Abdulla
Revised by Eric Thompson, Chanjia Cai, and Manuel Paez 
"""

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Model (Pre-trained on the COCO Dataset)
def get_model_instance_segmentation(num_classes):
    """
    Loads a pre-trained Mask R-CNN model and modifies its classification
    and mask prediction heads for a custom number of classes.

    Args:
        num_classes (int): The number of classes for the custom dataset,
                           including the background class.

    Returns:
        torch.nn.Module: The modified Mask R-CNN model ready for fine-tuning.
    """
    # load an instance segmentation model pre-trained on COCO, fpn_v2 provides better performance
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='COCO_V1', trainable_backbone_layers=3)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 
                                                        hidden_layer, 
                                                        num_classes)
    return model