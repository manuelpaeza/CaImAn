"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)

Written by Waleed Abdulla
Revised by Eric Thompson, Chanjia Cai, and Manuel Paez 
"""

import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .utils import thresholded_predictions 

# Model (Pre-trained on the COCO Dataset with last-layer fine-tuned)
############################################################

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO, fpn_v2 provides better performance
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='COCO_V1', trainable_backbone_layers=3)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def mrcnn_inference(model, img, eval_transform, 
                    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), 
                    thresh=0.5):
    """
    inference using Mask R-CNN network
    """
    model.eval()
    with torch.no_grad():
        x = eval_transform(img)
        x = x.to(device)
        predictions = model([x, ])
        pred = predictions[0]
    
    predicted_masks, predicted_boxes = thresholded_predictions(pred, threshold=thresh) 
    binarized_masks = (0.5+predicted_masks).detach().cpu().numpy().astype(np.uint8) 
    return predicted_masks, predicted_boxes, binarized_masks
