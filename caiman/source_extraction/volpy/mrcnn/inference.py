#!/usr/bin/env python
"""
Mask R-CNN
Inference for Model functions

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Revised by Eric Thompson, Changjia Cai, and Manuel Paez 
"""

import numpy as np
import torch 

from caiman.source_extraction.volpy.mrcnn.utils import ScaleImage, data_transform

def mrcnn_infer(model, 
                    img, 
                    eval_transform, 
                    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), 
                    thresh=0.5):
    """
    inference using Mask R-CNN network
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = eval_transform(img)
        x = x.to(device)
        predictions = model([x, ])
        pred = predictions[0]
    
    predicted_masks, predicted_boxes = thresholded_predictions(pred, threshold=thresh) 
    binarized_masks = (0.5+predicted_masks).detach().cpu().numpy().astype(np.uint8) 
    return predicted_masks, predicted_boxes, binarized_masks

def mrcnn_pytorch(model, img, size_range, confidence_threshold=0.5):
    """
    Performs inference using the PyTorch Mask R-CNN model and filters the results.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1)
    img_tensor = ScaleImage()(img_tensor) #Apply the same 0-1 scaling used during training
    img_tv_tensor = torchvision.tv_tensors.Image(img_tensor) #Wrap the tensor in the tv_tensors.Image class, as expected by the transform pipeline

    # Perform inference using the existing function
    _, _, binarized_masks = mrcnn_inference(
        model,
        img=img_tv_tensor, 
        thresh=confidence_threshold,
        eval_transform=data_transform(train=False),
        device=device
    )

    print(f"Model detected {len(binarized_masks)} raw masks before size filtering.")

    if binarized_masks.size == 0:
        return np.empty((0, *img.shape[:2]), dtype=bool)

    # Post-process the masks to filter by size
    mask_areas = binarized_masks.sum(axis=(1, 2))
    selection = np.logical_and(mask_areas > size_range[0] ** 2,
                               mask_areas < size_range[1] ** 2)
    
    filtered_masks = binarized_masks[selection]
    return filtered_masks.astype(bool)

def thresholded_predictions(pred, threshold=0.7):
    """
    Get masks and boxes for those above threshold
    """
    numels = len(torch.where(pred['scores'] >= threshold)[0])
    masks = pred['masks'][:numels].squeeze()
    boxes = pred['boxes'][:numels]
    
    return masks, boxes 
