#!/usr/bin/env python

import numpy as np
import os
import h5py
import torch
import torch.nn as nn
import torchvision

import caiman as cm
from caiman.paths import caiman_datadir
from caiman.utils.utils import download_model, download_demo
from caiman.source_extraction.volpy.mrcnn import neurons
import caiman.source_extraction.volpy.mrcnn.model as modellib

from caiman.source_extraction.volpy.mrcnn.model import get_model_instance_segmentation, mrcnn_inference
from caiman.source_extraction.volpy.mrcnn.neurons import data_transform 
from caiman.source_extraction.volpy.mrcnn.utils import ScaleImage
from caiman.source_extraction.volpy.mrcnn.config import Config

def mrcnn_pytorch_inference(model, img, size_range, confidence_threshold=0.5):
    """
    Performs inference using the PyTorch Mask R-CNN model and filters the results.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ---> FIX: Preprocessing pipeline now exactly matches the training dataset <---

    # 1. Convert numpy image (H, W, C) to a tensor (C, H, W)
    img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1)
    
    # 2. Apply the same 0-1 scaling used during training
    img_tensor = ScaleImage()(img_tensor)
    
    # 3. Wrap the tensor in the tv_tensors.Image class, as expected by the transform pipeline
    img_tv_tensor = torchvision.tv_tensors.Image(img_tensor)

    # Perform inference using the existing function
    _, _, binarized_masks = mrcnn_inference(
        model,
        img=img_tv_tensor,  # Pass the correctly wrapped tensor
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

def mrcnn(img, size_range, weights_path, confidence_threshold):
    config = neurons.NeuronsConfig()
    
    class InferenceConfig(config.__class__):
        # Run detection on one img at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.7
        IMAGE_RESIZE_MODE = "pad64"
        IMAGE_MAX_DIM = 512
        RPN_NMS_THRESHOLD = 0.7
        POST_NMS_ROIS_INFERENCE = 1000

    config = InferenceConfig()
    config.display()
    model_dir = os.path.join(caiman_datadir(), 'model')
    
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=model_dir,
                                  config=config)
        tf.keras.Model.load_weights(model.keras_model, weights_path, by_name=True)
        results = model.detect([img], verbose=1)
        r = results[0]
        selection = np.logical_and(r['masks'].sum(axis=(0,1)) > size_range[0] ** 2, 
                                   r['masks'].sum(axis=(0,1)) < size_range[1] ** 2)
        r['masks'] = r['masks'][:, :, selection]
        ROIs = r['masks'].transpose([2, 0, 1])   
    return ROIs

def test_mrcnn():
    weights_path = download_model('mask_rcnn') 
    print("weights_path", weights_path)  
    summary_images = cm.load(download_demo('demo_voltage_imaging_summary_images.tif'))
    ROIs = mrcnn(img=summary_images.transpose([1, 2, 0]), size_range=[5, 22], 
            weights_path=weights_path)
    assert ROIs.shape[0] == 14, 'fail to infer correct number of neurons'

def test_mrcnn_pytorch():
    """
    Test function for the PyTorch Mask R-CNN neuron detector.
    """
    # Load configuration to get model paths
    config = Config()

    # --- Use the PyTorch model weights ---
    # This test assumes you have already trained a model and saved the weights.
    # We point to the final epoch model from the training script.
    weights_path = os.path.join(config.MODEL_SAVE_DIR, f'mrcnn_epoch_{config.NUM_EPOCHS}.pt')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"PyTorch model weights not found at: {weights_path}\n"
                              "Please run the training script first.")
    print(f"Using PyTorch weights from: {weights_path}")

    # --- Load the model architecture and state ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model_instance_segmentation(num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # --- Load the same CaImAn demo data ---
    summary_images_path = download_demo('demo_voltage_imaging_summary_images.tif')
    summary_images = cm.load(summary_images_path)

    # --- Run inference with the PyTorch function ---
    # The original test transposes to (height, width, channels), which our function expects.
    ROIs = mrcnn_pytorch_inference(
        model=model,
        img=summary_images.transpose([1, 2, 0]),
        size_range=[5, 22],
        confidence_threshold=0.5
    )

    print(f"Inference complete. Found {ROIs.shape[0]} neurons.")
    # --- Assert the number of neurons found ---
    # Note: The number of detected neurons might differ from the original
    # TensorFlow model. Adjust the assertion number based on your model's performance.
    # For this example, we'll keep the original assertion value.
    assert ROIs.shape[0] == 14, f"Test failed: Expected 14 neurons, but found {ROIs.shape[0]}."
    print("\nTest passed successfully!")

if __name__ == "__main__":
    test_mrcnn_pytorch()
    # test_mrcnn()