#!/usr/bin/env python

import numpy as np
import os
import h5py
import torch
import torch.nn as nn
from collections import OrderedDict
import warnings

import caiman as cm
from caiman.paths import caiman_datadir
from caiman.utils.utils import download_model, download_demo
from caiman.source_extraction.volpy.mrcnn import neurons
import caiman.source_extraction.volpy.mrcnn.model as modellib

import torchvision.transforms.functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class ConvBoxHead(nn.Module):
    """
    A custom RoI box head that uses convolutions, matching the original model's architecture.
    """
    def __init__(self, in_channels, representation_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, representation_size, kernel_size=7, stride=1)
        self.bn1 = nn.BatchNorm2d(representation_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(representation_size, representation_size, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(representation_size)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x.flatten(start_dim=1)

def build_full_tf_to_pytorch_map():
    """
    Builds a highly detailed map to connect TensorFlow/Keras layer names to
    their PyTorch counterparts, including the Feature Pyramid Network (FPN).
    """
    mapping = {}
    # ResNet Backbone (C2-C5)
    for i in range(2, 6):
        pt_layer = f'layer{i-1}'
        block_counts = {2: 3, 3: 4, 4: 6, 5: 3}[i]
        for j in range(block_counts):
            block_char = chr(ord('a') + j)
            pt_block = f'backbone.body.{pt_layer}.{j}'
            if j == 0:
                mapping[f'res{i}{block_char}_branch1'] = f'{pt_block}.downsample.0.weight'
                mapping[f'bn{i}{block_char}_branch1'] = f'{pt_block}.downsample.1'
            mapping[f'res{i}{block_char}_branch2a'] = f'{pt_block}.conv1.weight'
            mapping[f'bn{i}{block_char}_branch2a'] = f'{pt_block}.bn1'
            mapping[f'res{i}{block_char}_branch2b'] = f'{pt_block}.conv2.weight'
            mapping[f'bn{i}{block_char}_branch2b'] = f'{pt_block}.bn2'
            mapping[f'res{i}{block_char}_branch2c'] = f'{pt_block}.conv3.weight'
            mapping[f'bn{i}{block_char}_branch2c'] = f'{pt_block}.bn3'

    # Feature Pyramid Network (FPN)
    for i in range(2, 6):
        mapping[f'fpn_c{i}p{i}'] = f'backbone.fpn.inner_blocks.{i-2}.weight'
        mapping[f'fpn_p{i}'] = f'backbone.fpn.layer_blocks.{i-2}.weight'

    # RPN (Region Proposal Network)
    mapping['rpn_conv_shared'] = 'rpn.head.conv.0.0.weight'
    mapping['rpn_class_raw'] = 'rpn.head.cls_logits.weight'
    mapping['rpn_bbox_pred'] = 'rpn.head.bbox_pred.weight'

    # RoI Heads (Box and Mask)
    mapping['mrcnn_class_conv1'] = 'roi_heads.box_head.conv1.weight'
    mapping['mrcnn_class_bn1'] = 'roi_heads.box_head.bn1'
    mapping['mrcnn_class_conv2'] = 'roi_heads.box_head.conv2.weight'
    mapping['mrcnn_class_bn2'] = 'roi_heads.box_head.bn2'
    mapping['mrcnn_class_logits'] = 'roi_heads.box_predictor.cls_score.weight'
    mapping['mrcnn_bbox_fc'] = 'roi_heads.box_predictor.bbox_pred.weight'
    for i in range(1, 5):
        mapping[f'mrcnn_mask_conv{i}'] = f'roi_heads.mask_head.0.{(i-1)*2}.weight'
        mapping[f'mrcnn_mask_bn{i}'] = f'roi_heads.mask_head.0.{(i-1)*2+1}'
    mapping['mrcnn_mask_deconv'] = 'roi_heads.mask_head.1.weight'
    mapping['mrcnn_mask'] = 'roi_heads.mask_predictor.mask_fcn_logits.weight'
    return mapping

def convert_and_load_weights(tf_weights_path, pytorch_model):
    """Loads and converts weights from the .h5 file into the PyTorch model."""
    print("🧠 Starting TensorFlow -> PyTorch weight conversion...")
    hf = h5py.File(tf_weights_path, 'r')
    tf_to_pt_map = build_full_tf_to_pytorch_map()
    new_state_dict = OrderedDict()

    for tf_name, pt_name in tf_to_pt_map.items():
        try:
            if 'kernel:0' in hf[tf_name]:
                kernel = torch.from_numpy(hf[tf_name]['kernel:0'][()]).permute(3, 2, 0, 1).contiguous()
                new_state_dict[pt_name] = kernel
                if 'bias:0' in hf[tf_name]:
                    bias = torch.from_numpy(hf[tf_name]['bias:0'][()])
                    new_state_dict[pt_name.replace('weight', 'bias')] = bias
            elif 'moving_mean:0' in hf[tf_name]:
                new_state_dict[f'{pt_name}.running_mean'] = torch.from_numpy(hf[tf_name]['moving_mean:0'][()])
                new_state_dict[f'{pt_name}.running_var'] = torch.from_numpy(hf[tf_name]['moving_variance:0'][()])
                new_state_dict[f'{pt_name}.weight'] = torch.from_numpy(hf[tf_name]['gamma:0'][()])
                new_state_dict[f'{pt_name}.bias'] = torch.from_numpy(hf[tf_name]['beta:0'][()])
        except KeyError:
            warnings.warn(f"Could not find TensorFlow layer: {tf_name}")

    missing, unexpected = pytorch_model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Weight conversion complete. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if unexpected or len(missing) > 10:
        warnings.warn(f"Potential conversion issues. Missing: {missing}, Unexpected: {unexpected}")
    hf.close()

# --- 3. Main Test Script ---

def mrcnn_pytorch(model, img, size_range, confidence_threshold, device):
    """Performs inference using the fully converted PyTorch model."""
    model.to(device)
    model.eval()
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    image_tensor = F.to_tensor(img).to(device)
    
    with torch.no_grad():
        results = model([image_tensor])
    r = results[0]
    scores = r['scores'].cpu().numpy()
    masks = r['masks'].cpu().numpy()
    selection = scores > confidence_threshold
    masks = masks[selection]
    masks_squeezed = masks.squeeze(axis=1)
    mask_areas = masks_squeezed.sum(axis=(1, 2))
    size_selection = (mask_areas > size_range[0] ** 2) & (mask_areas < size_range[1] ** 2)
    final_masks = masks_squeezed[size_selection]
    ROIs = final_masks > 0.5
    print(f"Detected {len(ROIs)} ROIs matching the criteria.")
    return ROIs

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
    
    # with tf.device(DEVICE):
    #    model = modellib.MaskRCNN(mode="inference", model_dir=model_dir,
    #                              config=config)
    #    tf.keras.Model.load_weights(model.keras_model, weights_path, by_name=True)
    #    results = model.detect([img], verbose=1)
    #    r = results[0]
    #    selection = np.logical_and(r['masks'].sum(axis=(0,1)) > size_range[0] ** 2, 
    #                               r['masks'].sum(axis=(0,1)) < size_range[1] ** 2)
    #    r['masks'] = r['masks'][:, :, selection]
    #    ROIs = r['masks'].transpose([2, 0, 1])   
    # return ROIs

def test_mrcnn():
    weights_path = download_model('mask_rcnn') 
    print("weights_path", weights_path)  
    summary_images = cm.load(download_demo('demo_voltage_imaging_summary_images.tif'))
    ROIs = mrcnn(img=summary_images.transpose([1, 2, 0]), size_range=[5, 22], 
            weights_path=weights_path)
    assert ROIs.shape[0] == 14, 'fail to infer correct number of neurons'
    
# def print_h5_structure(group, indent=""):
#    """Recursively prints the structure of an HDF5 group."""
#    for key in group.keys():
#        item = group[key]
#        if isinstance(item, h5py.Dataset):
#            print(f"{indent}Dataset: {key} (Shape: {item.shape})")
#        elif isinstance(item, h5py.Group):
#            print(f"{indent}Group: {key}")
#            print_h5_structure(item, indent + "  ")

if __name__ == "__main__":
    print("\n🚀 Starting PyTorch Mask R-CNN test with final corrections...")
    weights_path = download_model('mask_rcnn')
    summary_images = cm.load(download_demo('demo_voltage_imaging_summary_images.tif'))
    img_data = summary_images.transpose([1, 2, 0])

    backbone = resnet_fpn_backbone('resnet50', weights=None, trainable_layers=3)
    num_classes = 2  # neuron + background
    
    # FIX: Initialize the head with the correct arguments, removing 'resolution'.
    box_head = ConvBoxHead(in_channels=backbone.out_channels, representation_size=1024)

    model = MaskRCNN(backbone, num_classes=num_classes, box_head=box_head)
    
    convert_and_load_weights(weights_path, model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on device: {device}")
    
    ROIs = mrcnn_pytorch(model, img_data, size_range=[5, 22], confidence_threshold=0.7, device=device)

    print(f"Final check: Assertion requires 14 neurons. Found {ROIs.shape[0]}.")
    assert ROIs.shape[0] == 14, 'Failed to infer the correct number of neurons!'
    print("\n🎉 Test finished successfully! Correct number of ROIs detected.")
    
    