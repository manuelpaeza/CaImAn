#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import warnings
from collections import OrderedDict

# Caiman library imports
import caiman as cm
from caiman.utils.utils import download_model, download_demo
from torchvision.ops import nms, roi_align

# ###########################################################
# ## 1. FULL PYTORCH RE-IMPLEMENTATION OF THE CAIMAN MRCNN MODEL ##
# ###########################################################

class BatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        return super().forward(x)

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=1, stride=stride, bias=False)
        self.bn1 = BatchNorm(out_channels[0])
        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm(out_channels[1])
        self.conv3 = nn.Conv2d(out_channels[1], out_channels[2], kernel_size=1, bias=False)
        self.bn3 = BatchNorm(out_channels[2])
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if use_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels[2], kernel_size=1, stride=stride, bias=False),
                BatchNorm(out_channels[2])
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ResNet-101 Structure
        self.layer1 = self._make_layer(64, [64, 64, 256], 3)
        self.layer2 = self._make_layer(256, [128, 128, 512], 4, stride=2)
        self.layer3 = self._make_layer(512, [256, 256, 1024], 23, stride=2)
        self.layer4 = self._make_layer(1024, [512, 512, 2048], 3, stride=2)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [ResNetBlock(in_channels, out_channels, stride, use_downsample=True)]
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels[2], out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]

class FPN(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        self.c2_conv = nn.Conv2d(256, out_channels, kernel_size=1)
        self.c3_conv = nn.Conv2d(512, out_channels, kernel_size=1)
        self.c4_conv = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.c5_conv = nn.Conv2d(2048, out_channels, kernel_size=1)
        self.p2_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.p3_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.p4_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.p5_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.p6_maxpool = nn.MaxPool2d(kernel_size=1, stride=2)

    def forward(self, features):
        c2, c3, c4, c5 = features
        p5 = self.c5_conv(c5)
        p4 = F.interpolate(p5, scale_factor=2) + self.c4_conv(c4)
        p3 = F.interpolate(p4, scale_factor=2) + self.c3_conv(c3)
        p2 = F.interpolate(p3, scale_factor=2) + self.c2_conv(c2)
        p2, p3, p4, p5 = self.p2_conv(p2), self.p3_conv(p3), self.p4_conv(p4), self.p5_conv(p5)
        p6 = self.p6_maxpool(p5)
        return [p2, p3, p4, p5, p6]

class RPN(nn.Module):
    def __init__(self, in_channels, anchors_per_location):
        super().__init__()
        self.shared_conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.cls_logits = nn.Conv2d(512, 2 * anchors_per_location, kernel_size=1)
        self.bbox_pred = nn.Conv2d(512, 4 * anchors_per_location, kernel_size=1)

    def forward(self, x):
        shared = self.relu(self.shared_conv(x))
        return self.cls_logits(shared), self.bbox_pred(shared)

class ProposalLayer(nn.Module):
    def __init__(self, proposal_count, nms_threshold):
        super().__init__()
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def forward(self, rpn_probs, rpn_deltas, anchors):
        scores = rpn_probs[:, :, 1]
        pre_nms_limit = min(6000, anchors.shape[0])
        scores, order = scores.sort(descending=True)
        order, scores = order[:pre_nms_limit], scores[:pre_nms_limit]
        anchors, rpn_deltas = anchors[order, :], rpn_deltas[:, order, :]
        
        box_cy, box_cx = (anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2
        box_h, box_w = anchors[:, 2] - anchors[:, 0], anchors[:, 3] - anchors[:, 1]
        delta_cy, delta_cx = rpn_deltas[0, :, 0] * box_h, rpn_deltas[0, :, 1] * box_w
        delta_h, delta_w = torch.exp(rpn_deltas[0, :, 2]) * box_h, torch.exp(rpn_deltas[0, :, 3]) * box_w

        pred_boxes = torch.stack([pred_cy - delta_h / 2, pred_cx - delta_w / 2,
                                  pred_cy + delta_h / 2, pred_cx + delta_w / 2], dim=1)
        
        torchvision_boxes = pred_boxes[:, [1, 0, 3, 2]]
        keep = nms(torchvision_boxes, scores, self.nms_threshold)
        final_boxes = pred_boxes[keep[:self.proposal_count], :]
        return torch.clamp(final_boxes, 0, 1)

class MRCNNHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 1024, kernel_size=7)
        self.bn1 = BatchNorm(1024)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.bn2 = BatchNorm(1024)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(1024, num_classes)
        self.bbox_regressor = nn.Linear(1024, num_classes * 4)

        self.mask_conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.mask_bn1 = BatchNorm(256)
        self.mask_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.mask_bn2 = BatchNorm(256)
        self.mask_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.mask_bn3 = BatchNorm(256)
        self.mask_conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.mask_bn4 = BatchNorm(256)
        self.mask_deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.mask_logits = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        cls_x = self.relu(self.bn1(self.conv1(x)))
        cls_x = self.relu(self.bn2(self.conv2(cls_x)))
        cls_x = cls_x.view(cls_x.size(0), -1)
        cls_logits, bbox_deltas = self.classifier(cls_x), self.bbox_regressor(cls_x)
        
        mask_x = self.relu(self.mask_bn1(self.mask_conv1(x)))
        mask_x = self.relu(self.mask_bn2(self.mask_conv2(mask_x)))
        mask_x = self.relu(self.mask_bn3(self.mask_conv3(mask_x)))
        mask_x = self.relu(self.mask_bn4(self.mask_conv4(mask_x)))
        mask_x = self.relu(self.mask_deconv(mask_x))
        mask_logits = self.mask_logits(mask_x)
        return cls_logits, bbox_deltas, mask_logits

class MRCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNetBackbone()
        self.fpn = FPN()
        self.rpn = RPN(in_channels=256, anchors_per_location=3)
        self.proposal_layer = ProposalLayer(proposal_count=1000, nms_threshold=0.7)
        self.head = MRCNNHead(in_channels=256, num_classes=2)

    def generate_anchors(self, scales, ratios, shape, feature_stride, anchor_stride):
        scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
        scales, ratios = scales.flatten(), ratios.flatten()
        heights, widths = scales / np.sqrt(ratios), scales * np.sqrt(ratios)
        shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
        shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
        shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
        box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
        box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
        box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])
        boxes = np.concatenate([box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1)
        return boxes

    def get_all_anchors(self, image_shape):
        backbone_shapes = np.array([[int(np.ceil(image_shape[0] / s)), int(np.ceil(image_shape[1] / s))] for s in [4, 8, 16, 32, 64]])
        all_anchors = [self.generate_anchors((s,), [0.5, 1, 2], backbone_shapes[i], [4, 8, 16, 32, 64][i], 1) for i, s in enumerate((32, 64, 128, 256, 512))]
        all_anchors = np.concatenate(all_anchors, axis=0)
        all_anchors /= np.array([image_shape[0], image_shape[1], image_shape[0], image_shape[1]])
        return torch.from_numpy(all_anchors).float()

    def forward(self, x):
        img_shape = x.shape[2:]
        p_features = self.fpn(self.backbone(x))
        rpn_logits, rpn_deltas = zip(*[self.rpn(p) for p in p_features])
        rpn_logits = torch.cat([p.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 2) for p in rpn_logits], dim=1)
        rpn_deltas = torch.cat([p.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4) for p in rpn_deltas], dim=1)
        proposals = self.proposal_layer(F.softmax(rpn_logits, dim=-1), rpn_deltas, self.get_all_anchors(img_shape).to(x.device))
        roi_features = roi_align(p_features[:4], [proposals[:, [1, 0, 3, 2]]], output_size=7)
        cls_logits, bbox_deltas, mask_logits = self.head(roi_features)
        return [dict(rois=proposals, class_logits=cls_logits, class_probs=F.softmax(cls_logits, dim=-1), bbox_deltas=bbox_deltas, masks=mask_logits)]

# ###########################################################
# ## 2. WEIGHT CONVERSION AND MAIN TEST SCRIPT ##
# ###########################################################

def build_weight_map():
    mapping = {}
    mapping['conv1'] = 'backbone.conv1.weight'; mapping['bn_conv1'] = 'backbone.bn1'
    for i, blocks in enumerate([3, 4, 23, 3]):
        for j in range(blocks):
            block, stage = chr(ord('a') + j), i + 2
            prefix_tf = f'res{stage}{block}'; prefix_pt = f'backbone.layer{i+1}.{j}'
            mapping[f'{prefix_tf}_branch2a'] = f'{prefix_pt}.conv1.weight'; mapping[f'bn{stage}{block}_branch2a'] = f'{prefix_pt}.bn1'
            mapping[f'{prefix_tf}_branch2b'] = f'{prefix_pt}.conv2.weight'; mapping[f'bn{stage}{block}_branch2b'] = f'{prefix_pt}.bn2'
            mapping[f'{prefix_tf}_branch2c'] = f'{prefix_pt}.conv3.weight'; mapping[f'bn{stage}{block}_branch2c'] = f'{prefix_pt}.bn3'
            if j == 0:
                mapping[f'{prefix_tf}_branch1'] = f'{prefix_pt}.downsample.0.weight'; mapping[f'bn{stage}{block}_branch1'] = f'{prefix_pt}.downsample.1'
    for i in range(2, 6):
        mapping[f'fpn_c{i}p{i}'] = f'fpn.c{i}_conv.weight'; mapping[f'fpn_p{i}'] = f'fpn.p{i}_conv.weight'
    mapping['rpn_conv_shared'] = 'rpn.shared_conv.weight'; mapping['rpn_class_raw'] = 'rpn.cls_logits.weight'; mapping['rpn_bbox_pred'] = 'rpn.bbox_pred.weight'
    mapping['mrcnn_class_conv1'] = 'head.conv1.weight'; mapping['mrcnn_class_bn1'] = 'head.bn1'
    mapping['mrcnn_class_conv2'] = 'head.conv2.weight'; mapping['mrcnn_class_bn2'] = 'head.bn2'
    mapping['mrcnn_class_logits'] = 'head.classifier.weight'; mapping['mrcnn_bbox_fc'] = 'head.bbox_regressor.weight'
    mapping['mrcnn_mask_conv1'] = 'head.mask_conv1.weight'; mapping['mrcnn_mask_bn1'] = 'head.mask_bn1'
    mapping['mrcnn_mask_conv2'] = 'head.mask_conv2.weight'; mapping['mrcnn_mask_bn2'] = 'head.mask_bn2'
    mapping['mrcnn_mask_conv3'] = 'head.mask_conv3.weight'; mapping['mrcnn_mask_bn3'] = 'head.mask_bn3'
    mapping['mrcnn_mask_conv4'] = 'head.mask_conv4.weight'; mapping['mrcnn_mask_bn4'] = 'head.mask_bn4'
    mapping['mrcnn_mask_deconv'] = 'head.mask_deconv.weight'; mapping['mrcnn_mask'] = 'head.mask_logits.weight'
    return mapping

def convert_and_load_weights(tf_weights_path, pytorch_model):
    print("🧠 Starting final weight conversion...")
    hf = h5py.File(tf_weights_path, 'r')
    tf_to_pt_map = build_weight_map()
    new_state_dict = OrderedDict()
    rpn_group = hf['rpn_model'] # Correctly access the nested RPN group

    for tf_name, pt_name in tf_to_pt_map.items():
        try:
            # FIX: Explicitly handle the nested RPN group
            if tf_name in rpn_group:
                tf_layer_group = rpn_group[tf_name]
            else:
                tf_layer_group = hf[tf_name]

            if 'kernel:0' in tf_layer_group:
                kernel = torch.from_numpy(tf_layer_group['kernel:0'][()]).permute(3, 2, 0, 1).contiguous()
                new_state_dict[pt_name] = kernel
                if 'bias:0' in tf_layer_group:
                    new_state_dict[pt_name.replace('weight', 'bias')] = torch.from_numpy(tf_layer_group['bias:0'][()])
            elif 'moving_mean:0' in tf_layer_group: # BatchNorm
                new_state_dict[f'{pt_name}.running_mean'] = torch.from_numpy(tf_layer_group['moving_mean:0'][()])
                new_state_dict[f'{pt_name}.running_var'] = torch.from_numpy(tf_layer_group['moving_variance:0'][()])
                new_state_dict[f'{pt_name}.weight'] = torch.from_numpy(tf_layer_group['gamma:0'][()])
                new_state_dict[f'{pt_name}.bias'] = torch.from_numpy(tf_layer_group['beta:0'][()])
            elif 'weights:0' in tf_layer_group: # Dense layers
                weights = torch.from_numpy(tf_layer_group['weights:0'][()]).T
                new_state_dict[pt_name] = weights
                if 'bias:0' in tf_layer_group:
                    new_state_dict[pt_name.replace('weight', 'bias')] = torch.from_numpy(tf_layer_group['bias:0'][()])
        except KeyError:
             warnings.warn(f"Could not find or process TF layer group: {tf_name}")

    pytorch_model.load_state_dict(new_state_dict, strict=False)
    print("✅ Weight conversion complete.")
    hf.close()

def mrcnn_pytorch(model, img, size_range, confidence_threshold, device):
    model.to(device); model.eval()
    
    # FIX: Correctly handle multi-channel summary images first
    if img.ndim == 3 and img.shape[2] > 3:
        img = np.mean(img, axis=2) # Project to grayscale
    
    original_shape = img.shape
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    scale = 512 / max(original_shape[:2])
    resized_shape = (int(original_shape[0] * scale), int(original_shape[1] * scale))
    # Ensure caiman movie is created from the 3-channel image
    img_resized = np.array(cm.movie(img.astype(np.float32)).resize(fx=scale, fy=scale))
    
    padded_img = np.zeros((512, 512, 3), dtype=np.float32)
    padded_img[:resized_shape[0], :resized_shape[1], :] = img_resized
    image_tensor = torch.from_numpy(padded_img).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        results = model(image_tensor)[0]
    
    scores, rois, masks = results['class_probs'][:, 1], results['rois'], results['masks']
    keep = scores > confidence_threshold
    scores, rois, masks = scores[keep], rois[keep], masks[keep]

    final_masks = []
    for i in range(masks.shape[0]):
        y1, x1, y2, x2 = (rois[i] * 512).int()
        if y1 >= y2 or x1 >= x2: continue
        roi_mask = F.interpolate(masks[i].unsqueeze(0), size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False).squeeze(0)
        full_mask = np.zeros(resized_shape[:2], dtype=bool)
        full_mask[y1:y2, x1:x2] = (roi_mask[1] > 0.5).cpu().numpy()
        final_mask = np.array(cm.movie(full_mask.astype(np.float32)).resize(1/scale, 1/scale, interpolation='bilinear')) > 0.5
        final_masks.append(final_mask)
    
    if not final_masks: return np.array([])
    final_masks = np.stack(final_masks, axis=0)

    mask_areas = final_masks.sum(axis=(1, 2))
    size_selection = (mask_areas > size_range[0] ** 2) & (mask_areas < size_range[1] ** 2)
    final_masks = final_masks[size_selection]

    print(f"Detected {len(final_masks)} ROIs matching the criteria.")
    return final_masks

def test_mrcnn_pytorch():
    print("\n🚀 Starting test with full architectural reimplementation...")
    weights_path = download_model('mask_rcnn')
    summary_images = cm.load(download_demo('demo_voltage_imaging_summary_images.tif'))
    img_data = summary_images.transpose([1, 2, 0])

    model = MRCNNModel()
    convert_and_load_weights(weights_path, model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on device: {device}")
    
    ROIs = mrcnn_pytorch(model, img_data, size_range=[5, 22], confidence_threshold=0.95, device=device)

    print(f"Final check: Assertion requires 14 neurons. Found {ROIs.shape[0]}.")
    assert ROIs.shape[0] == 14, 'Failed to infer the correct number of neurons!'
    print("\n🎉 Test finished successfully! Correct number of ROIs detected.")

if __name__ == '__main__':
    test_mrcnn_pytorch()