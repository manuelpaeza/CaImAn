"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Revised by Eric Thompson, Chanjia Cai, and Manuel Paez 
"""

import os
import random
import colorsys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

# Root directory of the project
# ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
# from ..mrcnn import utils

def apply_mask(image, mask, color=(1,0,0), alpha=0.5):
    """
    Apply the given mask to the image. Alpha is opacity, from 0 (transparent) to 1 (opaque)

    From mrcnn
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def apply_masks(image, data_masks, color=(1,0, 0), alpha=0.5):
    """
    apply many masks (N x H x W) to given image
    adapted from mrcnn
    """
    masked_image = image.copy()
    
    for mask_ind, mask in enumerate(data_masks):
        masked_image = apply_mask(masked_image, mask, color, alpha=alpha)
        
    return masked_image

def draw_box(box, color='white', ax=None, line_width=0.5):
    """
    Draw a single rectangular bounding box on given axes object.
    
    Args:
        bbox: xmin, ymin, xmax, ymax
        color: matplotlib color
        alpha : float opaqueness level (0. to 1., where 1 is opaque), default 0.2
        ax : pyplot.Axes object axes object upon which rectangle will be drawn, default None
    
    Returns:
        ax: pyplot.Axes object
        rect: matplotlib Rectangle object
    """
    
    if ax is None:
        ax = pl.gca()
        
    box_origin = (box[0], box[1])
    box_height = box[3] - box[1] 
    box_width = box[2] - box[0]

    rect = Rectangle(box_origin, 
                     width=box_width, 
                     height=box_height,
                     color=color, 
                     alpha=1,
                     fill=None,
                     linewidth=line_width)
    ax.add_patch(rect)

    return ax, rect

def draw_boxes(boxes, color='white', ax=None, line_width=0.5):
    """
    given Nx4 bounding boxes, draw them all on given axes object

    Returns axes object and list of rects
    """
    if ax is None:
        ax = pl.gca()

    num_boxes = len(boxes)
    all_rects = []
    for box in boxes:
        ax, rect = draw_box(box, color=color, ax=ax, line_width=line_width)
        all_rects.append(rect)
        
    return ax, all_rects

def plot_volpy_segs(image, masks, min_v, max_v, outline_color, outline_width, figsize=(6,10), title=None):
    """
    plot volpy mask outlines

    image from volpy is mean, mean, corr
    """
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=figsize, sharex=True, sharey=True)  # w/h

    # Mean
    ax1.imshow(image[:,:,1], cmap='gray', 
               vmin=np.percentile(image[:,:,1], min_v), 
               vmax=np.percentile(image[:,:,1], max_v));
    ax1.set_title('Mean Image')
    ax2.imshow(image[:,:,1], cmap='gray', 
               vmin=np.percentile(image[:,:,1], min_v), 
               vmax=np.percentile(image[:,:,1], max_v));
    for mask in masks:
        ax2.plot(mask['all_points_x'], 
                 mask['all_points_y'], 
                 color=outline_color, 
                 linewidth=outline_width);
    ax2.set_title('Mean Image Seg')
    
    # Corr
    ax3.imshow(image[:,:,2], cmap='gray', 
               vmin=np.percentile(image[:,:,2], min_v), 
               vmax=np.percentile(image[:,:,2], max_v));
    ax3.set_title('Corr Image')
    ax4.imshow(image[:,:,2], cmap='gray', 
               vmin=np.percentile(image[:,:,2], min_v), 
               vmax=np.percentile(image[:,:,2], max_v));
    for mask in masks:
        ax4.plot(mask['all_points_x'], 
                 mask['all_points_y'], 
                 color=outline_color, 
                 linewidth=outline_width);
    ax4.set_title('Corr Image Seg')

    if title is not None:
        plt.suptitle(title, y=0.99, fontsize=16);
        
    plt.tight_layout()

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.

    from mrcnn
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
