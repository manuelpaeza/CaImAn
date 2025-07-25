"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Revised by Eric Thompson, Changjia Cai, and Manuel Paez 
"""

import os
import random
import colorsys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, Rectangle

def apply_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[float, float, float] = (1, 0, 0), alpha: float = 0.5) -> np.ndarray:
    """
    Applies a single colored mask to an image with transparency.

    This function handles both integer (e.g., uint8) and float images.
    - For float images, color values are assumed to be in the [0, 1] range.
    - For integer images, color values are scaled to the [0, 255] range.

    Args:
        image (np.ndarray): The input RGB image of shape (H, W, 3).
        mask (np.ndarray): A boolean or integer mask of shape (H, W) where
                           non-zero values indicate the area to mask.
        color (Tuple[float, float, float]): The RGB color for the mask, with
                                            values normalized between 0 and 1.
        alpha (float): The opacity of the mask, from 0 (transparent) to 1 (opaque).

    Returns:
        np.ndarray: The image with the mask applied, with the same dtype as the input.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def apply_masks(image: np.ndarray, mask: np.ndarray, color: Tuple[float, float, float], alpha: float = 0.5) -> np.ndarray:
    """
    Applies a single mask to an image.

    Args:
        image (np.ndarray): The input RGB image (H, W, 3) with values from 0-1.
        mask (np.ndarray): A boolean mask (H, W).
        color (Tuple[float, float, float]): The RGB color for the mask, normalized to 0-1.
        alpha (float): The transparency of the mask overlay.

    Returns:
        np.ndarray: The image with the mask applied.
    """
    masked_image = image.copy()
    
    for mask_ind, mask in enumerate(data_masks):
        masked_image = apply_mask(masked_image, mask, color, alpha=alpha)
        
    return masked_image

def draw_box(box: np.ndarray, color: str = 'white', ax: plt.Axes = None, line_width: float = 0.5) -> Tuple[plt.Axes, patches.Rectangle]:
    """
    Draws a single rectangular bounding box on a given axes object.

    Args:
        box (np.ndarray): A 1x4 array representing a single bounding box
                          in [xmin, ymin, xmax, ymax] format.
        color (str): The matplotlib color for the box outline.
        ax (plt.Axes, optional): The pyplot Axes object upon which the rectangle
                                 will be drawn. If None, the current axes are used.
        line_width (float): The width of the box outline.

    Returns:
        A tuple containing:
        - ax (plt.Axes): The axes object.
        - rect (patches.Rectangle): The created matplotlib Rectangle object.
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

def draw_boxes(box: np.ndarray, color: str = 'white', ax=None, line_width: float = 0.5) -> Tuple[plt.Axes, patches.Rectangle]:
    """
    Draws a single bounding box on a given axes object.

    Args:
        box (np.ndarray): A 1x4 array representing a single bounding box
                          in [xmin, ymin, xmax, ymax] format.
        color (str): The color of the box outline.
        ax (plt.Axes): The matplotlib axes object to draw on.
        line_width (float): The width of the box outline.

    Returns:
        A tuple containing the axes object and the created Rectangle patch.
    """
    if ax is None:
        ax = pl.gca()

    num_boxes = len(boxes)
    all_rects = []
    for box in boxes:
        ax, rect = draw_box(box, color=color, ax=ax, line_width=line_width)
        all_rects.append(rect)
       
    return ax, all_rects

def plot_volpy_segs(image: np.ndarray,
    masks: List[Dict],
    min_v: float,
    max_v: float,
    outline_color: str,
    outline_width: float,
    figsize: Tuple[int, int] = (6, 10),
    title: str = None
    ):
    """
    Plots Volpy mask outlines on mean and correlation images.

    The function creates a 2x2 subplot showing the mean image and correlation
    image, both with and without the segmentation outlines.

    Args:
        image (np.ndarray): The input image data, expected to be a 3D array
                            where the 3rd dimension contains mean and correlation
                            images (e.g., image[:,:,1] is mean, image[:,:,2] is corr).
        masks (List[Dict]): A list of mask dictionaries. Each dictionary must
                            contain 'all_points_x' and 'all_points_y' keys
                            representing the vertices of a polygon outline.
        min_v (float): The minimum percentile for contrast scaling (e.g., 1).
        max_v (float): The maximum percentile for contrast scaling (e.g., 99).
        outline_color (str): The color of the mask outlines.
        outline_width (float): The line width of the mask outlines.
        figsize (Tuple[int, int]): The size of the figure.
        title (str, optional): An optional super-title for the entire plot.
    """
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=figsize, sharex=True, sharey=True)  

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
    Generate N visually distinct random colors.

    To achieve this, colors are generated evenly spaced in HSV space
    and then converted to the RGB color space.

    Args:
        N (int): The number of colors to generate.
        bright (bool): If True, generate bright colors. Otherwise, generate darker colors.

    Returns:
        A list of N colors, where each color is a tuple of (R, G, B) values
        normalized between 0 and 1.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
