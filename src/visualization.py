import numpy as np
import cv2
import os
import random
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
import json


def show_image_mask_prediction(image: np.ndarray, true_mask: np.ndarray = None, pred_mask: np.ndarray = None, labelmap_path: str = None, cmap_name='tab20'):
    """
    Displays the input image alongside true and/or predicted masks using a consistent colormap.

    Args:
        image: RGB image (H, W, 3)
        true_mask: (H, W) integer mask or None
        pred_mask: (H, W) integer mask or None
        labelmap_path: path to the label map, which is a dict mapping class index to class name
        cmap_name: matplotlib colormap name

    Returns:
        None
    """
    assert os.path.exists(labelmap_path)
    with open(labelmap_path, "r") as f:
        labelmap = json.load(f)

    assert image.ndim == 3, "Image must be (H, W, 3)"
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    
    num_classes = len(labelmap) if labelmap else max(
        np.max(true_mask) if true_mask is not None else 0,
        np.max(pred_mask) if pred_mask is not None else 0,
    ) + 1
    
    colors = [cmap(i)[:3] for i in range(num_classes)]
    colors_255 = [(np.array(c) * 255).astype(np.uint8) for c in colors]

    def colorize(mask):
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(num_classes):
            color_mask[mask == i] = colors_255[i]
        return color_mask

    display_list = [image]
    titles = ["Input Image"]

    if true_mask is not None:
        display_list.append(colorize(true_mask))
        titles.append("True Mask")

    if pred_mask is not None:
        display_list.append(colorize(pred_mask))
        titles.append("Predicted Mask")

    plt.figure(figsize=(5 * len(display_list), 5))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.imshow(display_list[i])
        plt.title(titles[i])
        plt.axis('off')
    
    # Optional legend
    if labelmap:
        legend_elements = [matplotlib.patches.Patch(
            facecolor=colors[i], label=labelmap[str(i)]) for i in range(num_classes)]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                   borderaxespad=0., title="Classes")

    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    plt.show()


def count_pixels(mask_paths: List[str], labelmap:Dict[str, str], show:bool=True) -> defaultdict:
    """
    Count then number of pixels belonging to each class.
    Args:
        mask_paths (list[str]): Paths to the dataset masks.
        labelmap (dict): Dictionary mapping class indices (as strings) to class names.
        show (bool): set to True to show the pixel count as bar plot.
    Returns:
        None.
    """
    count = defaultdict(int)
    for p in mask_paths:
        mask = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        for class_idx_str in labelmap.keys():
            class_idx = int(class_idx_str)
            count[class_idx] += np.sum(mask==class_idx)
    if show:
        classes = list(labelmap.values())
        values = list(count.values())
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].bar(classes, values)
        ax[0].tick_params(axis='x', labelrotation=45)
        ax[0].set_xlabel('Classes')
        ax[0].set_ylabel('Number of pixels')
        ax[0].set_title("Class count including background class")
        ax[1].bar(classes[1:], values[1:])
        ax[1].tick_params(axis='x', labelrotation=45)
        ax[1].set_title("Class count (Zoom on foreground classes)")
        plt.tight_layout()
        plt.show()

    return count

def show_mask(mask: np.ndarray, scale: float = 0.1) -> None:
    """
    Displays the mask using OpenCV.

    Args:
        mask (np.ndarray): The mask to display (2D integer array).
        scale (float): Factor to resize the image for display.
    """
    height, width = mask.shape
    resized = cv2.resize(mask, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_NEAREST)
    normalized = (resized * (255 // (mask.max() or 1))).astype(np.uint8)
    cv2.imshow('Mask', normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    labelmap_path = "/home/salwa/Documents/code/traffic_sign_seg/data/labelmap.json"
    image_dir = "/home/salwa/Downloads/Data4/images"
    mask_dir = "/home/salwa/Downloads/Data4/masks"
    image_paths = [os.path.join(image_dir, i) for i in os.listdir(image_dir)]
    mask_paths = [os.path.join(mask_dir, i.replace("jpg", "png")) for i in os.listdir(image_dir)]
    
    sample_idx = random.sample(list(range(len(image_paths))), k=1)[0]
    print(sample_idx)
    image = cv2.imread(image_paths[sample_idx])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_paths[sample_idx], cv2.IMREAD_UNCHANGED)
    print(f"{image.shape} {mask.shape}") 

    show_image_mask_prediction(image, true_mask=mask, pred_mask=None, labelmap_path=labelmap_path, cmap_name='tab20')