import json
import numpy as np
import cv2
import os
from typing import List, Dict, Union, Tuple
from visualization import show_mask


def polygon_to_mask(img_shape: Tuple[int], points:Union[List[int], Tuple[int]]):
    """
    Converts polygon points into a binary mask.

    Args:
        img_shape (tuple): Shape of the output mask (height, width).
        points (list of tuple): List of (x, y) coordinates defining the polygon.

    Returns:
        np.ndarray: Boolean array of shape (height, width) with True inside the polygon.
    """
    mask = np.zeros(img_shape, dtype=np.uint8)
    points_array = np.array([points], dtype=np.int32)
    cv2.fillPoly(mask, [points_array], 1)
    return mask.astype(bool)

def create_mask_with_edges(json_path: str, labelmap: Dict[str,str], contour_width: int=0, contour_class_idx:int=None):
    """
    Creates a labeled mask from a LabelMe JSON file, with optional edge (contour) regions.

    Args:
        json_path (str): Path to the LabelMe-format JSON annotation file.
        labelmap (dict): Dictionary mapping class indices (as strings) to class names.
        contour_width (int, optional): Width of the edge (dilated contour) to include in the mask.
                                       If 0 or None, no edge is added.
        contour_class_idx (int): Index to assign to the contour class. Set to an integer 
                                or leave None if you want to set it automatically.
    Returns:
        np.ndarray: A 2D array with class indices assigned to regions and optionally edge class.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    img_shape = (data["imageHeight"], data["imageWidth"])

    full_mask = np.zeros(img_shape, dtype=np.uint8)
    edge_mask = np.zeros(img_shape, dtype=bool)

    name_to_index = {v: int(k) for k, v in labelmap.items()}

    for shape in data['shapes']:
        label = shape['label']
        if label not in labelmap.values():
            print(f"{label} is not in the labelmap!")
            continue
        class_index = name_to_index[label]
        points = [(int(x), int(y)) for x, y in shape['points']]
        obj_mask = polygon_to_mask(img_shape, points)

        full_mask[obj_mask & (full_mask == 0)] = class_index

        # Optional contour generation
        if contour_width and contour_width > 0:
            obj_mask_uint8 = obj_mask.astype(np.uint8)
            kernel = np.ones((contour_width * 2 + 1, contour_width * 2 + 1), np.uint8)
            dilated = cv2.dilate(obj_mask_uint8, kernel)
            edge_region = (dilated.astype(bool)) & ~obj_mask
            edge_mask |= edge_region & (full_mask == 0)

    if contour_width and contour_width > 0:
        if contour_class_idx is None:
            contour_class_idx = max([int(k) for k in labelmap.keys()]) + 1
        full_mask[edge_mask] = contour_class_idx

    return full_mask


def save_mask(mask: np.ndarray, save_path: str) -> None:
    """
    Saves the mask as an image.

    Args:
        mask (np.ndarray): The mask to save.
        save_path (str): File path where the mask should be saved.
    """
    cv2.imwrite(save_path, mask)

def generate_masks_from_folder(input_folder: str, output_folder: str, labelmap_path: str, show: bool, contour_width: int = 0, contour_class_idx:int=None) -> None:
    """
    Generates segmentation masks from all LabelMe JSON annotation files in a folder.

    Args:
        input_folder (str): Path to the folder containing LabelMe JSON annotation files.
        output_folder (str): Path to the folder where generated mask images will be saved.
        labelmap_path (str): Path to a JSON file mapping class indices (as strings) to class names.
        show (bool): If True, displays each generated mask resized for visualization.
        contour_width (int): Optional width of the contour to include as edge class.

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)
    with open(labelmap_path, 'r') as file:
        labelmap = json.load(file)

    annotation_filenames = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    for filename in annotation_filenames:
        print(filename)
        json_path = os.path.join(input_folder, filename)
        mask = create_mask_with_edges(json_path, labelmap, contour_width, contour_class_idx)
        save_path = os.path.join(output_folder, filename.replace(".json", ".png"))
        save_mask(mask, save_path)

        if show:
            show_mask(mask)
