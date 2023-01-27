import os
import shutil
from pathlib import Path
import json
from datetime import datetime
from typing import Optional, List, Sequence
import zipfile
import tempfile
from queue import Queue, Empty
from threading import Thread

import pytz
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage import measure

import boto3
from botocore import UNSIGNED
from botocore.client import Config

from .constants import CLASS_LABELS, COLORS, TINTS, DEFAULT_IMAGE_DIM

ARTHUR_BUCKET = 's3-bucket-arthur-public'
BASE_FOLDER = Path(__file__).parent.parent
API_DATA_FOLDER = BASE_FOLDER / "api-data"
DATA_DOWNLOAD_FOLDER = BASE_FOLDER / "api-data-download"
REFERENCE_DATA_FOLDER = BASE_FOLDER / "reference-data"
TRAIN_DATA_FOLDER = BASE_FOLDER / "training" / "data"
MODEL_FOLDER = BASE_FOLDER / "model"
MODEL_WEIGHTS_PATH = MODEL_FOLDER / "model_weights.pt"

AI4MARS_DATA_DIR = TRAIN_DATA_FOLDER / "ai4mars-dataset-merged-0.1/msl"
IMAGES_DIR = AI4MARS_DATA_DIR / "images/edr"
TRAIN_LABELS_DIR = AI4MARS_DATA_DIR / "labels/train"
VAL_LABELS_DIR = AI4MARS_DATA_DIR / "labels/test/masked-gold-min3-100agree"

s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))


def image_ids_in_folder(folder=API_DATA_FOLDER) -> List[str]:
    return [sf for sf in os.listdir(folder) if os.path.isdir(folder / sf) and not sf.startswith(".")]


def download_inference_dataset(skip_if_images_present=True):
    # if there are any images already in the api data folder, skip download by default
    if len(image_ids_in_folder(API_DATA_FOLDER)) > 0 and skip_if_images_present:
        return
    else:
        print("inference dataset folder not found, downloading...")
        # download images to api-data folder
        with tempfile.NamedTemporaryFile() as tmp:
            s3_client.download_file(ARTHUR_BUCKET, "mars/api-inference-data.zip", tmp.name)
            with zipfile.ZipFile(tmp.name, 'r') as zipref:
                image_ids = {x.split("/")[0] for x in zipref.namelist()}
                zipref.extractall(DATA_DOWNLOAD_FOLDER)
        for image_id in image_ids:
            if (not os.path.exists(API_DATA_FOLDER / image_id)) or (len(os.listdir(API_DATA_FOLDER / image_id)) == 0):
                shutil.copytree(DATA_DOWNLOAD_FOLDER / image_id, API_DATA_FOLDER / image_id)


def download_reference_data(skip_exists=True):
    if "images" in os.listdir(REFERENCE_DATA_FOLDER) and skip_exists:
        return
    else:
        print("reference dataset not found, downloading...")
        with tempfile.NamedTemporaryFile() as tmp:
            s3_client.download_file(ARTHUR_BUCKET, "mars/reference-data.zip", tmp.name)
            with zipfile.ZipFile(tmp.name, 'r') as zipref:
                zipref.extractall(REFERENCE_DATA_FOLDER)


def download_model(skip_exists=True):
    if (not os.path.isfile(MODEL_WEIGHTS_PATH)) or (not skip_exists):
        print("model weights not found, downloading...")
        s3_client.download_file(ARTHUR_BUCKET, 'mars/model_weights.pt', str(MODEL_WEIGHTS_PATH))


def load_inference_data(image_ids: Optional[List[str]] = None, folder=API_DATA_FOLDER) -> pd.DataFrame:
    subfolders = image_ids_in_folder(folder)
    print(f"found {len(subfolders)} image folders in {folder}")
    data = []
    for image_id in subfolders:
        # if an image IDs filter is applied skip this
        if (image_ids is not None) and (image_id not in image_ids):
            continue
        row = {'image_id': image_id,
               'image': str(folder / image_id / "image.jpg")}
        metadata_path = folder / image_id / "metadata.txt"
        with open(metadata_path, 'r') as f:
            metadata = json.loads(f.read())
        date = datetime.strptime(metadata['earth_date'], "%Y-%m-%d").replace(hour=12)
        row['date'] = pytz.utc.localize(date)
        row['martian_sol'] = metadata['sol']
        data.append(row)
    if len(data) > 0:
        print(f"returning {len(data)} images after filtering")
        return pd.DataFrame(data)
    else:
        print(f"no images after filtering, returning empty dataframe")
        return pd.DataFrame({'image_id': pd.Series(dtype=str), 'image': pd.Series(dtype=str),
                             'date': pd.Series(dtype="datetime64[ns, utc]"), 'martian_sol': pd.Series(dtype="Int64")})


def output_to_arthur_format(boxes: Sequence[Sequence]):
    formatted_boxes = []
    for box in boxes:
        leftx, topy, rightx, bottomy, confidence, class_idx = box
        formatted_boxes.append([int(class_idx), float(confidence), int(leftx), int(topy), int(rightx - leftx),
                                int(bottomy - topy)])
    return formatted_boxes


### Preprocessing & Training Helpers ###

def download_training_dataset(skip_if_folders_present=True):
    if "ai4mars-dataset-merged-0.1" in os.listdir(TRAIN_DATA_FOLDER) and skip_if_folders_present:
        return
    else:
        print("ai4mars dataset not found, downloading...")
        # download training package to train data folder
        with tempfile.NamedTemporaryFile() as tmp:
            s3_client.download_file(ARTHUR_BUCKET, "mars/ai4mars-dataset-merged-0.1.zip", tmp.name)
            with zipfile.ZipFile(tmp.name, 'r') as zipref:
                zipref.extractall(TRAIN_DATA_FOLDER)


try:
    train_label_files = sorted(os.listdir(TRAIN_LABELS_DIR))
    train_image_files = [x.split(".")[0] + ".JPG" for x in train_label_files]
except FileNotFoundError:
    train_label_files = None
    train_image_files = None

try:
    val_label_files = sorted(os.listdir(VAL_LABELS_DIR))
    val_image_files = [x.replace("_merged.png", ".JPG") for x in val_label_files]
except FileNotFoundError:
    val_label_files = None
    val_image_files = None


# DATA PROCESSING

def separate_masks(mask):
    """
    Take in a single mask where classes are represented by (0, 0, 0) ... (3, 3, 3) pixels
    and split it into four separate masks represented by 0's or 1's
    """
    soil_mask = (mask == 0).astype(int)
    bedrock_mask = (mask == 1).astype(int)
    sand_mask = (mask == 2).astype(int)
    bigrock_mask = (mask == 3).astype(int)
    return soil_mask, bedrock_mask, sand_mask, bigrock_mask


def tint_image(image, mask):
    masks = separate_masks(mask)
    color_mask = np.zeros(mask.shape, dtype=np.int64)
    for i, cur_mask in enumerate(masks):
        mask_class = CLASS_LABELS[i]
        color_mask += cur_mask * TINTS[mask_class]

    return np.clip(image + color_mask, a_min=0, a_max=255)


def plot_color_key():
    fig, axes = plt.subplots(ncols=4, figsize=(12, 3), sharex='all', sharey='all')
    for j, axis in enumerate(axes):
        class_label = CLASS_LABELS[j]
        color = COLORS[class_label]
        image = np.zeros(shape=(1, 1, 3), dtype=np.int64) + color
        axis.imshow(image)
        axis.set_title(class_label)


def plot_images_with_mask_and_tint(indices):
    fig, axes = plt.subplots(ncols=3, nrows=len(indices), figsize=(12, 4 * len(indices)), sharex='all', sharey='all')

    for i, idx in enumerate(indices):
        image = cv2.imread(os.path.join(IMAGES_DIR, train_image_files[idx]))
        mask = cv2.imread(os.path.join(TRAIN_LABELS_DIR, train_label_files[idx]))

        if len(indices) == 0:
            rowax = axes
        else:
            rowax = axes[i]

        rowax[0].imshow(image)
        rowax[1].imshow(mask)
        rowax[2].imshow(tint_image(image, mask))


def plot_image_with_mask_tint_and_boxes(indices):
    fig, axes = plt.subplots(ncols=4, nrows=len(indices), figsize=(16, 4 * len(indices)), sharex='all', sharey='all')

    for i, idx in enumerate(indices):
        image = cv2.imread(os.path.join(IMAGES_DIR, train_image_files[idx]))
        mask = cv2.imread(os.path.join(TRAIN_LABELS_DIR, train_label_files[idx]))

        box_image = image.copy()
        draw_boxes_on_image(box_image, mask)

        if len(indices) == 0:
            rowax = axes
        else:
            rowax = axes[i]

        rowax[0].set_ylabel(f"IDX: \n{idx} ", rotation=0, size='large')

        rowax[0].imshow(image)
        rowax[1].imshow(mask)
        rowax[2].imshow(tint_image(image, mask))
        rowax[3].imshow(box_image)

    # fig.tight_layout()


def mask_to_bounding_boxes(mask):
    mask_label = measure.label(mask)
    mask_props = measure.regionprops(mask_label)
    bboxes = []
    for prop in mask_props:
        bboxes.append([prop.bbox[1], prop.bbox[0], prop.bbox[4], prop.bbox[3]])
    return bboxes


def add_bounding_boxes_to_image(image, bboxes, color):
    for bbox in bboxes:
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 5)


def draw_boxes_on_image(image, mask):
    masks = separate_masks(mask)
    for i, cur_mask in enumerate(masks):
        mask_class = CLASS_LABELS[i]
        color = COLORS[mask_class]
        add_bounding_boxes_to_image(image, mask_to_bounding_boxes(cur_mask), color)


# CONVERSION / EXPORT
def coordinates_bbox_to_coco(bbox, class_idx):
    x_start = bbox[0]
    y_start = bbox[1]
    width = bbox[2] - x_start
    height = bbox[3] - y_start
    return [class_idx, 1, x_start, y_start, width, height]


def coco_bbox_to_yolo_string(coco_bbox, image_dim=DEFAULT_IMAGE_DIM):
    class_idx, _, start_x, start_y, width, height = coco_bbox
    center_x = (start_x + (width / 2)) / image_dim[0]
    center_y = (start_y + (height / 2)) / image_dim[1]
    return f"{class_idx} {center_x} {center_y} {width / image_dim[0]} {height / image_dim[1]}"


def coco_bbox_sizes(coco_bboxes):
    return [bb[4] * bb[5] for bb in coco_bboxes]


def proportional_coco_bbox_sizes(coco_bboxes, image_dim=DEFAULT_IMAGE_DIM):
    image_size = image_dim[0] * image_dim[1]
    return np.array(coco_bbox_sizes(coco_bboxes)) / image_size


def generate_coco_metadata(image_files, labels_dir, label_files, num_workers=1):
    # instantiate our results queue
    results = Queue()

    # define a function to take in an index and write the result to the results queue
    def process_idx(idx):
        image_path = os.path.join(IMAGES_DIR, image_files[idx])
        full_mask = cv2.imread(os.path.join(labels_dir, label_files[idx]))
        split_masks = separate_masks(full_mask)
        bboxes = []
        for class_idx in range(4):
            cur_bboxes = mask_to_bounding_boxes(split_masks[class_idx])
            for cur_bbox in cur_bboxes:
                coco_bbox = coordinates_bbox_to_coco(cur_bbox, class_idx)
                bboxes.append(coco_bbox)

        results.put((image_path, bboxes))

    # instantiate our inputs queue
    indices = Queue()

    # define a worker loop
    def do_work():
        # loop until the return statement is hit
        while True:
            try:
                # try to fetch an index to process
                idx = indices.get(timeout=1.)
            # if there is none, an Empty exception will be raised so we can return, completing this thread's work
            except Empty:
                return
            # otherwise we'll process the index and store the result
            process_idx(idx)
            # mark this task as completed -- not strictly necessary since we're joining on worker threads not the queue
            indices.task_done()

    # dump all of our indices into the input queue
    for i in range(len(label_files)):
        indices.put(i)

    # instantiate and start threads
    threads = []
    for tidx in range(num_workers):
        thread = Thread(target=do_work, daemon=True)
        threads.append(thread)
        thread.start()
    # wait for them to finish
    for thread in threads:
        thread.join()

    # once they're done, pull our results back from the queue
    image_paths = []
    bounding_boxes = []
    while not results.empty():
        image_path, bboxes = results.get()
        image_paths.append(image_path)
        bounding_boxes.append(bboxes)
        results.task_done()

    result = {"image": image_paths, "label": bounding_boxes}
    sorted_result = pd.DataFrame(result).sort_values(by=['image']).to_dict(orient='list')
    return sorted_result


def metadata_to_yolo_folder(metadata, target_folder, image_dim=DEFAULT_IMAGE_DIM, overwrite=False):
    # if the target folder is not empty
    if os.path.exists(target_folder) and not os.listdir(target_folder):
        if overwrite:
            print(f"removing contents of target folder '{target_folder}'")
            shutil.rmtree(target_folder)
        else:
            raise ValueError(f"target folder '{target_folder}' is not empty but overwrite is False!")

    # define and create subfolders
    images_folder = os.path.join(target_folder, "images")
    labels_folder = os.path.join(target_folder, "labels")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    # process each metadata file
    for i in range(len(metadata['image'])):
        image_path = metadata['image'][i]
        labels = metadata['label'][i]
        image_fname = image_path.split("/")[-1]
        label_fname = image_fname.replace(".JPG", ".txt")

        # copy image to new folder
        with open(os.path.join(images_folder, image_fname), 'wb') as outf:
            with open(image_path, 'rb') as inf:
                outf.write(inf.read())

        # reshape label
        with open(os.path.join(labels_folder, label_fname), 'w') as f:
            for label in labels:
                f.write(coco_bbox_to_yolo_string(label, image_dim) + "\n")


# FILTERING


def coco_box_is_nested(outside_box, inside_box):
    """
    Returns true if `inside_box` is completely inside of `outside_box`
    """
    outside_left_x = outside_box[2]
    outside_top_y = outside_box[3]
    outside_right_x = outside_left_x + outside_box[4]
    outside_bottom_y = outside_top_y + outside_box[5]

    inside_left_x = inside_box[2]
    inside_top_y = inside_box[3]
    inside_right_x = inside_left_x + inside_box[4]
    inside_bottom_y = inside_top_y + inside_box[5]

    return (inside_left_x >= outside_left_x and
            inside_top_y >= outside_top_y and
            inside_right_x <= outside_right_x and
            inside_bottom_y <= outside_bottom_y)


def get_nested_coco_boxes(labels):
    # build out our list of the indices of the labels in each class
    indices_by_class = {cl: [] for cl in CLASS_LABELS}
    for i, label in enumerate(labels):
        if len(label) == 0:
            raise ValueError(f"label was empty: full labels {labels}")
        cur_class = CLASS_LABELS[label[0]]
        indices_by_class[cur_class].append(i)

    # for each class
    nested_box_indices = []
    for class_label, indices in indices_by_class.items():
        # for each index in the class
        class_size = len(indices)
        for outside_list_idx in range(class_size):
            outside_label_idx = indices[outside_list_idx]
            inside_label_indices = indices[:outside_list_idx] + indices[outside_list_idx + 1:]
            # compare it to the other values
            for inside_label_idx in inside_label_indices:
                outside_box = labels[outside_label_idx]
                inside_box = labels[inside_label_idx]
                if coco_box_is_nested(outside_box, inside_box):
                    nested_box_indices.append(inside_label_idx)

    return list(np.sort(np.unique(nested_box_indices)))


def coco_box_coverage_area(coco_bboxes, image_dim=DEFAULT_IMAGE_DIM):
    """
    Return proportion of the image covered by boxes
    """
    image_size = image_dim[0] * image_dim[1]
    raw_sizes = np.array(coco_bbox_sizes(coco_bboxes))
    proportional_sizes = raw_sizes / image_size
    return proportional_sizes.sum()


def preprocess_label(coco_boxes, remove_nested_boxes=True, min_coverage=0.0, max_boxes=1000,
                     min_box_proportional_size=1.0, image_dim=DEFAULT_IMAGE_DIM):
    """
    Apply preprocessing and filtering to a set of COCO boxes. If the label passes the minimum coverage
    and maximum number of boxes, return the processed label. Otherwise, return None.
    """
    # remove nested boxes
    if remove_nested_boxes:
        nested_indices = get_nested_coco_boxes(coco_boxes)
        coco_boxes = np.delete(arr=coco_boxes, obj=nested_indices, axis=0)

    # next remove boxes that are too small
    prop_sizes = proportional_coco_bbox_sizes(coco_boxes)
    coco_boxes = coco_boxes[prop_sizes >= min_box_proportional_size]

    # then check that we have sufficient image coveraege
    if coco_box_coverage_area(coco_boxes.tolist()) < min_coverage:
        return None

    # finally validate there aren't too many boxes
    if coco_boxes.shape[0] > max_boxes:
        return None

    return coco_boxes
