
from os import path, listdir
import random
from typing import List, Callable, Any

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


CATEGORIES = ["amusement_park", "place_of_worship", "recreational_facility"]
COUNTRY_CODES = ["FRA", "RUS"]


preprocess = transforms.ToTensor()


def print_dir_counts(target: str, prefix=""):
    contents = [x for x in listdir(target) if not x.startswith(".")]
    files = [x for x in contents if path.isfile(path.join(target, x))]
    dirs = [x for x in contents if path.isdir(path.join(target, x))]

    line = prefix + target.split("/")[-1] + "/"
    if len(files) > 0:
        line += " " * (30 - len(line)) + f"[{len(files)} files]"
    print(line)
    for d in dirs:
        print_dir_counts(path.join(target, d), prefix + "  ")


def random_fname(category: str):
    country = "FRA" if random.random() < 0.5 else "RUS"
    folder = f"fmow-data/{category}/train/{country}"
    files = listdir(folder)
    fname = files[int(random.random() * len(files))]
    return path.join(folder, fname)


def load_and_preprocess_image(image_path: str):
    img = Image.open(image_path).convert('RGB')
    return preprocess(img)


def tensor_to_lime_array(tensor: Tensor):
    channels, pixdim1, pixdim2 = tensor.shape
    return torch.reshape(tensor, (pixdim1, pixdim2, channels)).double().numpy()


def batched_lime_array_to_tensor(batch: np.array):
    batch_size, pixdim1, pixdim2, channels = batch.shape
    raw_tensor = torch.from_numpy(batch)
    return torch.reshape(raw_tensor, (batch_size, channels, pixdim1, pixdim2)).float()


class ImageFilesDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[Any], transform: Callable = None):
        if len(image_paths) != len(labels):
            raise ValueError(f"image paths length {len(image_paths)} does not match labels {len(labels)}")
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[idx]
