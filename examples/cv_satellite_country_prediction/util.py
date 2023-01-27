from random import random

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from os import path, listdir

from fmow_model.data import ImageFilesDataset, preprocess


def predict_onto_df(df, model):
    dataset = ImageFilesDataset(list(df['satellite_image']), [float('nan') for _ in range(len(df))], transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    results = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            output = model.forward(inputs)
            result = F.softmax(output, dim=1).tolist()
            results.extend(result)
    results = np.array(results)
    df['russia'] = results[:, 1]
    return df


def create_batch_df(category, max_size=100, split="test"):
    folder = f"fmow-data/{category}/{split}"
    france_paths = [path.join(folder, "FRA", x) for x in listdir(path.join(folder, "FRA"))]
    russia_paths = [path.join(folder, "RUS", x) for x in listdir(path.join(folder, "RUS"))]
    all_paths = france_paths + russia_paths
    all_labels = [0 for _ in range(len(france_paths))] + [1 for _ in range(len(russia_paths))]

    rand_size = int(random() * ((max_size-1)*.7) + 1 + (max_size * .3))
    size = rand_size if rand_size < len(all_paths) else len(all_paths)
    indices = np.random.choice(np.array(range(len(all_paths))), size=size, replace=False)

    paths = [all_paths[idx] for idx in indices]
    russia_labels = [all_labels[idx] for idx in indices]
    country_labels = ['russia' if l == 1 else 'france' for l in russia_labels]

    return pd.DataFrame({'satellite_image': paths, 'country': country_labels,
                         'feature': category})
