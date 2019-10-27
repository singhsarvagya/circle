from __future__ import print_function, division
import torch
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class CirclesDataset(Dataset):
    """Circles dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.circles = pd.read_csv(root_dir + csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.circles)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(self.root_dir + self.circles.iloc[idx, 0])
        params = self.circles.iloc[idx, 1:]
        params = np.array([params])
        params = torch.from_numpy(params.astype('float32').reshape(3))

        if self.transform:
            img = self.transform(img)
        return img, params