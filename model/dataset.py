import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

import os

class FlowerDataset(Dataset):
    def __init__(self, df, 
                 shape, # 追加
                 transforms=None,
                 output_label=True,
                 one_hot_label=False,
                 image_name_col = "image_path",
                 label_col = "label"
                ):

        super().__init__()
        self.shape = shape
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.output_label = output_label
        self.one_hot_label = one_hot_label
        self.image_name_col = image_name_col
        self.label_col = label_col

        if output_label == True:
            self.labels = self.df[self.label_col].values
            if one_hot_label is True:
                self.labels = np.eye(self.df[self.label_col].max()+1)[self.labels]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):

        if self.output_label:
            target = self.labels[index]

        img  = get_img(self.df.loc[index][self.image_name_col])

        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.output_label == True:
            return img, target
        else:
            return img