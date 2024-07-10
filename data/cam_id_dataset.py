import os
import h5py

import torch
from torchvision.transforms import RandomCrop
from torch.utils.data import Dataset
from .midb_classes import model_to_id, id_to_model

from tqdm.auto import tqdm
from typing import *


class CamIdDataset(Dataset):
    def __init__(
        self,
        root_dir,
        ds_patch_size,
        model_patch_size,
        split,
        calc_class_weights=True,
        ignore_classes: Union[None, List[int]] = None,
        remap_labels=False
    ):
        self.remap_labels = remap_labels

        self.h5py_file = h5py.File(os.path.join(root_dir, f"{ds_patch_size}", f"{split}.hdf5"), "r")
        self.ds_idxs = torch.arange(len(self.h5py_file["labels"]))

        self.ds_labels = torch.from_numpy(self.h5py_file["labels"][:])
        if ignore_classes is not None:
            print("Ignoring classes:", [id_to_model[label] for label in ignore_classes])
            select_mask = torch.isin(self.ds_labels, torch.tensor(ignore_classes), invert=True)
            self.ds_idxs = self.ds_idxs[select_mask]
            self.ds_labels = self.ds_labels[select_mask]

        self.counter = {i: c for i, c in enumerate(self.ds_labels.bincount()) if c > 0}
        self.label_map = {label: idx for idx, label in enumerate(self.counter.keys())}
        self.inv_label_map = {idx: label for label, idx in self.label_map.items()}

        if calc_class_weights:
            average = sum(list(self.counter.values())) / len(self.counter)
            self.class_weights = torch.tensor([average / self.counter[label] for label in self.counter.keys()])
        else:
            self.class_weights = torch.tensor([1.0 for _ in self.counter.keys()])

        print(f"[INFO]: Model patch size = {model_patch_size}, Dataset patch size = {ds_patch_size}")
        if model_patch_size != ds_patch_size:
            self.transform = RandomCrop(model_patch_size)
        else:
            self.transform = torch.nn.Identity()

    def __len__(self):
        return len(self.ds_idxs)
    
    def __getitem__(self, idx):
        idx = self.ds_idxs[idx]
        img = torch.from_numpy(self.h5py_file["patches"][idx]).float().div(255)
        label = self.h5py_file["labels"][idx]

        if self.remap_labels:
            label = self.label_map[label]

        return self.transform(img), label

