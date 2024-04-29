import math

import torch
from torch.utils.data import Dataset
from .cam_id_dataset import CamIdDataset

from tqdm.auto import tqdm
from typing import *


class FsmDataset(Dataset):
    def __init__(
        self,
        cam_id_ds: CamIdDataset,
        lookahead_size=1024,
        
        dissim_vs_sim_ratio=1,
        calc_class_weights=True,
    ):
        self.cam_id_ds = cam_id_ds
        labels = self.cam_id_ds.ds_labels
        idxs = torch.arange(len(labels))

        num_chunks = math.ceil(len(idxs) / lookahead_size)
        label_chunks = torch.chunk(labels, num_chunks)
        idx_chunks = torch.chunk(idxs, num_chunks)

        num_similar_pairs = 0
        num_dissimilar_pairs = 0
        self.pair_idxs = []
        for label_chunk, idx_chunk in tqdm(
            zip(label_chunks, idx_chunks), total=num_chunks, desc="Creating pairs"
        ):
            chunk_idxs = torch.arange(len(label_chunk))
            idx_combinations = torch.combinations(chunk_idxs, 2)
            label_combinations = label_chunk[idx_combinations]

            similar_combination_mask = (
                (label_combinations[:, 0] == label_combinations[:, 1]).nonzero().flatten()
            )
            similar_combination_mask = similar_combination_mask[
                torch.randperm(len(similar_combination_mask))[:len(similar_combination_mask) // 4]
            ]
            dissimilar_combination_mask = (
                (label_combinations[:, 0] != label_combinations[:, 1]).nonzero().flatten()
            )
            dissimilar_combination_mask = dissimilar_combination_mask[
                torch.randperm(len(dissimilar_combination_mask))[
                    : len(similar_combination_mask) * dissim_vs_sim_ratio
                ]
            ]

            similar_pairs_idxs = idx_chunk[idx_combinations[similar_combination_mask]]
            dissimilar_pairs_idxs = idx_chunk[idx_combinations[dissimilar_combination_mask]]

            pair_idxs = torch.cat((similar_pairs_idxs, dissimilar_pairs_idxs), dim=0)
            pair_idxs = pair_idxs[torch.randperm(len(pair_idxs))]
            self.pair_idxs.append(pair_idxs)

            num_similar_pairs += len(similar_pairs_idxs)
            num_dissimilar_pairs += len(dissimilar_pairs_idxs)

        self.pair_idxs = torch.cat(self.pair_idxs, dim=0)

        if calc_class_weights:
            average = (num_similar_pairs + num_dissimilar_pairs) / 2
            self.class_weights = torch.tensor([average / num_dissimilar_pairs, average / num_similar_pairs])
        else:
            self.class_weights = torch.tensor([1.0, 1.0])

    def __len__(self):
        return len(self.pair_idxs)

    def __getitem__(self, idx):
        idx1, idx2 = self.pair_idxs[idx]
        img1, label1 = self.cam_id_ds[idx1]
        img2, label2 = self.cam_id_ds[idx2]
        label = 0 if label1 == label2 else 1
        return img1, img2, label
