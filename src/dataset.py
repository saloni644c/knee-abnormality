import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import load_series


class ExamDataset(Dataset):
    """One exam per item: a (n_slices, 1, H, W) tensor for one plane, plus its binary label."""

    def __init__(self, study_uids, labels, series_map, data_root, reference, plane, size=224):
        self.study_uids = list(study_uids)
        self.labels = labels            # dict: StudyInstanceUID -> 0 or 1
        self.series_map = series_map    # dict: StudyInstanceUID -> {plane: SeriesInstanceUID}
        self.data_root = data_root
        self.reference = reference
        self.plane = plane
        self.size = size

    def __len__(self):
        return len(self.study_uids)

    def __getitem__(self, i):
        uid = self.study_uids[i]
        series_id = self.series_map[uid][self.plane]
        series_dir = os.path.join(self.data_root, uid, series_id)
        vol = load_series(series_dir, self.reference, self.size)
        x = torch.from_numpy(vol).unsqueeze(1).float()
        y = torch.tensor(float(self.labels[uid]))
        return x, y