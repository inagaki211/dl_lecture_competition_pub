import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from src.dataprocessing import EEGPreprocessing

class ThingsMEGDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "data", baseline_start_ms=-500, baseline_end_ms=0, target_sampling_rate=120):
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"

        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))
        self.original_sampling_rate = 200
        # EEGPreprocessingクラスのインスタンスを初期化
        self.preprocessor = EEGPreprocessing(target_sampling_rate, baseline_start_ms, baseline_end_ms)
        # 追加: 被験者の総数を計算
        self._calculate_num_subjects()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = np.load(X_path)

        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = np.load(subject_idx_path)

        # データ前処理の適用
        # エポックの開始と終了（刺激開始を基準に -500 ms から 1000 ms ）
        epoch_start_ms = -500
        epoch_end_ms = 1000

        X = self.preprocessor.preprocess(X, self.original_sampling_rate, epoch_start_ms, epoch_end_ms)
        X = torch.from_numpy(X)

        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path))
            return X, y, torch.from_numpy(subject_idx)
        else:
            return X, torch.from_numpy(subject_idx)

    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]

    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]

    @property
    def num_subjects(self) -> int:
        return self._num_subjects

    def _calculate_num_subjects(self):
        unique_subjects = set()
        subject_idxs_dir = os.path.join(self.data_dir, f"{self.split}_subject_idxs")
        for file in glob(os.path.join(subject_idxs_dir, "*.npy")):
            subject_idx = np.load(file)
            unique_subjects.add(subject_idx.item())
        self._num_subjects = len(unique_subjects)
