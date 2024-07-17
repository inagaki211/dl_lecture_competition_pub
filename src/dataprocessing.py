import numpy as np
from scipy.signal import resample

class EEGPreprocessing:
    def __init__(self, target_sampling_rate: int, baseline_start_ms: int, baseline_end_ms: int):
        self.target_sampling_rate = target_sampling_rate
        self.baseline_start_ms = baseline_start_ms
        self.baseline_end_ms = baseline_end_ms

    def preprocess(self, signal, original_sampling_rate, epoch_start_ms, epoch_end_ms):
        # エポックの抽出
        signal = self.extract_epoch(signal, original_sampling_rate, epoch_start_ms, epoch_end_ms)
        # ダウンサンプリング
        signal = self.downsample(signal, original_sampling_rate, self.target_sampling_rate)
        # ベースライン補正
        signal = self.baseline_correction(signal, self.baseline_start_ms, self.baseline_end_ms)
        # スケーリング
        signal = self.scale(signal)
        return signal

    def extract_epoch(self, signal, original_rate, start_ms, end_ms):
        start_idx = int((start_ms / 1000) * original_rate)
        end_idx = int((end_ms / 1000) * original_rate)
        return signal[:, start_idx:end_idx]

    def downsample(self, signal, original_rate, target_rate):
        num_samples = int((signal.shape[1] / original_rate) * target_rate)
        return resample(signal, num_samples, axis=1)

    def baseline_correction(self, signal, epoch_start_ms, epoch_end_ms):
         # ベースライン期間のインデックス計算 (エポック開始時間からの相対位置)
        baseline_start_idx = int((self.baseline_start_ms - epoch_start_ms) / 1000 * self.target_sampling_rate)
        baseline_end_idx = int((self.baseline_end_ms - epoch_start_ms) / 1000 * self.target_sampling_rate)
        
        # ベースライン期間の値を抽出
        baseline_values = signal[:, baseline_start_idx:baseline_end_idx]
         # ベースライン期間が空でないかチェック
        if baseline_values.size == 0:
            raise ValueError("Baseline period is empty. Check the baseline start and end times relative to the epoch period.")
        
        # ベースライン期間の平均を計算
        baseline_mean = baseline_values.mean(axis=1, keepdims=True)
        
        # ベースライン平均をエポック全体から減算
        return signal - baseline_mean

    def scale(self, signal):
        return (signal - signal.mean()) / signal.std()
