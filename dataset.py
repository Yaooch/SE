import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from functions import *
class GetDataset(Dataset):
    def __init__(self, method):
        self.method = method
        # 读取训练集或测试集中的文件名
        with open('./dataset/training.txt', 'r') as f:
            self.file_names = [line.strip()[-12:] for line in f.readlines()]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        # 读取干净语音和带噪语音
        clean_path = os.path.join('./dataset/wavs_clean', file_name)
        noisy_path = os.path.join('./dataset/wavs_noisy', file_name)
        clean_waveform, _ = torchaudio.load(clean_path)
        noisy_waveform, _ = torchaudio.load(noisy_path)

        clean_waveform = clean_waveform[0]  # 假设是单通道音频
        noisy_waveform = noisy_waveform[0]
        noise_waveform = noisy_waveform - clean_waveform

        # 计算STFT和IRM
        clean_stft = calculate_stft(clean_waveform)
        noisy_stft = calculate_stft(noisy_waveform)
        noise_stft = calculate_stft(noise_waveform)

        # 计算MASK
        if self.method == 'iam':
            mask = calculate_iam(clean_stft, noisy_stft)
        elif self.method == 'psm':
            mask = calculate_psm(clean_stft, noisy_stft)
        elif self.method == 'irm':
            mask = calculate_irm(clean_stft, noise_stft)
        elif self.method == 'orm':
            mask = calculate_orm(clean_stft, noise_stft)
        elif self.method == 'ibm':
            mask = calculate_ibm(clean_stft, noise_stft)
        else:  # self.method == 'direct'
            mask = torch.abs(clean_stft)

        feature = torch.unsqueeze(torch.abs(noisy_stft).T, 0)  # feature维度：[channel=1, time_frames, freq_bins]
        label = torch.unsqueeze(mask.T, 0)  # label维度同feature

        return {'feature': feature, 'label': label}