import torch
from torch.utils.data import Dataset
import librosa
import numpy as np

class getdatapath(Dataset):
    def __init__(self, file_name):
        with open(file_name, 'r', encoding='utf-8') as file:
            self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        clean_path = './dataset/wavs_clean/' + line[-12:]
        noisy_path = './dataset/wavs_noisy/' + line[-12:]

        # return {'noisy_path' : noisy_path , 'clean_path' : clean_path}
        return noisy_path, clean_path
