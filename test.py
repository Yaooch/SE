import torch
import numpy as np
import librosa
import os
from torch import nn
import soundfile as sf
import pesq
import torch.nn.functional as F
import torchaudio
from functions import *
from getdatapath import *
from pystoi.stoi import stoi

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), padding=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(5, 5), padding=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(5, 5), padding=(2, 2))

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x

test_path = './dataset/test.txt'
test_data = getdatapath(test_path)

def audio_enhance(path, method):
    window = torch.hann_window(512)
    window = window.to(device)
    data = torchaudio.load(path)[0]
    data = data.to(device)
    stft = torch.stft(data, n_fft=512, hop_length=128, window=window, return_complex=True)
    magnitude = torch.squeeze(torch.abs(stft), 0)
    phase = torch.squeeze(torch.angle(stft), 0)

    magnitude = magnitude.to(device)
    phase = phase.to(device)
    feature = torch.unsqueeze(torch.unsqueeze(magnitude.T, 0), 0)
    with torch.no_grad():
        mask = model(feature)
    mask = mask.squeeze(0).squeeze(0).T

    if method == 'direct':
        en_magnitude = mask
    else:
        if method == 'ibm':
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
        mask = mask.clamp(min=1e-7, max=1)
        en_magnitude = magnitude * mask

    frame = torch.istft(en_magnitude * torch.exp(1j * phase), n_fft=512, hop_length=128, window=window)
    frame = frame.cpu()
    frame = frame.numpy()
    return frame


fs = 16000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
method_list = ['ibm', 'iam', 'irm', 'psm', 'orm']

for method in method_list:
    model = CNN()
    model = model.to(device)
    model.load_state_dict(torch.load('./model/model_' + method + '.pth'))
    model.eval()

    pesq_enhanced_sum = 0
    stoi_ehanced_sum = 0
    pesq_raw_sum = 0
    stoi_raw_sum = 0

    for data in test_data:
        noisy_path, clean_path = data
        audio_enhanced = audio_enhance(noisy_path, method = method)
        audio_clean = librosa.load(clean_path, sr=fs)[0]
        audio_noisy = librosa.load(noisy_path, sr=fs)[0]

        if len(audio_enhanced) < len(audio_clean):
            pad_length = len(audio_clean) - len(audio_enhanced)
            audio_enhanced = np.pad(audio_enhanced, (0, pad_length), 'constant')

        output_path = './enhanced_audio/' + method + '/' + noisy_path[-12:]
        sf.write(output_path, audio_enhanced, fs)
        pesq_enhanced_sum += pesq.pesq(fs, audio_clean, audio_enhanced, 'wb')
        pesq_raw_sum += pesq.pesq(fs, audio_clean, audio_noisy, 'wb')
        stoi_ehanced_sum += stoi(audio_clean, audio_enhanced, fs)
        stoi_raw_sum += stoi(audio_clean, audio_noisy, fs)

    if method == method_list[0]:
        total_param = sum(p.numel() for p in model.parameters())
        print(f'模型的总参数为:{total_param}, 以下是不同方法增强后对应的客观评价指标:')
        print(f'原始数据平均pesq值为{pesq_raw_sum / len(test_data)},平均stoi值为{stoi_raw_sum / len(test_data)}')
    print(f'{method}方法增强后平均pesq值为{pesq_enhanced_sum / len(test_data)}, 平均stoi值为{stoi_ehanced_sum / len(test_data)}')


