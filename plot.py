import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

list = ['clean', 'noisy', 'ibm', 'iam', 'irm', 'psm', 'orm']
fs = 16000
for i in range(len(list)):
    path = '../p257_028/' + list[i] + '.wav'
    wave, _ = librosa.load(path, sr = fs)
    wave = wave * 1.0 / (max(abs(wave)))
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 2, 1)
    # plt.title('带噪信号语谱图')
    plt.specgram(wave, Fs = fs, scale_by_freq = True, sides = 'default', cmap = 'viridis')
    plt.xlabel('time(s)')
    plt.ylabel('frequency(hz)')
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    times = np.arange(len(wave)) / float(fs)
    plt.plot(times, wave)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()

    plt.show()