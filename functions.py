import torch
def calculate_stft(signal, n_fft=512, hop_length=128):
    window = torch.hann_window(n_fft)
    stft = torch.stft(signal, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    return stft

def calculate_ibm(clean_stft, noise_stft):
    clean_mag = torch.abs(clean_stft)
    noise_mag = torch.abs(noise_stft)
    ibm = torch.zeros(clean_stft.shape)
    ibm[clean_mag > noise_mag] = 1.0
    return ibm

def calculate_iam(clean_stft, noisy_stft):
    clean_mag = torch.abs(clean_stft)
    noisy_mag = torch.abs(noisy_stft)
    iam = clean_mag / (noisy_mag + 1e-7)
    return iam.clamp(min=0, max=1)

def calculate_irm(clean_stft, noise_stft):
    clean_mag = torch.abs(clean_stft)
    noise_mag = torch.abs(noise_stft)
    irm = clean_mag ** 2 / (clean_mag ** 2 + noise_mag ** 2  + 1e-7)
    irm = irm ** 0.5
    return irm.clamp(min=0, max=1)  # 避免除以零和其他潜在的数值问题

def calculate_psm(clean_stft, noisy_stft):
    clean_mag = torch.abs(clean_stft)
    noisy_mag = torch.abs(noisy_stft)
    clean_pha = torch.angle(clean_stft)
    noisy_pha = torch.angle(noisy_stft)
    psm = (clean_mag / (noisy_mag + 1e-7)) * torch.cos(clean_pha - noisy_pha)
    return psm.clamp(min=0, max=1)  # 避免除以零和其他潜在的数值问题

def calculate_orm(clean_stft, noise_stft):
    clean_mag = torch.abs(clean_stft)
    noise_mag = torch.abs(noise_stft)
    coherent_part = torch.real(clean_stft * torch.conj(noise_stft))  # 相干部分
    orm = (clean_mag ** 2 + coherent_part) / (clean_mag ** 2 + noise_mag ** 2 + 2 * coherent_part + 1e-7)
    return orm.clamp(min=0, max=1)
