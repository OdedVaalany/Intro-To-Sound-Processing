"""
This file will define the general utility functions you will need for you implementation throughout this ex.
We suggest you start with implementing and testing the functions in this file.

NOTE: each function has expected typing for it's input and output arguments. 
You can assume that no other input types will be given and that shapes etc. will be as described.
Please verify that you return correct shapes and types, failing to do so could impact the grade of the whole ex.

NOTE 2: We STRONGLY encourage you to write down these function by hand and not to use Copilot/ChatGPT/etc.
Implementaiton should be fairly simple and will contribute much to your understanding of the course material.

NOTE 3: You may use external packages for fft/stft, you are requested to implement the functions below to 
standardize shapes and types.
"""
import torchaudio as ta
import soundfile as sf
import torch
import typing as tp
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import scipy
import numpy as np


def create_single_sin_wave(frequency_in_hz, total_time_in_secs=3, sample_rate=16000):
    timesteps = np.arange(0, total_time_in_secs * sample_rate) / sample_rate
    sig = np.sin(2 * np.pi * frequency_in_hz * timesteps)
    return torch.Tensor(sig).float()


def load_wav(abs_path: tp.Union[str, Path]) -> tp.Tuple[torch.Tensor, int]:
    """
    This function loads an audio file (mp3, wav).
    If you are running on a computer with gpu, make sure the returned objects are mapped on cpu.

    abs_path: path to the audio file (str or Path)
    returns: (waveform, sample_rate)
        waveform: torch.Tensor (float) of shape [1, num_channels]
        sample_rate: int, the corresponding sample rate
    """
    return ta.load(abs_path)


def do_stft(wav: torch.Tensor, n_fft: int = 1024) -> torch.Tensor:
    """
    This function performs STFT using win_length=n_fft and hop_length=n_fft//4.
    Should return the complex spectrogram.

    hint: see torch.stft.

    wav: torch tensor of the shape (1, T) or (B, 1, T) for the batched case.
    n_fft: int, denoting the number of used fft bins.

    returns: torch.tensor of the shape (1, n_fft, *, 2) or (B, 1, n_fft, *, 2), where last dim stands for real/imag entries.
    """
    if len(wav.shape) == 2:
        return do_stft(wav.unsqueeze(0), n_fft).squeeze(0)
    return torch.view_as_real(torch.stft(wav.squeeze(1), n_fft=n_fft, win_length=n_fft,
                                         hop_length=n_fft//4, window=torch.torch.ones(n_fft), return_complex=True, onesided=False)).unsqueeze(1)


def do_istft(spec: torch.Tensor, n_fft: int = 1024) -> torch.Tensor:
    """
    This function performs iSTFT using win_length=n_fft and hop_length=n_fft//4.
    Should return the complex spectrogram.

    hint: see torch.istft.

    spec: torch.tensor of the shape (1, n_fft, *, 2) or (B, 1, n_fft, *, 2), where last dim stands for real/imag entries.
    n_fft: int, denoting the number of used fft bins.

    returns: torch tensor of the shape (1, T) or (B, 1, T) for the batched case.

    NOTE: you may need to use torch.view_as_complex.
    """
    if len(spec.shape) == 4:
        return do_istft(spec.unsqueeze(0), n_fft).squeeze(0)
    return torch.istft(torch.view_as_complex(spec).squeeze(1), window=torch.ones(n_fft), length=n_fft if spec.shape[-2] == 1 else None, n_fft=n_fft, win_length=n_fft, onesided=False, hop_length=n_fft//4).unsqueeze(1)


def do_fft(wav: torch.Tensor) -> torch.Tensor:
    """
    This function performs fast fourier trasform (FFT) .

    hint: see scipy.fft.fft / torch.fft.rfft, you can convert the input tensor to numpy just make sure to cast it back to torch.

    wav: torch tensor of the shape (1, T).

    returns: corresponding FFT transformation considering ONLY POSITIVE frequencies, returned tensor should be of complex dtype.
    """
    return torch.fft.rfft(wav)


def plot_spectrogram(wav: torch.Tensor, n_fft: int = 1024, sr=16000) -> None:
    """
    This function plots the magnitude spectrogram corresponding to a given waveform.
    The Y axis should include frequencies in Hz and the x axis should include time in seconds.

    wav: torch tensor of the shape (1, T) or (B, 1, T) for the batched case.

    NOTE: for the batched case multiple plots should be generated (sequentially by order in batch)
    """
    if len(wav.shape) == 2:
        wav = wav.unsqueeze(1)
    for w in wav:
        plt.specgram(w[0], NFFT=n_fft, Fs=sr)
        plt.xlabel('Time (sec)')
        plt.ylabel('Magnitude')
        plt.show()


def plot_fft(wav: torch.Tensor) -> None:
    """
    This function plots the FFT transform to a given waveform.
    The X axis should include frequencies in Hz.

    NOTE: As abs(FFT) reflects around zero, please plot only the POSITIVE frequencies.

    wav: torch tensor of the shape (1, T) or (B, 1, T) for the batched case.
    """
    if len(wav.shape) == 2:
        wav = wav.unsqueeze(1)
    for w in wav:
        plt.plot(torch.abs(do_fft(w))[0])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.show()
