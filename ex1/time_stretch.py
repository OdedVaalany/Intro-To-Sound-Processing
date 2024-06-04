"""
In this file we will experiment with naively interpolating a signal on the time domain and on the frequency domain.

We reccomend you answer this file last.
"""
import general_utilities as gu
import torchaudio as ta
import soundfile as sf
import torch
import typing as tp
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import scipy
import numpy as np
from general_utilities import *
from torch.nn.functional import interpolate


def naive_time_stretch_temporal(wav: torch.Tensor, factor: float):
    """
    Q:
      write a function that uses a simple linear interpolation across the temporal dimension
      stretching/squeezing a given waveform by a given factor.
      Use imported 'interpolate'.

    1. load audio_files/Basta_16k.wav
    2. use this function to stretch it by 1.2 and by 0.8.
    3. save files using ta.save(fpath, stretch_wav, 16000) and listen to the files. What happened?
       Explain what differences you notice and why that happened in your PDF file

    Do NOT include saved audio in your submission.
    """
    return interpolate(wav, scale_factor=factor, mode='linear')


def naive_time_stretch_stft(wav: torch.Tensor, factor: float):
    """
    Q:
      write a function that converts a given waveform to stft, then uses a simple linear interpolation 
      across the temporal dimension stretching/squeezing by a given factor and converts the stretched signal 
      back using istft.
      Use general_utilities for STFT / iSTFT and imported 'interpolate'.

    1. load audio_files/Basta_16k.wav
    2. use this function to stretch it by 1.2 and by 0.8.
    3. save files using ta.save(fpath, stretch_wav, 16000) and listen to the files. What happened?
       Explain what differences you notice and why that happened in your PDF file

    Do NOT include saved audio in your submission.
    """
    stft = gu.do_stft(wav)
    stretched_stft = interpolate(stft.squeeze(1), scale_factor=(
        factor, 1), mode='bilinear').unsqueeze(1)
    stretched_wav = gu.do_istft(stretched_stft)
    return stretched_wav


if __name__ == "__main__":
    wav, rs = gu.load_wav('./audio_files/Basta_16k.wav')
    wav = wav.unsqueeze(0)
    ta.save('./audio_files/Basta_16k_stretched_0.8.wav',
            naive_time_stretch_temporal(wav, 0.8)[0], rs)
    ta.save('./audio_files/Basta_16k_stretched_1.2.wav',
            naive_time_stretch_temporal(wav, 1.2)[0], rs)

    wav = wav.permute(1, 0, 2)
    ta.save('./audio_files/Basta_16k_stretched_stft_0.8.wav',
            naive_time_stretch_stft(wav, 0.8)[0], rs)
    ta.save('./audio_files/Basta_16k_stretched_stft_1.2.wav',
            naive_time_stretch_stft(wav, 1.2)[0], rs)
