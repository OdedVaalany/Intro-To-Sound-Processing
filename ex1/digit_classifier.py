

"""
This file will implement a digit classifier using rule-based dsp methods.
As all digit waveforms are given, we could take that under consideration, of our RULE-BASED system.

We reccomend you answer this after filling all functions in general_utilities.
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

# --------------------------------------------------------------------------------------------------
#     Part A        Part A        Part A        Part A        Part A        Part A        Part A
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# In this part we will get familiarized with the basic utilities defined in general_utilities
# --------------------------------------------------------------------------------------------------


def self_check_fft_stft():
    """
    Q:
    1. create 1KHz and 3Khz sine waves, each of 3 seconds length with a sample rate of 16KHz.
    2. In a single plot (3 subplots), plot (i) FFT(sine(1Khz)) (ii) FFT(sine(3Khz)), 
       (iii) FFT(sine(1Khz) + sine(3Khz)), make sure X axis shows frequencies. 
       Use general_utilities.plot_fft
    3. concatate [sine(1Khz), sine(3Khz), sine(1Khz) + sine(3Khz)] along the temporal axis, and plot
       the corresponding MAGNITUDE STFT using n_fft=1024. Make sure Y ticks are frequencies and X
       ticks are seconds.

    Include all plots in your PDF
    """
    SR, TT = 16000, 3
    first_sine = gu.create_single_sin_wave(1000, TT, SR).unsqueeze(0)
    second_sine = gu.create_single_sin_wave(3000, TT, SR).unsqueeze(0)
    combined_sine = first_sine + second_sine
    batch = torch.stack([first_sine, second_sine, combined_sine], dim=0)
    gu.plot_fft(batch)


def audio_check_fft_stft():
    """
    Q:
    1. load all phone_*.wav files in increasing order (0 to 11)
    2. In a single plot (2 subplots), plot (i) FFT(phone_1.wav) (ii) FFT(phone_2.wav). 
       Use general_utilities.plot_fft
    3. concatate all phone_*.wav files in increasing order (0 to 11) along the temporal axis, and plot
       the corresponding MAGNITUDE STFT using n_fft=1024. Make sure Y ticks are frequencies and X
       ticks are seconds.

    Include all plots in your PDF
    """
    phone_wav = [
        ta.load(f"./audio_files/phone_digits_8k/phone_{i}.wav")[0] for i in range(12)]
    phone_wav_stacked = torch.stack(phone_wav, dim=0)
    phone_wav_concat = torch.cat(phone_wav, dim=-1)
    gu.plot_fft(phone_wav_stacked)
    gu.plot_spectrogram(phone_wav_concat, n_fft=1024)


# --------------------------------------------------------------------------------------------------
#     Part B        Part B        Part B        Part B        Part B        Part B        Part B
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Digit Classifier
# --------------------------------------------------------------------------------------------------

def classify_single_digit(wav: torch.Tensor) -> int:
    """
    Q:
    Write a RULE-BASED (if - else..) function to classify a given single digit waveform.
    Use ONLY functions from general_utilities file.

    Hint: try plotting the fft of all digits.

    wav: torch tensor of the shape (1, T).

    return: int, digit number
    """
    wav_fft = (torch.sort(torch.abs(gu.do_fft(wav)
                                    ).squeeze(0), descending=True)[1][:2])
    if torch.equal(wav_fft, torch.Tensor([94, 134])):
        return 0
    elif torch.equal(wav_fft, torch.Tensor([121,  70])):
        return 1
    elif torch.equal(wav_fft, torch.Tensor([70, 134])):
        return 2
    elif torch.equal(wav_fft, torch.Tensor([148,  70])):
        return 3
    elif torch.equal(wav_fft, torch.Tensor([77, 121])):
        return 4
    elif torch.equal(wav_fft, torch.Tensor([77, 134])):
        return 5
    elif torch.equal(wav_fft, torch.Tensor([77, 148])):
        return 6
    elif torch.equal(wav_fft, torch.Tensor([121,  85])):
        return 7
    elif torch.equal(wav_fft, torch.Tensor([85, 134])):
        return 8
    elif torch.equal(wav_fft, torch.Tensor([85, 148])):
        return 9
    elif torch.equal(wav_fft, torch.Tensor([94, 121])):
        return 10
    elif torch.equal(wav_fft, torch.Tensor([94, 148])):
        return 11
    else:
        return -1


def classify_digit_stream(wav: torch.Tensor) -> tp.List[int]:
    """
    Q:
    Write a RULE-BASED (if - else..) function to classify a waveform containing several digit stream.
    The input waveform will include at least a single digit in it.
    The input waveform will have digits waveforms concatenated on the temporal axis, with random zero
    padding in-between digits.
    You can assume that there will be at least 100ms of zero padding between digits
    The function should return a list of all integers pressed (in order).

    Use STFT from general_utilities file to answer this question.

    wav: torch tensor of the shape (1, T).

    return: List[int], all integers pressed (in order).
    """
    stft = gu.do_stft(wav, 800)
    stft = stft.permute(2, 1, 0, 3).unsqueeze(1)

    istft = gu.do_istft(stft, 800).squeeze(1)
    st = []
    i = 0
    while i < istft.shape[0]:
        output = classify_single_digit(istft[i].unsqueeze(0))
        if output != -1:
            st.append(output)
            i += 5
        else:
            i += 1
    return st
