from general_utilities import *
import torch
import os
ROOT = str(Path(__file__).parent)
print(ROOT)


def test_get_audio_files():
    files, sr = load_wav(ROOT + "/audio_files/phone_digits_8k/phone_0.wav")
    assert sr == 8000
    assert len(files) == 1
    assert files.ndim == 2


def test_stft():
    files, sr = load_wav(ROOT + "/audio_files/phone_digits_8k/phone_0.wav")
    S = do_stft(files, n_fft=100)
    assert S.ndim == 4
    assert S.shape[1] == 100
    assert S.shape[-1] == 2


def test_stft_batch():
    files = [
        load_wav(ROOT + f"/audio_files/phone_digits_8k/phone_{i}.wav")[0] for i in range(6)]
    files = torch.stack(files)
    S = do_stft(files, n_fft=100)
    assert S.ndim == 5
    assert S.shape[-3] == 100
    assert S.shape[-1] == 2


def test_istft():
    dummy = torch.randn(1, 100, 10, 2)
    wav = do_istft(dummy, n_fft=100)
    assert wav.ndim == 2
    assert wav.shape[0] == 1


def test_tstft_batch():
    dummy = torch.randn(6, 1, 100, 10, 2)
    wav = do_istft(dummy, n_fft=100)
    assert wav.ndim == 3
    assert wav.shape[0] == 6

    dummy = torch.randn(6, 1, 200, 10, 2)
    wav = do_istft(dummy, n_fft=200)
    assert wav.ndim == 3
    assert wav.shape[0] == 6


def test_strucutre():
    assert os.path.exists(
        f"{ROOT}/digit_classifier.py"), "Missing digit_classifier.py file"
    assert os.path.exists(
        f"{ROOT}/general_utilities.py"), "Missing general_utilities.py file"
    assert os.path.exists(
        f"{ROOT}/time_stretch.py"), "Missing time_stretch.py file"
