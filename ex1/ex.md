# Ex 1 | Intro To Sound Proccessing 208230474

## Time Stretch

I encountered an issue when using the `naive_time_stretch_stft` function to
stretch the sound in the time domain. The output sounds incorrect when I try to
stretch the time with a factor of 0.8, as the voices sound higher than the
original. Similarly, when I stretch it with a factor of 1.2, the voices sound
lower. This can be explained by the fact that stretching the signal in the time
domain does not take into consideration the frequency dimensions, resulting in
a disruption of the harmonic structure of the sound and destroying the
frequencies.

However, when I stretch the signal with `naive_time_stretch_stft` in its
frequency domain, even though we play it faster and slower, since we didn't
break the harmonic structure of the frequencies, the signal sounds the same
with the same frequencies, just with a different rhythm.

## Self Check FFT

when I run a simpale test on my 'self_check_fft_stft' function. i've generate
1KHz sine wave and 3KHz sine wave and i plot them together

### 1KHz sine wave fft

![image](./assets/1kh_fft.png '1KHz sine wave fft')

### 3KHz sine wave fft

![image](./assets/3kh_fft.png '1KHz sine wave fft')

### 1KHz+3KHz sine wave fft

![image](./assets/1kh+3kh_fft.png '1KHz sine wave fft')

## Digit Classifier Pard B

### phone_1.wav fft

![image](./assets/phone_1_fft.png '1KHz sine wave fft')

### phone_2.wave fft

![image](./assets/phone_2_fft.png '1KHz sine wave fft')

### all digits on a specto

![image](./assets/spect.png '1KHz sine wave fft')
