import librosa
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
# scope of y range from 0 ~ 12000, meaning Prms / Pref
# dB = 20 * log(Prms / Pref)

# INPUT FILE!!!
y, sr = librosa.load("*.wav") # MUST BE INT SECONDS
t = (librosa.get_duration(y=y, sr=sr))
t = int(t)
# Number of samples in normalized_tone
N = sr * t
yf = rfft(y)
xf = rfftfreq(N, 1 / sr)

plt.plot((xf), (np.abs(yf)))
plt.xlim([0, 8000])

plt.title('Harmonics Response')

plt.show()