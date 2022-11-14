import librosa
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.fft import rfft, rfftfreq
import math
import glob
import os
from itertools import chain
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# scope of y range from 0 ~ 12000, meaning Prms / Pref
# dB = 20 * log(Prms / Pref)
f0 = []
os.chdir("C:/Users/David/Desktop/vibtxt/Wrist")

files = glob.glob('[AG]*_Wrist_*.txt')
for n in files:
  with open(n, 'r') as f:
        for line in f: 
            f0.append([float(x) for x in line.split()])

# delete 0 in f0
for i in range(len(f0)):
    f0[i] = [j for j in f0[i] if (j > 400 and j < 900)]


sr = 1
t = [] # len of f0[i]
for i in range(len(f0)):
    t.append(len(f0[i]))
N = [] #sr * t
for i in range(len(f0)):
    N.append( sr * t[i])

res = []
for i in range(len(f0)):
    res.append(fft(f0[i]))

print(len(res))
for i in range(len(res)):
    plt.plot(res[i])


plt.xlim([1, 1000])
plt.ylim([0, 4000])

plt.title('Vibrato Frequency')
plt.show()