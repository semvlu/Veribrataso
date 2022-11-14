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

os.chdir("C:/Users/David/Desktop/Good")

y = []
sr = []
t = []
N = []
xf = []
yf = []

files = librosa.util.find_files("C:/Users/David/Desktop/Good",
                                ext=['wav']) 
files = np.asarray(files)  

for n in files:
    y_tmp, sr_tmp = librosa.load(n)
    y.append(y_tmp)
    sr.append(sr_tmp)
    
    t_tmp = librosa.get_duration(y=y_tmp,sr=sr_tmp)
    t_tmp = int(t_tmp)
    t.append(t_tmp)

    N_tmp = sr_tmp * t_tmp
    N.append(N_tmp)
    yf.append(rfft(y_tmp))
    xf.append(rfftfreq(N_tmp,1/sr_tmp))

for i in range(len(y)):
    plt.plot(xf[i], np.abs(yf[i]))

plt.xlim([0,8000])
plt.title('Harmonics Response')
plt.show()

