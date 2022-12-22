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
# cancel scientific notation
np.set_printoptions(suppress=True)

#--------------------------------
# do the regression or classifier
#--------------------------------

y = []
sr = []
t = []
N = []
xf = []
yf = []
mu = []

files = librosa.util.find_files("/Users/thethamespond/Desktop/Thumb",
                                ext=['wav']) 
files = np.asarray(files)  

for n in files:
    y_tmp, sr_tmp = librosa.load(n)
    f0_tmp, voiced_flag_tmp, voiced_probs_tmp = librosa.pyin(y_tmp, 
    fmin=librosa.note_to_hz('G4'), fmax=librosa.note_to_hz('B5'))


    y.append(y_tmp)
    sr.append(sr_tmp)
    
    t_tmp = librosa.get_duration(y=y_tmp,sr=sr_tmp)
    t_tmp = int(t_tmp)
    t.append(t_tmp)

    N_tmp = sr_tmp * t_tmp
    N.append(N_tmp)

    y_tmp = np.real(rfft(y_tmp))
    yf.append(np.abs(y_tmp))
    
    xf.append(rfftfreq(N_tmp,1/sr_tmp))


    # --------------------------------------
    f0_tmp = [i for i in f0_tmp if (i > 0)]
    f0_tmp = np.array(f0_tmp)
    f0_tmp = f0_tmp[~np.isnan(f0_tmp)]
    
    f0_note = [] # to note (A4-A5)
    for i in range(len(f0_tmp)):
        f0_note.append(librosa.hz_to_note(f0_tmp[i]))

    note_cnt = np.array([0,0,0,0,0,0,0,0]) # A4 B4 C5 D5 E5 F5 G#5 A5
    for i in range(len(f0_note)):
        if(f0_note[i] == 'A4'):
            note_cnt[0]+=1
        elif(f0_note[i] == 'B4'):
            note_cnt[1]+=1
        elif(f0_note[i] == 'C5'):
            note_cnt[2]+=1
        elif(f0_note[i] == 'D5'):
            note_cnt[3]+=1
        elif(f0_note[i] == 'E5'):
            note_cnt[4]+=1
        elif(f0_note[i] == 'F5'):
            note_cnt[5]+=1
        elif(f0_note[i] == 'G♯5'):
            note_cnt[6]+=1
        elif(f0_note[i] == 'A5'):
            note_cnt[7]+=1
    
    true_note = np.argmax(note_cnt)
    if(true_note == 0):
        true_pitch = librosa.note_to_hz('A4')
    elif(true_note == 1):
        true_pitch = librosa.note_to_hz('B4')
    elif(true_note == 2):
        true_pitch = librosa.note_to_hz('C5')
    elif(true_note == 3):
        true_pitch = librosa.note_to_hz('D5')
    elif(true_note == 4):
        true_pitch = librosa.note_to_hz('E5')
    elif(true_note == 5):
        true_pitch = librosa.note_to_hz('F5')
    elif(true_note == 6):
        true_pitch = librosa.note_to_hz('G♯5')
    elif(true_note == 7):
        true_pitch = librosa.note_to_hz('A5')

    f0_tmp = [i for i in f0_tmp if (i > true_pitch - 20 and i < true_pitch + 20)]
    f0_tmp = np.array(f0_tmp)

    mu.append(f0_tmp.mean())
    # --------------------------------------

"""
r = 4
c = int(len(files) / 4)

fig, ax = plt.subplots(r,c)
cnt = 0

for i in range(r):
    for j in range(c):
        axes = ax[i][j]
        axes.plot(xf[cnt], 20*np.log10(yf[cnt]) ) 
        #axes.set_xlim([0, mu[cnt] * 6 + 100])
        #axes.set_ylim([0,80])
        cnt += 1
        if(cnt >= len(files)):
            break

fig.suptitle('Harmonics Response (Fingertip')
plt.show()
"""



#----------------------------------------------------------

# delete idx +-1200 for max
fund = np.array([[0, 0, 0]]) # f0 [pitch, amp, index]
harm = np.array([[0, 0, 0]]) # harmonics [pitch, amp, index]


# get f0, it's possible that the harmonic is greater than f0
# len(xf) & len(yf)=66151, max(xf)~=11025.0, 1 Hz occupies about 6 elements
for i in range(len(xf)):
    xf_tmp = np.array(xf[i])
    yf_tmp = np.array(yf[i])
    idx = np.argmax(yf_tmp)

    # f0 must < 900
    while (xf[i][idx] > 900):
        for j in range(idx-1200, idx+1200):
            xf_tmp[j] = 0
            yf_tmp[j] = 0
        idx = np.argmax(yf_tmp)

    #re_idx = np.where(xf[i] == xf[i])
    fund = np.append(fund, [[ xf[i][idx],yf[i][idx], idx ]], axis=0)
# delete the 1st row
fund = np.delete(fund, 0, axis=0)


# get the next 5 harmonics & their amplitude
for i in range(5):
    for j in range(len(xf)): #16 
        idx = int(fund[j][2]) * (i+2) # fund[j][2]: index of the f0, j: file order, (i+2)th harmonics
        harm = np.append(harm, [[ xf[i][idx], yf[i][idx], idx ]], axis=0)
# delete the 1st row
harm = np.delete(harm, 0, axis=0)


seq = np.array([[0,0,0,0,0,0]]) # seq: [f0, f1, f2, f3, f4, f5], final size: 16x6

for i in range(len(xf)):
    seq = np.append(seq, [[ fund[i][1], harm[i][1], harm[i+16][1], 
                            harm[i+32][1], harm[i+48][1], harm[i+64][1] ]], axis=0)
seq = np.delete(seq, 0, axis=0)
print(seq)
# Normalize, set f5 = 1 and calculate the ratio
for i in range(len(xf)):
    for j in range(6):
        seq[i][j] /= min(seq[i])

# delete f0
#seq = np.delete(seq, 0, axis=1)


seqcnt = [0, 1, 2, 3, 4, 5]
r = 4
c = int(len(files) / 4)

fig, ax = plt.subplots(r,c)
cnt = 0
for i in range(r):
    for j in range(c):
        axes = ax[i][j]
        axes.scatter(seqcnt, 20*np.log10( seq[cnt] / min(seq[cnt] )))
        cnt += 1
        if(cnt >= len(files)):
            break
fig.suptitle('Harmonics Ratio (Thumb')

plt.show()