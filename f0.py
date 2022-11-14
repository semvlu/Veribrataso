import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# INPUT FILE!!!
y, sr = librosa.load("*.wav") 
S = np.abs(librosa.stft(y))
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

# funemental frequency
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C4'), fmax=librosa.note_to_hz('C6'))
times = librosa.times_like(f0)



fig, ax = plt.subplots()

# y-axis in linear scope, can be 'log'
img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
y_axis = 'linear', x_axis = 'time', ax=ax)

ax.plot(times, f0, label='f0', color='cyan', linewidth=3)

fig.colorbar(img, ax=ax, format="%+2.0f dB")

# set y-axis (frequency) bounds, extract the middle one as pivot
samp = int(len(f0) / 2)
ax.set_ylim(f0[samp]-100, f0[samp]+100)
plt.tight_layout()

# YOU NAME IT!!!
fig.savefig('*.png')
plt.show()