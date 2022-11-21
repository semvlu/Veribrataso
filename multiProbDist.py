import math
import glob
import os
from itertools import chain
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

f0 = []
os.chdir("C:/Users/David/Desktop/vibtxt/Good")

files = glob.glob('[A-G]*_Good_*.txt')
for n in files:
  with open(n, 'r') as f:
        for line in f: 
            f0.append([float(x) for x in line.split()])

"""
# 2D list to 1D
good = list(chain.from_iterable(good))
"""

# delete 0 in f0
for i in range(len(f0)):
    f0[i] = [j for j in f0[i] if (j > 400 and j < 900)]
#f0[2] = [i for i in f0[1] if(i > 600)]
#f0[3] = [i for i in f0[2] if(i > 600)]

r = 4
c = 4
fig, ax = plt.subplots(r,c)
cnt = 0

for i in range(r):
    for j in range(c):
        sns.histplot(ax=ax[i] [j],data=f0[cnt], kde=True, bins=100)
        cnt += 1
        if(cnt >= len(f0)):
            break
        

#ax[0][2].set_title('A5_0')
#ax[0][3].set_title('A5_1')

fig.suptitle("Good Probability Distribution", fontsize=16)
plt.show()

"""
good = [i for i in good if (i > 400 and i < 900)]
"""

"""
sum = 0.0
for i in range(0, len(f0)):
    sum += f0[int(i)]

# mean
mu = sum / len(f0)

# exp2 = E(x^2)
exp2 = 0
for i in range(0, len(f0)):
    exp2 += pow(f0[int(i)], 2)
#sigma
sigma = math.sqrt(exp2 / len(f0) - pow(mu, 2))
"""
