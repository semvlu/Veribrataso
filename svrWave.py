import math
import glob
import os
from itertools import chain
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

def dtw(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n+1, m+1))
    
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s[i-1] - t[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix

f0 = []

os.chdir("C:/Users/David/Desktop/vibtxt/Good")

files = glob.glob('[A-G]*_Good_*.txt')
for n in files:
  with open(n, 'r') as f:
        for line in f: 
            f0.append([float(x) for x in line.split()])

for i in range(len(f0)):
    f0[i] = [j for j in f0[i] if (j > 400 and j < 900)]
#for i in range(2,4):
#    f0[i] = [j for j in f0[i] if (j > 600 and j < 900)]


X = []
for i in range (len(f0)):
    X.append(np.array(f0[i]))

y = []
for i in range (len(f0)):
    y.append(np.sin(X[i]).ravel())

feat = []
for i in range(len(f0)):
    feat.append(dtw(X[i], y[i]))

svr_rbf = SVR(kernel="rbf", C=1, gamma='auto', epsilon=0.1)
lw = 2

r=4
c=4
fig, ax = plt.subplots(r, c, sharey=True)

cnt = 0
for i in ax:
    for j in i:
        j.plot(X[cnt],
            svr_rbf.fit(X[cnt].reshape(-1,1), y[cnt]).predict(X[cnt].reshape(-1,1)),
            lw=lw, color='teal')

        j.scatter(
            X[cnt][svr_rbf.support_],
            y[cnt][svr_rbf.support_],
            facecolor="none", edgecolor='red',s=50,label="RBF support vec")

        j.scatter(
            X[cnt][np.setdiff1d(np.arange(len(X[cnt])), svr_rbf.support_)],
            y[cnt][np.setdiff1d(np.arange(len(X[cnt])), svr_rbf.support_)],
            facecolor="none", edgecolor="k",s=50,label="other training data")

        cnt += 1
        if(cnt >= len(f0)):
            break

#ax[0][2].set_title('A5_0')
#ax[0][3].set_title('A5_1')

ax[0][0].legend(bbox_to_anchor=(0.2, 1.6),ncol=1,fancybox=True,shadow=True)
fig.text(0.5, 0.04, "data", ha="center", va="center")
fig.text(0.06, 0.5, "target", ha="center", va="center", rotation="vertical")
fig.suptitle("Support Vector Regression (Good)", fontsize=14)
plt.show()







def dtw(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n+1, m+1))

    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s[i-1] - t[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix