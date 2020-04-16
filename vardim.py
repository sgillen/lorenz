import numpy as np
import time

# Implements the variation estimators described here: https://arxiv.org/pdf/1101.1444.pdf
traj_name = "conv.npy"
X = np.load('./trajs/' + traj_name)
X = X[:,0].reshape(-1,1)
#X = X[-1000:]

order=1

def V(l,ord):
    return 1 / (2 * len(X) - l) * np.sum(np.linalg.norm(X[l:] - X[:-l],ord=ord))

D = 2 - 1/(order*np.log(2))*(np.log(V(2,order)) - np.log(V(1,order)))

start = time.time()
print("dim = ", D)
print(time.time() - start)
