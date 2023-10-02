import numpy as np
import matplotlib.pyplot as plt

# Deterioration parameters
alpha = 5 # a coefficient of the shape parameter
beta = 3.5 # the rate parameter (equals to 1/theta, where theta is the scale parameter)
# Time parameters
tStart = 0
tEnd = 100
# Sampling frequency window
fs = np.array([2, 5])
# Number of histories
nbHist = 100000

# Time vector:
dt = np.array([tStart])
while np.sum(dt)<tEnd:
    dt = np.append(dt, 1/np.random.uniform(low=fs[0], high=fs[1]))
t = np.cumsum(dt)

# Drawn of several paths (associated to the same time sampling)
# Shape parameter
k = alpha*(dt[1:].reshape(dt[1:].shape[0], 1)*np.ones((1, nbHist)))
# Drawn of increments
I = np.random.gamma(shape=k, scale=1/beta)
I = np.concatenate((np.zeros((1, nbHist)), I), axis=0)
# Degradation calculation
Y = np.cumsum(I, axis=0)

# Illustration
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
for id in range(np.minimum(75, nbHist)):
    ax.plot(t, Y[:, id], '.-', color=(0.8, 0.8, 0.8))
ax.plot(t, np.mean(Y, axis=1), '.-', color=(0, 0, 1))
ax.plot(t, alpha/beta*t, '.-', color=(1, 0, 0))
ax.set_xlabel('Time')
ax.set_ylabel('Degradation')
ax.set_title('Degradation with homogeneous gamma process')
ax.grid(True)
plt.tight_layout()
plt.show()
