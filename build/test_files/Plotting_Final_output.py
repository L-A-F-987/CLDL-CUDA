import matplotlib.pyplot as plt
import numpy as np

Unfiltered = np.loadtxt('ecg50hz.dat')
filtered = np.loadtxt("ecg_filtered.dat")

plt.figure("Unfiltered")
plt.plot(Unfiltered)

plt.figure("Filtered")
plt.plot(filtered)
plt.show()

