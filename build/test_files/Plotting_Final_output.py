import matplotlib.pyplot as plt
import numpy as np

Unfiltered = np.loadtxt('ecg50hz.dat')
filtered = np.loadtxt("ecg_filtered.dat")

plt.figure("Unfiltered")
plt.plot(Unfiltered)

output = plt.figure("output :Learning Rate 0.1")
plt.plot(filtered)
plt.savefig("output :Learning Rate 0.1.pdf")
plt.show()

