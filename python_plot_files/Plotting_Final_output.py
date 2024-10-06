import matplotlib.pyplot as plt
import numpy as np

Unfiltered = np.loadtxt('../build/ecg_tests/ecg_50Hznoise.dat')
filtered = np.loadtxt("../build/ecg_tests/ecg_filtered.dat")

plt.figure("Unfiltered")
plt.plot(Unfiltered)

output = plt.figure("output :Learning Rate 0.1")
plt.plot(filtered)
plt.show()

