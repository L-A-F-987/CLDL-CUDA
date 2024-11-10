import matplotlib.pyplot as plt
import numpy as np

Unfiltered = np.loadtxt('../build/ecg_tests/ecg_50Hznoise.dat')
filtered = np.loadtxt("../build/ecg_tests/ecg_filtered.dat")
output = np.loadtxt("../build/ecg_tests/ecg_output.dat")

plt.figure("Unfiltered")
plt.plot(Unfiltered)

plt.figure("Output")
plt.plot(output)

output = plt.figure("Filtered")
plt.plot(filtered)
plt.show()

