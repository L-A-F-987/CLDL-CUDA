import matplotlib.pyplot as plt
import numpy as np

left = np.loadtxt('ch1.dat')
right = np.loadtxt("ch2.dat")

print(len(left))

plt.plot(left[:,1],label = "left")
plt.plot(right[:,1],label = "right")
plt.legend()
plt.show()