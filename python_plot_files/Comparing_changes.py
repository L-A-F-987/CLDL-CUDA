import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def average_calculation(x):

    average = np.average(x)

    return(average)


def read_all_files():
    file_names = glob('per_sample_outputs/*')

    print(len(file_names))

    array_per_sample_changed = np.zeros(int(len(file_names)/2))

    array_per_sample_original = np.zeros(int(len(file_names)/2))

    array_of_sizes_per_sample_changed = np.zeros(int(len(file_names)/2))



    file_names.sort()

    n_changes = 0
    n_original = 0
    
    for i in file_names:

        t = i.split("_")
        size = t[8].split(".")[0]

        if t[7] =="changes":
            array_of_sizes_per_sample_changed[n_changes] = int(size)
            n_changes += 1
    
    array_of_sizes_per_sample_changed.sort()

    for i in file_names:

        t = i.split("_")
        size = t[8].split(".")[0]

        if t[7] =="changes":
            array = np.loadtxt(i)
            av = average_calculation(array)/((10)**9)

            idx = (np.where(array_of_sizes_per_sample_changed == int(size)))[0][0]
            #print(idx)
            array_per_sample_changed[idx] = av

        if t[7] =="original":
            array = np.loadtxt(i)
            av = average_calculation(array)/((10)**9)

            idx = (np.where(array_of_sizes_per_sample_changed == int(size)))[0][0]
            #print(idx)
            array_per_sample_original[idx] = av
        



    return array_per_sample_changed,array_per_sample_original,array_of_sizes_per_sample_changed

average_changed,average_original,sizes_changed = read_all_files()

max_difference_time = max(average_original)/max(average_changed)
print(max_difference_time)

min_difference_time = average_original[1]/average_changed[1]
print(min_difference_time)

#Halfway point between maxes
pos_between_maxes = (max(average_original) - max(average_changed))/2


plt.subplot(1, 2, 1)
#plt.hlines(0.125,min(sizes_changed),max(sizes_changed),label = "maxium sampling interval",color = 'green',linestyle = "--")
plt.plot(sizes_changed[1:],average_changed[1:],label = "Final",marker = "o",color = 'firebrick')
plt.plot(sizes_changed[1:],average_original[1:],label = "Original",marker = "o",color = 'forestgreen')
plt.ylabel("Average Sampling Interval [ms]")
plt.xlabel("Nneurons in The Input Layer")
plt.title("Avg Sampling Interval of Original and Final Library")
plt.legend()


plt.subplot(1, 2, 2)
#plotting the sampling rate
#plt.hlines(8000,min(sizes_changed),max(sizes_changed),label = "minimum sampling rate",color = 'green',linestyle = "--")
plt.plot(sizes_changed[1:],1/average_changed[1:],label = "Final",marker = "o",color = 'firebrick')
plt.plot(sizes_changed[1:],1/average_original[1:],label = "Original",marker = "o",color = 'forestgreen')
plt.title("Avg Sampling Frequency of Original and Final Library")
plt.ylabel("Average Sampling Frequency [Hz]")
plt.xlabel("Nneurons in The Input Layer")
print(max(1/average_original[1:]))
print(max(1/average_changed[1:]))
plt.legend()
plt.show()



