import matplotlib.pyplot as plt
import numpy as np

def average_calculation(x,y):
    sum = 0
    for i in x:
        sum +=i
    
    to_s = 10**9

    sum = sum/to_s
    average = sum/(len(x))
    maximum = max(x)/to_s
    Minimum = min(x)/to_s

    print("Now Printing Per ",y,"\n")
    print("Average time taken per",y,":  ",average,"s")
    print("Max Time Taken",y,":  ",maximum,"s")
    print("Lowest Time Taken",y,":  ",Minimum,"s")
    print("Total Time Taken:",y,":  ",sum,"s")
    print("number of ",y,"per second if min:",1/Minimum)
    print("number of ",y,"per second average:",1/average,"\n")


    fig_per_sample = plt.figure()
    plt.hlines(average,0,len(x),linestyle = "--",color = "red",label = "average",linewidth = 2)
    plt.plot(x/to_s)
    plt.title(y)
    plt.legend()
    plt.show()

    return(average,maximum)

def avg_and_max_only(a):
    sum = 0
    for i in x:
        sum +=i
    
    to_s = 10**9

    sum = sum/to_s
    average = sum/(len(x))
    maximum = max(x)/to_s
    Minimum = min(x)/to_s

    return(average,maximum)




a = '../recorded_testing_results/edited/'

one_input = np.loadtxt(a+"edited_1_gpu_time_taken_per_sample.txt")
fifty_input = np.loadtxt(a+"edited_50_gpu_time_taken_per_sample.txt")
one_fifty_input = np.loadtxt(a+"edited_150_gpu_time_taken_per_sample.txt")
two_hundred_input = np.loadtxt(a+"edited_200_gpu_time_taken_per_sample.txt")

list_of_sizes = [1,50,150,200]

averages = np.ones(len(list_of_sizes))
maximums = np.ones(len(list_of_sizes))


print(averages)

averages[0],maximums[0] = average_calculation(one_input,"1")
averages[1],maximums[2]  = average_calculation(fifty_input,"50")
averages[2],maximums[2]  = average_calculation(one_fifty_input,"150")
averages[3],maximums[3]  = average_calculation(two_hundred_input,"200")



plt.title("Maximum Time")
plt.xlabel("Number of elements in the first layer")
plt.scatter(list_of_sizes,maximums)
plt.show()

plt.title("Average Time")
plt.xlabel("Number of elements in the first layer")
plt.scatter(list_of_sizes,maximums)
plt.show()




