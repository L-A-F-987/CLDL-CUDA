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


    fig = plt.figure()
    plt.hlines(average,0,len(x),linestyle = "--",color = "red",label = "average",linewidth = 2)
    plt.plot(x/to_s)
    plt.title(y)
    plt.legend()
    fig.savefig(y)
    plt.show()


    return()





GPU_Timer_per_sample = np.loadtxt('gpu_time_taken_per_sample.txt')
average_calculation(GPU_Timer_per_sample,"SAMPLE")

GPU_Timer_per_input_prop = np.loadtxt('gpu_time_taken_per_input_prop.txt')
average_calculation(GPU_Timer_per_input_prop,"INPUT PROP")

GPU_Timer_per_input_update = np.loadtxt('gpu_time_taken_per_input_update.txt')
average_calculation(GPU_Timer_per_input_update,"INPUT UPDATE")

GPU_Timer_per_error_update = np.loadtxt('gpu_time_taken_per_error_update.txt')
average_calculation(GPU_Timer_per_error_update,"ERROR UPDATE")

GPU_Timer_per_error_prop = np.loadtxt('gpu_time_taken_per_error_prop.txt')
average_calculation(GPU_Timer_per_error_prop,"ERROR PROP")

GPU_Timer_per_nInputs_neurons = np.loadtxt('gpu_nInputs_neuron.txt')
average_calculation(GPU_Timer_per_nInputs_neurons,"nInputs Neurons")


print("\n\n\nPrinting layer specific function times\n\n")

GPU_Timer_per_calc_outputs = np.loadtxt('Calc_outputs.txt')
average_calculation(GPU_Timer_per_calc_outputs,"Calc Output (Layer)")

GPU_Timer_per_allocate_int = np.loadtxt('allocate_int.txt')
average_calculation(GPU_Timer_per_allocate_int,"Allocate Int")

GPU_Timer_per_memcpy = np.loadtxt("gpu_memcpy_time.txt")
average_calculation(GPU_Timer_per_memcpy,"MemCpy Time")










