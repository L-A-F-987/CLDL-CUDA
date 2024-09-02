
#include <chrono>

#include "cldl/Net.h"

#include <iostream>
#include <stdio.h>
#include <thread>

#define _USE_MATH_DEFINES
#include <math.h>


using namespace std; 

//Creating an error and output variable for use later
    double error = 0;
    double output;


//The nlayers should be an integer of the total number of hidden layers required not including the input layer
    const int nLayers = 2;

//Neuron array should hold the number of neurons for each layer, each array element is a
//single input 
    int Neurons_array[nLayers];


    int *nNeurons = Neurons_array;


//setting up initial inputs
    const int nInputs = 1;

    double Array_of_0s_for_initial_inputs[nInputs];

    double *pointer_to_array_of_0s = Array_of_0s_for_initial_inputs;






    int main(int argc, char* argv[]){

    std::cout<<"Made it to the Start :)\n\n";



//Opening the .dat file and the output file
//in the final program this should be replaced with the mic inputs
    FILE *finput = fopen("ecg50hz.dat","rt");
	FILE *foutput = fopen("ecg_filtered.dat","wt");

//All files to store how long it takes for various gpu functions
    FILE *f_gpu_time_per_sample = fopen("gpu_time_taken_per_sample.txt","wt");

    FILE *f_gpu_time_per_input_update = fopen("gpu_time_taken_per_input_update.txt","wt");

    FILE *f_gpu_time_per_input_prop = fopen("gpu_time_taken_per_input_prop.txt","wt");

    FILE *f_gpu_time_per_error_update = fopen("gpu_time_taken_per_error_update.txt","wt");

    FILE *f_gpu_time_per_error_prop = fopen("gpu_time_taken_per_error_prop.txt","wt");





//generating a network to be used

//Filling Neurons_array with some arbitray numbers to test network
//Setting the output layer to be of size 1
    Neurons_array[0] = nInputs;
    Neurons_array[1] = 1;
    Neurons_array[nLayers-1] = 1;

//Filling Input array with 0s array 

    for(int i = 0; i<= nInputs;i++){
    Array_of_0s_for_initial_inputs[i] = 0;
    }   


//Varifying that the pointer points to the first element of the array



//Creating the Network 
    Net *net;
    net = new Net::Net(nLayers,nNeurons,nInputs);


//Initialises the network with: weights, biases and activation function
// for Weights; W_Zeroes sets to 0 , W_Ones sets to 1 , W_random sets to a randome value
// for Bias; B_None sets to , B_Random sets to a random value
// for activations functions; Act_Sigmoid, Act_Tanh or Act_None
    net->initNetwork(Neuron::W_ONES, Neuron::B_NONE, Neuron::Act_Sigmoid);



//Setting all intial inputs to 0
    net -> setInputs(pointer_to_array_of_0s);


//Setting Learning Rate
    net -> setLearningRate(0.001);



//Setting up a variable that allows for access to read the final output of the network
    Layer *output_layer = net -> getLayer(nLayers-1);
    Neuron *output_neuron = output_layer ->getNeuron(0);
    int number_of_outputs = output_layer ->getnNeurons();


//Getting variable that allows for access to input layer
    Layer *input_layer = net ->getLayer(0);
    Neuron *input_Neuron_0 = input_layer -> getNeuron(0);
    int number_of_inputs = input_layer ->getnNeurons();



//printing to files
    fprintf(foutput,"%f \n",output);

    fprintf(f_gpu_time_per_input_prop,"%i \n",int_gpu_time_taken_input_prop_total);

    fprintf(f_gpu_time_per_sample,"%i \n",gpu_timer_1_sample_time);

    fprintf(f_gpu_time_per_error_update,"%i \n",gpu_time_taken_per_update_error_total);

    fprintf(f_gpu_time_per_error_prop,"%i \n",gpu_time_taken_per_error_prop_total);

    



	}

    auto elapsed = std::chrono::high_resolution_clock::now() - start;

    long long microseconds_taken = std::chrono::duration_cast<std::chrono::microseconds>(
        elapsed).count();

    
    std::cout<<"Time Taken:     "<<microseconds_taken<<"µs\n";

    fclose(finput);
	fclose(foutput);
    fclose(f_gpu_time_per_sample);
    fclose(f_gpu_time_per_input_update);
    fclose(f_gpu_time_per_error_update);
    fclose(f_gpu_time_per_input_prop);
    fclose(f_gpu_time_per_error_prop);

//fprintf(stderr,"Written the filtered ECG to 'ecg_filtered.dat'\n");


    std::cout<<"Made it to the End :)\n\n\n";
    
    



}