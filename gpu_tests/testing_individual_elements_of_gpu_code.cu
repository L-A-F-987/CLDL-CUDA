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
    const int nLayers = 200;

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


//All files to store how long it takes for various gpu functions

        
        FILE *f_prop_inputs = fopen("prop_inputs.txt","wt");
        FILE *f_calc_output = fopen("Calc_outputs.txt","wt");
        FILE *f_allocate_input = fopen("allocate_int.txt","wt");
        FILE *f_gpu_calc_outputs = fopen("gpu_calc_output.txt","wt");
        FILE *f_gpu_memcpy = fopen("gpu_memcpy_time.txt","wt");
        FILE *f_gpu_nInputs_neuron = fopen("gpu_nInputs_neuron.txt","wt");
        FILE *f_gpu_dot_product = fopen("gpu_dot_product.txt","wt");





//generating a network to be used

//Filling Neurons_array with some arbitray numbers to test network
//Setting the output layer to be of size 1
        Neurons_array[0] = nInputs;
        Neurons_array[nLayers-1] = 1;

//Filling Input array with 0s array 

        for(int i = 0; i<= nInputs;i++){
        Array_of_0s_for_initial_inputs[i] = 0;
        }   


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

        for(int i = 0; i < 10000;i++){

//getting the time for calculating the outputs
            auto calc_outputs_start = std::chrono::high_resolution_clock::now();

            input_layer->calcOutputs();

            auto calc_outputs_total = std::chrono::high_resolution_clock::now() - calc_outputs_start;

            fprintf(f_calc_output,"%i \n",calc_outputs_total);

//getting times for allocating an integer 

            int * testlayerHasReported;

            auto allocate_int_start = std::chrono::high_resolution_clock::now();

            gpu_allocateInt(&testlayerHasReported,0);

            auto allocate_int_total = std::chrono::high_resolution_clock::now() - allocate_int_start;
    
            fprintf(f_allocate_input,"%i \n",allocate_int_total);

//Testing cudaMemcpy() time taken

            auto cudaMemcpy_int_start = std::chrono::high_resolution_clock::now();

            cudaMemcpy(testlayerHasReported, &testlayerHasReported, sizeof(int), cudaMemcpyHostToDevice);

            auto cudaMemcpy_int_total = std::chrono::high_resolution_clock::now() - cudaMemcpy_int_start;

            fprintf(f_gpu_memcpy,"%i \n",cudaMemcpy_int_total);

//nInputs time taken 

            auto nInputs_time_start = std::chrono::high_resolution_clock::now();

            int* nInputs = output_neuron -> nInputs;

            auto nInputs_time_total = std::chrono::high_resolution_clock::now() - nInputs_time_start;

            fprintf(f_gpu_nInputs_neuron,"%i \n",nInputs_time_total);


        }


// Replicating the dot product code for testing 


    
    
        fclose(f_prop_inputs);
        fclose(f_calc_output);
        fclose(f_allocate_input);
        fclose(f_gpu_calc_outputs);
        fclose(f_gpu_memcpy);
        fclose(f_gpu_nInputs_neuron);
        fclose(f_gpu_dot_product);
    }


