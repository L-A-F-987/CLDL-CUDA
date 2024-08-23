
#include <chrono>

#include "cldl/Net.h"

#include <iostream>
#include <stdio.h>
#include <thread>

#define _USE_MATH_DEFINES
#include <math.h>


using namespace std; 

//Creating an error and output variable for use later
double error;
double output;


//The nlayers should be an integer of the total number of hidden layers required not including the input layer
const int nLayers = 5;

//Neuron array should hold the number of neurons for each layer, each array element is a
//single input 
int Neurons_array[nLayers];


int *nNeurons = Neurons_array;


//setting up initial inputs
const int nInputs = 100;

double Array_of_0s_for_initial_inputs[nInputs];

double *pointer_to_array_of_0s = Array_of_0s_for_initial_inputs;






int main(int argc, char* argv[]){

    std::cout<<"Made it to the Start :)\n\n\n";



//Opening the .dat file and the output file
//in the final program this should be replaced with the mic inputs
    FILE *finput = fopen("ecg50hz.dat","rt");
	FILE *foutput = fopen("ecg_filtered.dat","wt");

//
//
//generating a network to be used

//Filling Neurons_array with some arbitray numbers to test network

for(int i =0; i<=nLayers;i++){
    Neurons_array[i]=i+1;


}

//Filling Input array with 0s array 

for(int i = 0; i<= nInputs;i++){
    Array_of_0s_for_initial_inputs[i] = 0;


}

//Varifying that the pointer points to the first element of the array
std::cout<<"Checking that the nNeurons pointed matches the values stored:\n";
std::cout<<"Memmory Address and value at Address    "<<nNeurons<<":     "<<*nNeurons<<"\n\n";


//Creating the Network 

Net *net;
net = new Net::Net(nLayers,nNeurons,nInputs);

//Initialises the network with: weights, biases and activation function
// for Weights; W_Zeroes sets to 0 , W_Ones sets to 1 , W_random sets to a randome value
// for Bias; B_None sets to , B_Random sets to a random value
//for activations functions; Act_Sigmoid, Act_Tanh or Act_None
net->initNetwork(Neuron::W_ONES, Neuron::B_NONE, Neuron::Act_Sigmoid);



//Setting all intial inputs to 0
net -> setInputs(pointer_to_array_of_0s);


//Setting Learning Rate
net -> setLearningRate(0.001);



//Setting up a variable n that allows for access to read the final output of the network
Layer *l = net -> getLayer(nLayers-1);
Neuron *n = l ->getNeuron(0);






auto start = std::chrono::high_resolution_clock::now();


    for(int i=0;;i++) 
	{
        //reading the input signal and generating the ref_noise
		double input_signal;		
		if (fscanf(finput,"%lf\n",&input_signal)<1) break;
		double ref_noise = sin(2*M_PI/20*i);

        //Updating the inputs to the network
        for(int i = 0;i<nInputs;i++){
        Array_of_0s_for_initial_inputs[i+1] = Array_of_0s_for_initial_inputs[i];
        }

        Array_of_0s_for_initial_inputs[0] = input_signal;

        net -> setInputs(pointer_to_array_of_0s);


        //propegating the sample forwards
        net ->propInputs();

        //storing output of the function and calculation error
        output = net ->getOutput(0);


        error = ref_noise-output;

        //Setting the backward error and updating weights
        net->setBackwardError(error);
        net->propErrorBackward();
        net ->updateWeights();

		fprintf(foutput,"%f \n",input_signal);
	}

    auto elapsed = std::chrono::high_resolution_clock::now() - start;

    long long microseconds_taken = std::chrono::duration_cast<std::chrono::microseconds>(
        elapsed).count();

    
    std::cout<<"Time Taken:     "<<microseconds_taken<<"µs\n";

    fclose(finput);
	fclose(foutput);

    //fprintf(stderr,"Written the filtered ECG to 'ecg_filtered.dat'\n");


    std::cout<<"Made it to the End :)\n\n\n";
    
    



}