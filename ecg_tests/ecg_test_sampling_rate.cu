
#include <chrono>

#include "cldl/Net.h"

#include <cuda_profiler_api.h>

#include <iostream>
#include <stdio.h>
#include <thread>

#define _USE_MATH_DEFINES
#include <math.h>


using namespace std; 

//The nlayers should be an integer of the total number of hidden layers required not including the input layer
const int nLayers = 3;

//Neuron array should hold the number of neurons for each layer, each array element is a
//single input 
int nNeurons[nLayers];

//setting up initial inputs
const int nInputs = 100;

double delay_line[nInputs];

int main(int argc, char* argv[]){

    std::cout<<"Made it to the Start :)\n\n";

    //Opening the .dat file and the output file
    //in the final program this should be replaced with the mic inputs
    FILE *finput = fopen("../../ecg_tests/ecg50hz.dat","rt");
    FILE *foutput = fopen("ecg_filtered.dat","wt");

    //
    //
    //generating a network to be used

    //Filling Neurons_array with some arbitray numbers to test network
    //Setting the output layer to be of size 1
    nNeurons[0] = nInputs;
    nNeurons[1] = nInputs/4;
    nNeurons[2] = 1;

    //Filling Input array with 0s array 

    for(int i = 0; i< nInputs;i++){
	delay_line[i] = 0;

    }


    //Varifying that the pointer points to the first element of the array
    //std::cout<<"Checking that the nNeurons pointed matches the values stored:\n";
    //std::cout<<"Memmory Address and value at Address    "<<nNeurons<<":     "<<*nNeurons<<"\n\n";


    //Creating the Network 

    Net *net;
    net = new Net(nLayers,nNeurons,nInputs);

    //Initialises the network with: weights, biases and activation function
    // for Weights; W_Zeroes sets to 0 , W_Ones sets to 1 , W_random sets to a randome value
    // for Bias; B_None sets to , B_Random sets to a random value
    //for activations functions; Act_Sigmoid, Act_Tanh or Act_None
    net->initNetwork(Neuron::W_RANDOM, Neuron::B_NONE, Neuron::Act_Sigmoid);

    //Setting all intial inputs to 0
    net -> setInputs(delay_line);

    //Setting Learning Rate
    net -> setLearningRate(0.1);

    //Setting up a variable that allows for access to read the final output of the network
    Layer *output_layer = net -> getLayer(nLayers-1);
    Neuron *output_neuron = output_layer ->getNeuron(0);
    int number_of_outputs = output_layer ->getnNeurons();


    //Getting variable that allows for access to input layer
    Layer *input_layer = net->getLayer(0);
    Neuron *input_Neuron_0 = input_layer->getNeuron(0);
    int number_of_inputs = input_layer->getnNeurons();

    std::cout << "Number of Inputs:"<<number_of_inputs<<"\n";
    std::cout << "Number of Outputs:"<<number_of_outputs<<"\n";
    std::cout << "Number of Layers:"<<net->getnLayers()<<"\n";
    std::cout << "Number of Total Neurons:"<<net->getnNeurons()<<"\n";
    std::cout << "Neurons Array:";

    
    for(int i = 0;i < nLayers;i++) {
	std::cout << nNeurons[i] << ",";
    }
    std::cout << "\n";

    auto start = std::chrono::high_resolution_clock::now();

    double fs = 1000; // Hz
    double noise_f = 50; //Hz
    double norm_noise_f = noise_f / fs;
    int nSamples = 0;

    for(int i=0;;i++) 
	{
	    //reading the input signal and generating the ref_noise
	    double input_signal;		
	    if (fscanf(finput,"%lf\n",&input_signal)<1) break;
	    nSamples++;
	    input_signal -= 2048.0;
	    input_signal /= 2048.0;

	    double ref_noise = sin(2*M_PI*norm_noise_f*(double)i);

	    //Updating the inputs to the network
	    for(int i = (nInputs-1); i > 0;i--){
		delay_line[i] = delay_line[i-1];
	    }

	    delay_line[0] = ref_noise;
        
	    net -> setInputs(delay_line);

	    //propegating the sample forwards
	    net ->propInputs();

	    //storing output of the function and calculation error
	    //double canceller = net->getOutput(0);

	    //double error = input_signal - canceller;

	    //Setting the backward error and updating weights
	    double error = net->setBackwardError_LMS(error);
	    net->propErrorBackward();
	    //net->updateWeights();

	    fprintf(foutput,"%f %f \n",error, input_signal);
	}

    auto elapsed = std::chrono::high_resolution_clock::now() - start;

    const auto seconds_taken = (double)(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count())/1E6;
    const double maxSamplingRate = nSamples / seconds_taken;

    printf("Time taken = %f s, max sampling rate = %f Hz\n", seconds_taken, maxSamplingRate);

    fclose(finput);
    fclose(foutput);

    //fprintf(stderr,"Written the filtered ECG to 'ecg_filtered.dat'\n");


    std::cout<<"Made it to the End :)\n\n\n";
    
    



}

