#include <chrono>

#include "cldl/Net.h"

#include <iostream>
#include <stdio.h>
#include <thread>

#define _USE_MATH_DEFINES
#include <math.h>


//wav reading library 
#include "wav_reader.h"

using namespace std; 

//Creating an error and output variable for use later
    double error = 0;
    double output;


//The nlayers should be an integer of the total number of hidden layers required not including the input layer
    const int nLayers = 3;

//Neuron array should hold the number of neurons for each layer, each array element is a
//single input 
    int Neurons_array[nLayers];


    int *nNeurons = Neurons_array;


//setting up initial inputs (Max of 4.5k per layer since this crashes the jetson)
    const int nInputs = 3000;

    double Array_of_0s_for_initial_inputs[nInputs];

    double *pointer_to_array_of_0s = Array_of_0s_for_initial_inputs;


int main(int argc, char* argv[]) {


	std::cout<<"Made it to the Start :)\n\n";


//generating a network to be used

//Filling Neurons_array with some arbitray numbers to test network
//Setting the output layer to be of size 1
    Neurons_array[0] = nInputs;
    Neurons_array[1]= nInputs;
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


//variable to read the first input layer neuron

    std::cout<<"Number of Inputs:"<<number_of_inputs<<"\n";
    std::cout<<"Number of Outputs:"<<number_of_outputs<<"\n";
    std::cout<<"Number of Layers:"<<net->getnLayers()<<"\n";
    std::cout<<"Number of Total Neurons:"<<net->getnNeurons()<<"\n";
    std::cout<<"Neurons Array:";

    for(int i = 0;i<nLayers;i++){
    std::cout<<Neurons_array[i]<<",";
    }
    std::cout<<"\n";


//Wav file section
        char input[50];
        if (argc == 1)
		strcpy(input, "long_drill_16bit_int.wav");
        else
		strncpy(input, argv[1], 50);

	std::cout<<"\n\n\n`";

//initialising variables for the loop


	
	WAVread wavread;
	wavread.open(input);
	wavread.printHeaderInfo();
	FILE* f1 = fopen("ch1.dat","wt");
	FILE* f2 = fopen("ch2.dat","wt");
	long n = 0;

// variable for timing the while loop 

    auto time_taken_for_while_loop_start= std::chrono::high_resolution_clock::now();

	while (wavread.hasSample()) {

		WAVread::StereoSample s = wavread.getStereoSample();
		double t = (double)n / (double)(wavread.getFs());
		fprintf(f1,"%f %f\n",t,s.left);
		fprintf(f2,"%f %f\n",t,s.right);


//getting the input to the system 

        float input = s.left - s.right;

        for(int i = nInputs-1;i>0;i--){
        Array_of_0s_for_initial_inputs[i] = Array_of_0s_for_initial_inputs[i-1];
        }

        Array_of_0s_for_initial_inputs[0] = input;

//sum of left and right for comparison 

        float sum_left_right = s.left + s.right;

        net -> setInputs(pointer_to_array_of_0s);

        net ->propInputs();

//storing output of the function and calculation error

        output = net->getOutput(0);

        error = sum_left_right - output;

//setting and error prop

        net->setBackwardError(error);

        net->propErrorBackward();

        net ->updateWeights();

        break;


	}

    auto time_taken_for_while_loop_total = std::chrono::high_resolution_clock::now() - time_taken_for_while_loop_start;

    long long int time_taken_for_while_loop_print_micro = std::chrono::duration_cast<std::chrono::microseconds>(
        time_taken_for_while_loop_total).count();

    long long int time_taken_for_while_loop_print_seconds = std::chrono::duration_cast<std::chrono::seconds>(
        time_taken_for_while_loop_total).count();


    std::cout<<"end of Loop\n";
    std::cout<<"Time Taken:    "<<time_taken_for_while_loop_print_micro<<" microseconds\n";
    std::cout<<"Time Taken:    "<<time_taken_for_while_loop_print_seconds<<" seconds\n";
	fclose(f1);
	fclose(f2);
	wavread.close();
	return EXIT_SUCCESS;
}