
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
const int nLayers = 2;

//Neuron array should hold the number of neurons for each layer, each array element is a
//single input 
int Neurons_array[nLayers];


int *nNeurons = Neurons_array;


//setting up initial inputs
const int nInputs = 500;

double Array_of_0s_for_initial_inputs[nInputs];

double *pointer_to_array_of_0s = Array_of_0s_for_initial_inputs;

int main(int argc, char* argv[]){

        std::cout<<"Made it to the Start :)\n\n";

        //Opening the .dat file and the output file
        //in the final program this should be replaced with the mic inputs
        char *file="../../ecg_tests/ecg50hz.dat";
        FILE *finput = fopen(file,"rt");
	FILE *ferror = fopen("ecg_filtered.dat","wt");
        FILE *fnoise = fopen("ecg_50Hznoise.dat","wt");
        FILE *fout = fopen("ecg_output.dat","wt");
        FILE *finput_arrived = fopen("ecg_inputs.dat","wt");
        FILE *finput_array =fopen("ecg_input_array.dat","wt");

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
        net = new Net(nLayers,nNeurons,nInputs);


        //Initialises the network with: weights, biases and activation function
        // for Weights; W_Zeroes sets to 0 , W_Ones sets to 1 , W_random sets to a randome value
        // for Bias; B_None sets to , B_Random sets to a random value
        // for activations functions; Act_Sigmoid, Act_Tanh or Act_None
        net->initNetwork(Neuron::W_ONES, Neuron::B_NONE, Neuron::Act_Sigmoid);



        //Setting Learning Rate
        net -> setLearningRate(0.0015);



        //Setting up a variable that allows for access to read the final output of the network
        Layer *output_layer = net -> getLayer(nLayers-1);
        Neuron *output_neuron = output_layer ->getNeuron(0);
        int number_of_outputs = output_layer ->getnNeurons();


        //Getting variable that allows for access to input layer
        Layer *input_layer = net ->getLayer(0);
        Neuron *input_Neuron_0 = input_layer -> getNeuron(0);
        int number_of_inputs = input_layer ->getnNeurons();

        //variale to read the first input layer neuron
        double neuron_one_layer_one;


        std::cout<<"Number of Inputs:"<<number_of_inputs<<"\n";
        std::cout<<"Number of Outputs:"<<number_of_outputs<<"\n";
        std::cout<<"Number of Layers:"<<net->getnLayers()<<"\n";
        std::cout<<"Number of Total Neurons:"<<net->getnNeurons()<<"\n";
        std::cout<<"Neurons Array:";

        for(int i = 0;i<nLayers;i++){
            std::cout<<Neurons_array[i]<<",";
        }
    
        std::cout<<"\n";




        auto start = std::chrono::high_resolution_clock::now();


        for(int i=0;;i++) 
	    {


                //reading the input signal and generating the ref_noise
	        double input_signal;		
	        if (fscanf(finput,"%lf\n",&input_signal)<1) break;
	        double ref_noise = sin(2*M_PI/20*i);
        
                input_signal/=1000;
                //Updating the inputs to the network
                for(int i = nInputs-1;i>0;i--){
                Array_of_0s_for_initial_inputs[i] = Array_of_0s_for_initial_inputs[i-1];
                }

                Array_of_0s_for_initial_inputs[0] = ref_noise;
        
                net -> setInputs(Array_of_0s_for_initial_inputs);


                //propegating the sample forwards
                
                net ->propInputs();

                //storing output of the function and calculating error
                //output = net->getOutput(0);

                //error = input_signal - output;

                //Setting the backward error and updating weights

                error = net->setBackwardError_LMS(input_signal);

                //std::cout<<output_layer->getBackwardError(0)<<" ,"<<error<<"\n";

                //std::getc(stdin);

                net->propErrorBackward();

                //net ->updateWeights();

            
                //printf("%p  other value is %p #n",pointer_to_array_of_0s,Array_of_0s_for_initial_inputs);
                fprintf(ferror,"%f \n",error);
                fprintf(fnoise,"%f \n",ref_noise);
                fprintf(fout,"%f \n",output);
                fprintf(finput_arrived,"%f \n",input_signal);
                fprintf(finput_array,"%f \n",Array_of_0s_for_initial_inputs[nInputs-1]);
        
	        }

        auto elapsed = std::chrono::high_resolution_clock::now() - start;

        long long microseconds_taken = std::chrono::duration_cast<std::chrono::microseconds>(
        elapsed).count();

    
        std::cout<<"Time Taken:     "<<microseconds_taken<<"µs\n";

        fclose(finput);
	fclose(ferror);
        fclose(fout);
        fclose(fnoise);
        fclose(finput_arrived);
        fclose(finput_array);


        std::cout<<"Made it to the End :)\n\n\n";
    
    
}