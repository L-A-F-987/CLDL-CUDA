
#include <chrono>

#include "cldl/Net.h"
#include "../jetsonGPIO.h"


#include <iostream>
#include <thread>



using namespace std; 


int main(int argc, char* argv[]){

    std::cout<<"Made it to the Start :)\n\n\n";



    //Trialing the GPIO Pins
    cout << "Testing the GPIO Pins" << endl;

    jetsonGPIO redLED = gpio165 ;
    jetsonGPIO pushButton = gpio166 ;
    gpioExport(pushButton) ;




    //Network Code

    constexpr int nLayers = 7;
    int nNeurons[nLayers] = {10,7,5,3,2,2,1};
    int* nNeuronsP = nNeurons;
    constexpr int nInputs = 500;

    double inputs[nInputs];
    for(int i = 0; i<=nInputs;i++){
        inputs[i] = i;
    };
    double* inputsP = inputs;

    //Creating Network

    Net *net;

    net = new Net(nLayers, nNeuronsP, nInputs);

    net->initNetwork(Neuron::W_RANDOM, Neuron::B_NONE, Neuron::Act_Sigmoid);


    //setting elements
    net->setInputs(inputsP);
    net->setLearningRate(0.1);
    net->setErrorCoeff(0,0,0,0,0,0);

    //running network

    int iterations = 10;
    double outputs[iterations]; 

    auto start = std::chrono::high_resolution_clock::now(); 

    for(int i = 0; i <=iterations;i++){
    net->setInputs(inputsP);
    net->propInputs();
    net->setBackwardError(0.1);
    net->propErrorBackward();
    net->updateWeights();
    outputs[i] = net ->getOutput(0);
    };

auto stop = std::chrono::high_resolution_clock::now(); 

auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)/1000;


    std::cout<<"First output:    "<<outputs[0]<<"\nFinal output:    "<<outputs[iterations-1]<<"\nTime Taken:    "<<duration.count()<<"\n\n\n";

    //getting general network information 

    int n = net->getnNeurons();
    int l = net->getnLayers();

    std::cout<<"number of neurons:  "<<n<<"\n";
    std::cout<<"number of layers:  "<<l<<"\n\n\n";
    



}