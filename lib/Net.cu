
#include "cldl/Net.h"

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <ctgmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <iostream>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <numeric>
#include <vector>

using namespace std;

__host__ Net::Net(int _nLayers, int* _nNeurons, int _nInputs) {
    nLayers = _nLayers; //no. of layers including inputs and outputs layers
    layers = new Layer*[nLayers];
    int* nNeuronsp = _nNeurons; //no. of neurons in each layer (note this is an array)
    nInputs=_nInputs;   // the no. of inputs to the network (i.e. the first layer)

    int nInput = 0; //temporary variable to use within the scope of for loop
    for (int i=0; i<nLayers; i++){
        int numNeurons= *nNeuronsp; //no. neurons in this layer
        if (i==0){nInput=nInputs;}
        /* no. inputs to the first layer is equal to no. inputs to the network */
        layers[i]= new Layer(numNeurons, nInput);
        nNeurons += numNeurons;
        nWeights += (numNeurons * nInput);
        nInput=numNeurons;
        /*no. inputs to the next layer is equal to the number of neurons
         * in the current layer. */
        nNeuronsp++; //point to the no. of neurons in the next layer
    }
    nOutputs=layers[nLayers-1]->getnNeurons();
    errorGradient= new double[nLayers];


    //added by luca, used to store all neurons
    cudaMalloc( (void**) &all_Neurons, sizeof(Neuron)*nNeurons);
    cudaMalloc( (void**) &neurons_each_layer, sizeof(int)*nLayers);
}

__host__ Net::~Net(){
    for (int i=0; i<nLayers; i++){
        delete layers[i];
    }
    delete[] layers;
    delete[] errorGradient;

    cudaFree(all_Neurons);
}

__host__ void Net::initNetwork(Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am){
    for (int i=0; i<nLayers; i++){
        layers[i]->initLayer(i, _wim, _bim, _am);
    }

}

__host__ void Net::setLearningRate(double _learningRate){
    learningRate=_learningRate;
    for (int i=0; i<nLayers; i++){
        layers[i]->setlearningRate(learningRate);
    }
}

__host__ void Net::setErrorCoeff(double _globalCoeff, double _backwardsCoeff,
                                 double _midCoeff, double _forwardCoeff,
                                 double _localCoeff, double  _echoCoeff) {
    for (int i=0; i<nLayers; i++){
        layers[i]->setErrorCoeff(_globalCoeff, _backwardsCoeff, _midCoeff,
                                 _forwardCoeff, _localCoeff, _echoCoeff);
    }
}

// this is only for testing
__host__ void Net::setWeights(double* _weightsList) {
    for (int i=0;i<nLayers;i++) {
        layers[i]->setWeights(_weightsList);
    }
}

//*************************************************************************************
//forward propagation of inputs:
//*************************************************************************************

__host__ void Net::setInputs(double* _inputs){
    inputs=_inputs;
    layers[0]->setInputs_layer_level_only(inputs); //sets the inputs to the first layer only
}



__host__ double Net::single_block_lms(double* _inputs,double input_signal){
    inputs = _inputs;
    double error = 0;

    //setting inputs with memcpy
    layers[0]->setInputs_layer_level_only(inputs);

    layers[0]->single_block_launch(layers,nLayers,input_signal,&error);

    return(error);
}

__host__ void Net::propInputs() {
    for (int i=0;i<nLayers-1; i++) {
// Calculates the output to the given layer using a layer function

        layers[i]->calcOutputs(layers[i+1]->inputs_a_Pinned,layers[i+1]->nNeurons);
    }
    layers[nLayers-1]->calcOutputs_final_layer();
}


//*************************************************************************************
//back propagation of error
//*************************************************************************************

__host__ void Net::setBackwardError(double _leadError){
    /* this is only for the final layer */
    theLeadError = _leadError;
    layers[nLayers-1]->setBackwardError(theLeadError);
}

__host__ double Net::setBackwardError_LMS(double _input_signal){
    /* this is only for the final layer */
    double input_signal = _input_signal;
    double error = layers[nLayers-1]->setBackwardError_LMS(input_signal);
    return(error);

}

__host__ void Net::propErrorBackward() {
    for (int i = nLayers - 1; i > 0; i--) {
        layers[i]->calcErrorWeightProductSum(layers[i-1]->gpu_neurons,layers[i-1]->inputs_a_Pinned);
    }
    layers[0]->updateWeights_first_layer();
}

//*************************************************************************************
//learning:
//*************************************************************************************

__host__ void Net::updateWeights(){
    //for (int i=nLayers-1; i>=0; i--){
    //   layers[i]->updateWeights();
    //}
    layers[0]->updateWeights_first_layer();
    
}

//single block launch code
__global__ void gpu_single_block_launch(double* input_array, double input_signal,int nLayers,int* neurons_each_layer,Neuron* all_Neurons){

    //prop inputs

    //set and calc error

    //prop error backward
}

//added by luca

__host__ void Net::storing_all_neurons_in_array_for_single_block_launch(){

    int starting_index = 0;
    for(int i = 0;i<nLayers;i++){

        int num_neurons = layers[i]->nNeurons;
        
        cudaMemcpy(all_Neurons+starting_index,layers[i]->gpu_neurons, sizeof(Neuron)*num_neurons,cudaMemcpyHostToDevice);

        cudaMemcpy(neurons_each_layer+i,&num_neurons, sizeof(int),cudaMemcpyHostToDevice);

        starting_index += num_neurons;
    }
    
}

__host__ double Net::single_block_launch(double* input_array, double input_signal){

    double error;

    //set input layer
    memcpy(layers[0]->inputs_a_Pinned,layers[0]->inputs_a_Pageable,sizeof(double)*nInputs);

    gpu_single_block_launch<<<1,128>>>(input_array,input_signal,nLayers,neurons_each_layer,all_Neurons);
    cudaDeviceSynchronize();

    return error;

    
}



//*************************************************************************************
// getters:
//*************************************************************************************

__host__ int Net::getnLayers(){
    return (nLayers);
}

__host__ int Net::getnNeurons(){
    return (nNeurons);
}

__host__ int Net::getnInputs(){
    return (nInputs);
}

__host__ int Net::getnOutputs(){
    return (nOutputs);
}

__host__ Layer* Net::getLayer(int _layerIndex){
    assert(_layerIndex<nLayers);
    return (layers[_layerIndex]);
}

__host__ double Net::getOutput(int _neuronIndex) {
    return layers[nLayers-1]->getOutput(_neuronIndex);
}

//added by luca 
__host__ double Net::getOutput_no_memcpy(int _neuronIndex){
    return layers[nLayers-1]->getOutput_no_memcpy(_neuronIndex);
}


__host__ void Net::printInitialWeights() {
    FILE* initweights = nullptr;
    initweights = fopen("initial_weights.tsv", "wt");
    for (int i=0; i<nLayers;i++) {
        layers[i]->printWeights(initweights);
    }
    fclose(initweights);
}

__host__ void Net::printWeights() {
    FILE* weights = nullptr;
    weights = fopen("updated_weights.tsv", "wt");
    for (int i=0; i<nLayers;i++) {
        layers[i]->printWeights(weights);
    }
    fclose(weights);
}

