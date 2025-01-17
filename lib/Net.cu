
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
    cudaMalloc((void**) &nNeurons_array, sizeof(int)*nLayers);

    nNeurons_array_pageable = _nNeurons;

    cudaMemcpy(nNeurons_array, nNeurons_array_pageable, sizeof(int)*nLayers, cudaMemcpyHostToDevice);

    //storing all neurons in a single array, note this only needed if you are going to use the single block method program

    cudaMalloc((void**) &gpu_neuron_pointers,sizeof(Neuron)*nNeurons);

    int index_for_memcpy = 0;
    for(int i = 0; i<nLayers;i+=1){
    cudaMemcpy(gpu_neuron_pointers+index_for_memcpy,layers[i]->gpu_neurons,sizeof(Neuron)*layers[i]->nNeurons,cudaMemcpyHostToDevice);
    index_for_memcpy += nNeurons;
    }

    //creating array that stores all of the start indexes and n_neurons_per_block

    /*
    int* _nNeurons_per_block_calcOut = new int[nLayers];
    int* _nNeurons_per_block_calcErrorWeight = new int[nLayers];

    int* _start_idx_calcOut = new int[nLayers];
    int* _start_idx_calcErrorWeight = new int[nLayers];

    //gpu_neuron_pointers array

    gpu_neuron_pointers = new Neuron*[nLayers];

    for(int i = 0; i<nLayers; i++){

    _start_idx_calcOut[i] = layers[i]->start_idx_for_reduction;
    _start_idx_calcErrorWeight[i] = layers[i]->start_idx_for_reduction_calcWeightProduct_sum;

    _nNeurons_per_block_calcOut[i] = layers[i]->number_of_concurrent_neurons_per_thread_block;
    _nNeurons_per_block_calcErrorWeight[i] = layers[i]->number_of_concurrent_neurons_per_thread_block_calcWeight_Product_sum;


    //storing the neuron pointers

    gpu_neuron_pointers[i] = layers[i]->gpu_neurons;

    //printf("gpu_pointers:%p\n",gpu_neuron_pointers + i);
    //printf("output at pointer 0:%p\n",gpu_neuron_pointers[i]);
    }

    cudaMalloc((void**)&start_idx_calcOut, sizeof(int)*nLayers) ; // host pinned
    cudaMalloc((void**)&start_idx_calcErrorWeight, sizeof(int)*nLayers) ; // host pinned
    cudaMalloc((void**)&nNeurons_per_block_calcOut, sizeof(int)*nLayers) ; // host pinned
    cudaMalloc((void**)&nNeurons_per_block_calcErrorWeight, sizeof(int)*nLayers) ; // host pinned

    cudaMemcpy(start_idx_calcOut, _start_idx_calcOut, sizeof(int)*nLayers, cudaMemcpyHostToDevice);
    cudaMemcpy(start_idx_calcErrorWeight, _start_idx_calcErrorWeight, sizeof(int)*nLayers, cudaMemcpyHostToDevice);
    cudaMemcpy(nNeurons_per_block_calcOut, _nNeurons_per_block_calcOut, sizeof(int)*nLayers, cudaMemcpyHostToDevice);
    cudaMemcpy(nNeurons_per_block_calcErrorWeight, _nNeurons_per_block_calcErrorWeight, sizeof(int)*nLayers, cudaMemcpyHostToDevice);
    */

    //*********End of Variables added for single block implementation */
}

__host__ Net::~Net(){
    for (int i=0; i<nLayers; i++){
        delete layers[i];

        //added by luca 
        //delete gpu_neuron_pointers[i];
    }
    delete[] layers;
    delete[] errorGradient;

    //added by luca 

    //delete[] gpu_neuron_pointers;

    cudaFree(nNeurons_array);

    cudaFree(start_idx_calcOut);
    cudaFree(start_idx_calcErrorWeight);
    cudaFree(nNeurons_per_block_calcOut);
    cudaFree(nNeurons_per_block_calcErrorWeight);

    //added Dec 20
    cudaFree(gpu_neuron_pointers);
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

    layers[0]->single_block_launch(  layers,  nLayers,  input_signal,  &error,  nNeurons_array,  nNeurons_per_block_calcOut,  nNeurons_per_block_calcErrorWeight,  start_idx_calcOut,  start_idx_calcErrorWeight,  gpu_neuron_pointers);

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

__host__ double Net::set_and_propErrorBackward_lms(double _input_signal ) {

    double input_signal = _input_signal;

    //setting error and proping the first layer 
    double error = layers[nLayers-1]->calcErrorWeightProductSum_LMS(layers[nLayers-2]->gpu_neurons,layers[nLayers-2]->inputs_a_Pinned,input_signal);
    //cudaDeviceSynchronize();
    for (int i = nLayers - 2; i > 0; i--) {
        layers[i]->calcErrorWeightProductSum(layers[i-1]->gpu_neurons,layers[i-1]->inputs_a_Pinned);
    }
    layers[0]->updateWeights_first_layer();

    return error;
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

