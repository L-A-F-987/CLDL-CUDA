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
}

__host__ Net::~Net(){
    for (int i=0; i<nLayers; i++){
        delete layers[i];
    }
    delete[] layers;
    delete[] errorGradient;
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
    layers[0]->setInputs(inputs); //sets the inputs to the first layer only

}

__host__ void Net::propInputs() {
    for (int i=0;i<nLayers-1; i++) {
// Calculates the output to the given layer using a layer function
        layers[i]->calcOutputs();
        double* layerOutputs = layers[i]->getOutputs();
// Propagates the new outputs to the Input of the next layer
        layers[i+1]->propInputs(layerOutputs);
    }
    layers[nLayers-1]->calcOutputs();
}

//*************************************************************************************
//back propagation of error
//*************************************************************************************

__host__ void Net::setBackwardError(double _leadError){
    /* this is only for the final layer */
    theLeadError = _leadError;
    layers[nLayers-1]->setBackwardError(theLeadError);
}

__host__ void Net::propErrorBackward() {
    double* sumlist;
    for (int i = nLayers - 1; i > 0; i--) {
        sumlist = layers[i]->calcErrorWeightProductSum();
        layers[i-1]->propErrorBackward(sumlist);
    }
}

//*************************************************************************************
//learning:
//*************************************************************************************

__host__ void Net::updateWeights(){
    for (int i=nLayers-1; i>=0; i--){
        layers[i]->updateWeights();
    }
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

