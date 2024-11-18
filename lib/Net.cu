
#include "cldl/Net.h"

//added by luca 
#include <helper_cuda.h>

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
    //printf("%i\n",layers[0]->nNeurons);
}

__host__ void Net::propInputs() {
    for (int i=0;i<nLayers-1; i++) {
// Calculates the output to the given layer using a layer function
        //[i]->calcOutputs_final_layer();
        layers[i]->calcOutputs(layers[i+1]->gpu_neurons,layers[i+1]->nNeurons);
// Propagates the new outputs to the Input of the next layer
        //layers[i+1]->propInputs(layers[i]->get_output_array_Pinned);
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
    double* sumlist;
    for (int i = nLayers - 1; i > 0; i--) {
        sumlist = layers[i]->calcErrorWeightProductSum();
        layers[i-1]->propErrorBackward(sumlist);
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

//added by luca gpu_set_inputs wth less blocks to launch, moved from layer level to test the graphs

__global__ void gpu_setInputs_less_blocks_net_level(Neuron* n, double *list, int nNeurons) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i = tid;i<nNeurons;i+=blockDim.x * gridDim.x){
        for(int j = threadIdx.x;j<nNeurons;j+=128){
            n[i].inputs[j] = list[j];
        }
    }
}

__global__ void gpu_calcOutputs_less_blocks_net_level(Neuron* neurons, int* layerHasReported, int nNeurons){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i = tid;i<nNeurons;i+=blockDim.x * gridDim.x){
        device_calcOutput(&neurons[i], layerHasReported);
    }
    __syncthreads();
}   

__global__ void gpu_getOutputs_net_level(Neuron* n, double* _outputs, int nNeurons){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid<nNeurons) {
        _outputs[tid] = *n[tid].output;
    }
}
//added by luca
//creating graph, see: https://developer.nvidia.com/blog/cuda-10-features-revealed/ for inspiration behind this

__host__ void Net::cudaGraph_generation_and_launch(){

    cudaStream_t stream1, stream2,streamForGraph;
    cudaEvent_t calcOutputs,getOutputs,set_inputs ;
    cudaGraphExec_t instance;
    cudaGraph_t graph;

    checkCudaErrors(cudaStreamCreate(&stream1));
    checkCudaErrors(cudaStreamCreate(&streamForGraph));

    checkCudaErrors(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

    

    //adding in prop Inputs

    for(int i=0;i<nLayers-1;i++){

    //getting the number of inputs for the given layer
        int nInputs_i =  layers[i]->nInputs;

        int B;

        if(nInputs_i>9){
            B = 8;
        }
        else{
            B = 8 * std::ceil(nInputs_i/8);
        }
        int T = 128;

        // Calculates the output to the given layer using a layer function
        gpu_calcOutputs_less_blocks_net_level<<<B, T, 0, stream1>>>(layers[i]->gpu_neurons,layers[i]->h_bPinned,layers[i]->nNeurons);
        cudaEventRecord(calcOutputs, stream1);


        //getting the output from layer i


        //cudaEventRecord(propInputs_Loop, stream1);
        cudaStreamWaitEvent(stream2, calcOutputs,0);
        gpu_getOutputs_net_level<<<B, T, 0, stream2>>>(layers[i]->gpu_neurons, layers[i]->get_output_array_Pinned, layers[i]->nNeurons);
        cudaEventRecord(getOutputs, stream2);

        //props the inputs to the next layers
        int nInputs_i_p_1 =  layers[i+1]->nInputs;


        if(nInputs_i_p_1>9){
            B = 8;
        }
        else{
            B = 8 * std::ceil(nInputs_i_p_1/8);
        }
        cudaStreamWaitEvent(stream1, getOutputs,0);
        gpu_setInputs_less_blocks_net_level<<<B, T, 0, stream1>>>(layers[i+1]->gpu_neurons,layers[i]->get_output_array_Pinned,nInputs_i_p_1);
        cudaEventRecord(set_inputs, stream1);

        
        }

    //cudaEventRecord(propInputs_Loop, stream1);
    gpu_calcOutputs_less_blocks_net_level<<<8, 128, 0, stream1>>>(layers[nLayers-1]->gpu_neurons,layers[nLayers-1]->h_bPinned,layers[nLayers-1]->nNeurons);
    

    checkCudaErrors(cudaStreamEndCapture(stream1, &graph));

    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

    cudaGraphLaunch(instance, streamForGraph);

    checkCudaErrors(cudaGraphDestroy(graph));

    
  }



