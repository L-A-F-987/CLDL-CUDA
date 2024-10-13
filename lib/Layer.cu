#include "cldl/Layer.h"

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
#include <fstream>

#define MAX_BLOCKSIZE 512


// GPU FUNCTIONS //

__global__ void gpu_setLearningRate(Neuron* n, double _learningRate, int nNeurons) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<nNeurons)
        device_setLearningRate(&n[i], _learningRate);
}

__global__ void gpu_setInputs(Neuron* n, double *list, int nNeurons) {
    int i = threadIdx.x; // Input index
    int j = (blockIdx.x*blockDim.y) + threadIdx.y; // Neuron index
    if(j < nNeurons)
        n[j].inputs[i] = list[i];
}

__global__ void gpu_setErrorCoeff(Neuron *n, double _globalCoeff, double _backwardsCoeff,
                                  double _midCoeff, double _forwardCoeff,
                                 double _localCoeff, double _echoCoeff, int nNeurons) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<nNeurons) {
        *n[i].backwardsCoeff = _backwardsCoeff;
        *n[i].midCoeff = _midCoeff;
        *n[i].forwardCoeff = _forwardCoeff;
        *n[i].globalCoeff = _globalCoeff;
        *n[i].localCoeff = _localCoeff;
        *n[i].echoCoeff = _echoCoeff;
    }
}

__global__ void gpu_setWeights(Neuron* n, double *list, int nNeurons) {
    int i = threadIdx.x; // Input index
    int j = (blockIdx.x*blockDim.y) + threadIdx.y; // Neuron index
    if(j < nNeurons)
        n[j].weights[i] = list[i];
}

__global__ void gpu_setBackwardError(Neuron*n, double _leadBackwardError, int nNeurons) {
    double leadBackwardError = _leadBackwardError;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<nNeurons)
        device_setBackwardError(leadBackwardError, &n[i]);
}

__global__ void gpu_calcErrorWeightProductSum(Neuron* n, int nNeurons, int nInputs, double* sumlist) {
    int i = threadIdx.x; // Input index
    int j = (blockIdx.x*blockDim.y) + threadIdx.y; // Neuron index

    if(i < nInputs && j < nNeurons)
        n[j].ErrorWeightProducts[i] = n[j].weights[i] * (*n[j].backwardError);
    __syncthreads();

    if (j == 0) {
        double sum = 0.0;
        for (int a = 0; a < nNeurons; a++) {
            sum += n[a].ErrorWeightProducts[i];
        }
        sumlist[i] = sum;
    }
}

/*__global__ void gpu_setForwardError(Neuron*n, double _leadForwardError) {
    int i = threadIdx.x;
    *n[i].forwardError = _leadForwardError;
}*/

__global__ void gpu_calcOutputs(Neuron* neurons, int* layerHasReported){
    device_calcOutput(&neurons[blockIdx.x]);
    __syncthreads();
    device_calcOutputCont(&neurons[blockIdx.x], layerHasReported);
    __syncthreads();
}

__global__ void gpu_propErrorBackwards(Neuron *n, double* _sumList, int nNeurons) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double* sumList = _sumList;
    if (i<nNeurons)
        device_propErrorBackward(sumList[i], &n[i]);
}

__global__ void gpu_updateWeights(Neuron *n, int nNeurons){
    int i = threadIdx.x;    //Input index
    int j = (blockIdx.x*blockDim.y) + threadIdx.y;  //Neuron index
    //double force = 1;
    if (j<nNeurons) {
        n[j].weights[i] += (*n[j].learningRate) * n[j].inputs[i] * (*n[j].backwardError); // * force;
    }
}

__global__ void gpu_getOutputs(Neuron* n, double* _outputs, int nNeurons){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<nNeurons) {
        _outputs[i] = *n[i].output;
    }
}

/*

Added by luca, Purpose is to implement pinned memory to skip the pageable memeory used
by the Layer::CalcOutputs funtion

For beckground on this See: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/ 

*/

//General function to check the allocation of the memory was successful
inline
cudaError_t Layer::checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}


__host__ int* Layer::generating_pinned_memory_address(){


    const unsigned int bytes =  sizeof(int);


    //int *h_aPageable, *h_bPageable;   
    //int *h_aPinned, *h_bPinned;


    h_aPageable = (int*)malloc(bytes);                   
    h_bPageable = (int*)malloc(bytes);

    checkCuda( cudaMallocHost((void**)&h_aPinned, bytes) ); // host pinned
    checkCuda( cudaMallocHost((void**)&h_bPinned, bytes) ); // host pinned


    return(0);
}



//end of code added by Luca

// HOST FUNCTIONS //

__host__ Layer::Layer(int _nNeurons, int _nInputs){
    nNeurons = _nNeurons; // number of neurons in this layer
    nInputs = _nInputs; // number of inputs to each neuron


    neurons = (Neuron*) (malloc(sizeof(Neuron) * nNeurons));
    for (int i=0; i<nNeurons; i++){
        Neuron* j = new Neuron(nInputs);
        neurons[i] = *j;
    }

    cudaMalloc((void**) &gpu_sumlist, sizeof(double)*_nInputs);
    cudaMalloc((void**) &gpu_weights, sizeof(double)*nInputs);
    cudaMalloc( (void**) &gpu_inputs, sizeof(double)*nInputs);
    cudaMalloc( (void**) &gpu_neurons, sizeof(Neuron)*nNeurons);
    cudaMemcpy(gpu_neurons, neurons, sizeof(Neuron)*nNeurons, cudaMemcpyHostToDevice);

//added by luca 

    generating_pinned_memory_address();

}

__host__ Layer::~Layer(){
    for(int i=0;i<nNeurons;i++) {
        neurons[i].~Neuron();
    }
    free(neurons);
    cudaFree(gpu_inputs);
    cudaFree(gpu_neurons);

//added by Luca 
//freeing some of the memory that was not freed in the original program
    cudaFreeHost(gpu_inputs);
    cudaFreeHost(gpu_weights);
    cudaFreeHost(gpu_neurons);
    cudaFreeHost(gpu_sumlist);

//Added by Luca
//Freeing Pinned and Pagable Memory
    cudaFreeHost(h_aPinned);
    cudaFreeHost(h_bPinned);
    free(h_aPageable);
    free(h_bPageable);
}

//*************************************************************************************
//initialisation:
//*************************************************************************************

__host__ void Layer::initLayer(int _layerIndex, Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am){
    myLayerIndex = _layerIndex;
    for (int i=0; i<nNeurons; i++){
        neurons[i].initNeuron(i, myLayerIndex, _wim, _bim, _am);
    }
}

__host__ void Layer::setlearningRate(double _learningRate){
    int B = std::ceil(float(nNeurons)/MAX_BLOCKSIZE);   // Total number of blocks required
    int T = MAX_BLOCKSIZE;
    if (nNeurons<MAX_BLOCKSIZE){
        T = nNeurons;
    }
    //printf("%d, %d\n", B, T);
    learningRate=_learningRate;
    gpu_setLearningRate<<<B,T>>>(gpu_neurons, learningRate, nNeurons);
    cudaDeviceSynchronize();
}

__host__ void Layer::setErrorCoeff(double _globalCoeff, double _backwardsCoeff,
                            double _midCoeff, double _forwardCoeff,
                            double _localCoeff, double  _echoCoeff) {
    int B = std::ceil(float(nNeurons)/MAX_BLOCKSIZE);   // Total number of blocks required
    int T = MAX_BLOCKSIZE;
    if (nNeurons<MAX_BLOCKSIZE){
        T = nNeurons;
    }
    gpu_setErrorCoeff<<<B,T>>>(gpu_neurons, _globalCoeff, _backwardsCoeff,
                                      _midCoeff, _forwardCoeff, _localCoeff, _echoCoeff, nNeurons);
    cudaDeviceSynchronize();
}

//this method is for testing only
__host__ void Layer::setWeights(double* _weightsList) {
    cudaMemcpy(gpu_weights, _weightsList, sizeof(double)*nInputs,cudaMemcpyHostToDevice);
    int nThreads = nInputs * nNeurons;          // Total number of CUDA threads required
    int blockYDim = MAX_BLOCKSIZE/nInputs;      // Size of a block's Y dimension
    int blockSize = nInputs * blockYDim;        // Size of required block
    int B = std::ceil(float(nThreads)/blockSize);   // Total number of blocks required
    dim3 T = dim3(nInputs, blockYDim);          // 2D block dimensions
    gpu_setWeights<<<B,T>>>(gpu_neurons, gpu_weights, nNeurons);
}

//*************************************************************************************
//forward propagation of inputs:
//*************************************************************************************

__host__ void Layer::setInputs(double *_inputs) {
    inputs = _inputs;
    cudaMemcpy(gpu_inputs, inputs, sizeof(double)*nInputs,cudaMemcpyHostToDevice);

    int nThreads = nInputs * nNeurons;          // Total number of CUDA threads required
    int blockYDim = MAX_BLOCKSIZE/nInputs;      // Size of a block's Y dimension
    int blockSize = nInputs * blockYDim;        // Size of required block
    int B = std::ceil(float(nThreads)/blockSize);   // Total number of blocks required
    dim3 T = dim3(nInputs, blockYDim);          // 2D block dimensions
    gpu_setInputs<<<B,T>>>(gpu_neurons, gpu_inputs, nNeurons);

    cudaDeviceSynchronize();
}

__host__ void Layer::propInputs(double* _gpu_InputOutputs) {
    int nThreads = nInputs * nNeurons;          // Total number of CUDA threads required
    int blockYDim = MAX_BLOCKSIZE/nInputs;      // Size of a block's Y dimension
    int blockSize = nInputs * blockYDim;        // Size of required block
    int B = std::ceil(float(nThreads)/blockSize);   // Total number of blocks required
    dim3 T = dim3(nInputs, blockYDim);          // 2D block dimensions
    gpu_setInputs<<<B,T>>>(gpu_neurons, _gpu_InputOutputs, nNeurons);
    cudaDeviceSynchronize();
}

__host__ void Layer::calcOutputs(){

    //block id gets neuron

//original code for testing
/*
    int* _layerHasReported;
    gpu_allocateInt(&_layerHasReported, 0);
    cudaMemcpy(_layerHasReported, &layerHasReported, sizeof(int), cudaMemcpyHostToDevice);
    gpu_calcOutputs<<<nNeurons, 1>>>(gpu_neurons, _layerHasReported);
    cudaDeviceSynchronize();
    cudaMemcpy(&layerHasReported, _layerHasReported, sizeof(int), cudaMemcpyDeviceToHost);
*/
//edited code by luca

//change is that memcpy is used in place of cudaMemcpy before and after the gpu_calcOutput, also removed the allocate

    memcpy(h_aPinned, h_aPageable, sizeof(int));
    
    gpu_calcOutputs<<<nNeurons,1>>>(gpu_neurons, h_bPinned);
    cudaDeviceSynchronize();

    memcpy(&h_bPinned, h_aPageable, sizeof(int));

}

//*************************************************************************************
//forward propagation of error:
//*************************************************************************************

/*__host__ void Layer::setForwardError(double _leadForwardError){
    this is only for the first layer
    leadForwardError=_leadForwardError;
    gpu_setForwardError<<<1,nNeurons>>>(gpu_neurons, leadForwardError);
    cudaDeviceSynchronize();
}*/

//__host__ void Layer::propErrorForward(int _index, double _value){
//    for (int i=0; i<nNeurons; i++){
//        neurons[i]->propErrorForward(_index, _value);
//    }
//}

/*__host__ double Layer::getForwardError(int _neuronIndex){
    return (neurons[_neuronIndex].getForwardError());
}*/

//*************************************************************************************
//back propagation of error:
//*************************************************************************************

__host__ void Layer::setBackwardError(double _leadBackwardError) {
    leadBackwardError = _leadBackwardError;
    int B = std::ceil(float(nNeurons)/MAX_BLOCKSIZE);   // Total number of blocks required
    int T = MAX_BLOCKSIZE;
    if (nNeurons<MAX_BLOCKSIZE){
        T = nNeurons;
    }
    //printf("%d, %d\n", B, T);
    gpu_setBackwardError<<<B,T>>>(gpu_neurons, leadBackwardError, nNeurons);
    cudaDeviceSynchronize();
}

__host__ double* Layer::calcErrorWeightProductSum() {
    int nThreads = nInputs * nNeurons;          // Total number of CUDA threads required
    int blockYDim = MAX_BLOCKSIZE/nInputs;      // Size of a block's Y dimension
    int blockSize = nInputs * blockYDim;        // Size of required block
    int B = std::ceil(float(nThreads)/blockSize);   // Total number of blocks required
    dim3 T = dim3(nInputs, blockYDim);          // 2D block dimensions
    //printf("%d, %d, %d\n", B, nInputs, blockYDim);
    gpu_calcErrorWeightProductSum<<<B,T>>>(gpu_neurons, nNeurons, nInputs, gpu_sumlist);
    cudaDeviceSynchronize();
    return gpu_sumlist;
}

__host__ void Layer::propErrorBackward(double* _sumList) {
    int B = std::ceil(float(nNeurons)/MAX_BLOCKSIZE);   // Total number of blocks required
    int T = MAX_BLOCKSIZE;
    if (nNeurons<MAX_BLOCKSIZE){
        T = nNeurons;
    }
    //printf("%d, %d\n", B, T);
    gpu_propErrorBackwards<<<B,T>>>(gpu_neurons, _sumList, nNeurons);
    cudaDeviceSynchronize();
}

//*************************************************************************************
//learning:
//*************************************************************************************

__host__ void Layer::updateWeights() {
    int nThreads = nInputs * nNeurons;          // Total number of CUDA threads required
    int blockYDim = MAX_BLOCKSIZE/nInputs;      // Size of a block's Y dimension
    int blockSize = nInputs * blockYDim;        // Size of required block
    int B = std::ceil(float(nThreads)/blockSize);   // Total number of blocks required
    dim3 T = dim3(nInputs, blockYDim);          // 2D block dimensions

    gpu_updateWeights<<<B,T>>>(gpu_neurons, nNeurons);
    cudaDeviceSynchronize();
}

//*************************************************************************************
//getters:
//*************************************************************************************

__host__ Neuron* Layer::getNeuron(int _neuronIndex){
    return (&neurons[_neuronIndex]);
}

__host__ int Layer::getnNeurons(){
    return (nNeurons);
}

__host__ double* Layer::getOutputs(){
    double* _outputs;
    cudaMalloc(&_outputs, sizeof(double)*nNeurons);
    int B = std::ceil(float(nNeurons)/MAX_BLOCKSIZE);   // Total number of blocks required
    int T = MAX_BLOCKSIZE;
    if (nNeurons<MAX_BLOCKSIZE){
        T = nNeurons;
    }
    gpu_getOutputs<<<B,T>>>(gpu_neurons, _outputs, nNeurons);
    return _outputs;
}

__host__ double Layer::getOutput(int _neuronIndex) {
    return (neurons[_neuronIndex].getOutput());
}

__host__ double Layer::getErrorWeightProductSum(int index) {
    double _sum = 0.0;
    double* sum = gpu_sumlist + index;
    cudaMemcpy(&_sum, sum, sizeof(double), cudaMemcpyDeviceToHost);
    return _sum;
}

__host__ double Layer::getBackwardError(int _neuronIndex){
    return (neurons[_neuronIndex].getBackwardError());
}

__host__ void Layer::printWeights(FILE* weights) {
    for (int i=0;i<nNeurons;i++) {
        for (int j=0;j<nInputs;j++) {
            fprintf(weights,"%f, ", neurons[i].getWeight(j));
        }
        fprintf(weights,"\n");
    }
    fprintf(weights,"\n");
}

