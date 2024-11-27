
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

//added by luca gpu_set_inputs wth less blocks to launch 

__global__ void gpu_setInputs_less_blocks(Neuron* n, double *list, int nNeurons,int nInputs) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i = tid;i<nNeurons;i+=gridDim.x*blockDim.x){
        for(int j = threadIdx.x;j<nInputs;j+=128){
            n[i].inputs[j] = list[j];
        }
    }
    __syncthreads();
}


//added by luca, function to integrate mutliple kernel launches into a single launch

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

//added by luca
//Since for the LMS application the number of blocks launched for layer 1 in 1, can just sync threads and prop backwards straight away
__global__ void gpu_setBackwardError_lms(Neuron*n, double _leadBackwardError, int nNeurons) {
    double leadBackwardError = _leadBackwardError;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i = blockIdx.x;i<nNeurons;i+= blockDim.x){ 
        device_setBackwardError(leadBackwardError, &n[i]);
    }
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

//added by luca



__global__ void gpu_calcErrorWeightProductSum_less_blocks(Neuron* n,Neuron* neuron_previous_layer, int nNeurons, int nInputs, double* sumlist,double* inputs_previous_layer) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for(int j = blockIdx.x;j<nInputs;j+= blockDim.x){ 
        device_calcErrorWeightProductSum_less_blocks(n,nNeurons,sumlist,j);
    }
    __syncthreads();

    for(int i = tid;i<nInputs;i+=blockDim.x * gridDim.x){
        device_propErrorBackward(sumlist[i], &neuron_previous_layer[i]);
        }
    
    //updating weights associated with current neurons
    __syncthreads();
    for(int i = tid;i<nInputs;i+=blockDim.x * gridDim.x){
        for(int j = threadIdx.x;j<nInputs;j+=128){
            neuron_previous_layer[i].weights[j] += (*neuron_previous_layer[i].learningRate) *inputs_previous_layer[j] * (*neuron_previous_layer[i].backwardError);
        }
    }
    __syncthreads();
}


__global__ void gpu_calcOutputs(Neuron* neurons, int* layerHasReported){
    device_calcOutput(&neurons[blockIdx.x], layerHasReported);
    __syncthreads();
}

//added by luca, gpu_calcOutputs_with less block launches

__global__ void gpu_calcOutputs_less_blocks(Neuron* neurons, int* layerHasReported, int nNeurons,double* _get_output_array_Pinned,int nNeurons_next_layer,double* inputs,double* inputs_next_layer,int start_idx_for_reduction,int number_of_concurrent_neurons_per_thread_block){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    //printf("tid/128:%i\n",tid/start_idx_for_reduction);
    for(int i = tid/start_idx_for_reduction;i<nNeurons;i+=gridDim.x*number_of_concurrent_neurons_per_thread_block){
        device_calcOutput_using_layer_level_inputs(&neurons[i], layerHasReported,inputs,inputs_next_layer,_get_output_array_Pinned,i,start_idx_for_reduction);
    } 
}   

__global__ void gpu_calcOutputs_less_blocks_final_layer(Neuron* neurons, int* layerHasReported, int nNeurons,double* _get_output_array_Pinned,double* inputs_a_Pinned,int start_idx_for_reduction,int number_of_concurrent_neurons_per_thread_block){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i = tid/start_idx_for_reduction;i<nNeurons;i+=gridDim.x*number_of_concurrent_neurons_per_thread_block){
        device_calcOutput_using_layer_level_inputs_no_prop(&neurons[i], layerHasReported,inputs_a_Pinned,_get_output_array_Pinned,i,start_idx_for_reduction);
    }
}   



//added by luca, to set sum to zero
__global__ void layer_set_sum_zero(Neuron* neurons){
    setSum_zero(&neurons[blockIdx.x]);
}

__global__ void gpu_propErrorBackwards(Neuron *n, double* _sumList, int nNeurons) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double* sumList = _sumList;
    if (i<nNeurons)
        device_propErrorBackward(sumList[i], &n[i]);
}


__global__ void gpu_propErrorBackwards_less_blocks(Neuron *n, double* _sumList, int nNeurons,double* inputs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double* sumList = _sumList;
    for(int i = tid;i<nNeurons;i+=blockDim.x * gridDim.x){
        device_propErrorBackward(sumList[i], &n[i]);
        }
    
    //updating weights associated with current neurons
    __syncthreads();
    for(int i = blockIdx.x;i<nNeurons;i+=gridDim.x){
        for(int j = threadIdx.x;j<nNeurons;j+=128){
            n[i].weights[j] += (*n[i].learningRate) * inputs[j] * (*n[i].backwardError);
        }
    }
}



__global__ void gpu_updateWeights(Neuron *n, int nNeurons){
    int i = threadIdx.x;    //Input index
    int j = (blockIdx.x*blockDim.y) + threadIdx.y;  //Neuron index
    //double force = 1;
    if (j<nNeurons) {
        n[j].weights[i] += (*n[j].learningRate) * n[j].inputs[i] * (*n[j].backwardError); // * force;
    }
}

//added by luca less blocks gpu_update_weights
__global__ void gpu_updateWeights_less_blocks(Neuron *n, int nNeurons,double* inputs){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i = tid;i<nNeurons;i+=blockDim.x * gridDim.x){
        for(int j = threadIdx.x;j<nNeurons;j+=128){
            n[i].weights[j] += (*n[i].learningRate) * inputs[j] * (*n[i].backwardError);
        }
    }
}

__global__ void gpu_updateWeights_first_layer_less_blocks(Neuron *n, int nNeurons,double* inputs){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i = tid;i<nNeurons;i+=blockDim.x * gridDim.x){
        for(int j = threadIdx.x;j<nNeurons;j+=128){
            n[i].weights[j] += (*n[i].learningRate) * inputs[j] * (*n[i].backwardError);
        }
    }
}



__global__ void gpu_getOutputs(Neuron* n, double* _outputs, int nNeurons){
   int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid<nNeurons) {
        _outputs[tid] = *n[tid].output;
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


__host__ int* Layer::generating_pinned_memory_address(int nInputs){


    const unsigned int bytes =  sizeof(int);


    //int *h_aPageable, *h_bPageable;   
    //int *h_aPinned, *h_bPinned;


    h_aPageable = (int*)malloc(bytes);                   
    h_bPageable = (int*)malloc(bytes);

    inputs_a_Pageable = (double*)malloc(sizeof(double)*nInputs);



    //getoutputs pageable
    get_output_Pageable = (double*)malloc(sizeof(double));

    get_output_array_Pageable = (double*)malloc(sizeof(double)*nNeurons);

    checkCuda(cudaMallocHost((void**)&h_aPinned, bytes) ); // host pinned
    checkCuda(cudaMallocHost((void**)&h_bPinned, bytes) ); // host pinned
    checkCuda(cudaMallocHost((void**)&inputs_a_Pinned, sizeof(double)*nInputs));
    checkCuda(cudaMallocHost((void**)&get_output_Pinned, sizeof(double)));

    checkCuda(cudaMallocHost((void**)&get_output_array_Pinned, sizeof(double)*nNeurons));

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
    //generating pinned addresses
    generating_pinned_memory_address(nInputs);


    //trying to add section that will allow for multiple neurons to be calculated by the same block if there are less than T/2 threads being used
    
    threads_per_block = 128;
    

    if(nInputs<threads_per_block/2 + 1){
        int n_reductions_needed = ceil(log2(nInputs));
        start_idx_for_reduction = pow(2,5);
        number_of_concurrent_neurons_per_thread_block = threads_per_block/start_idx_for_reduction;

        printf("number_of_concurrent_neurons_per_thread_block: %i\nstart_idx_for_reduction: %i\n",number_of_concurrent_neurons_per_thread_block,start_idx_for_reduction);
    }
    else{
        start_idx_for_reduction = threads_per_block;
        number_of_concurrent_neurons_per_thread_block = 1;
        printf("number_of_concurrent_neurons_per_thread_block: %i\nstart_idx_for_reduction: %i\n",number_of_concurrent_neurons_per_thread_block,start_idx_for_reduction);
    }


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
    cudaFreeHost(inputs_a_Pinned);
    free(h_aPageable);
    free(h_bPageable);
    free(inputs_a_Pageable);
    
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
    inputs_a_Pageable = _inputs;
    //cudaMemcpy(gpu_inputs, inputs, sizeof(double)*nInputs,cudaMemcpyHostToDevice);
    memcpy(inputs_a_Pinned,inputs_a_Pageable,sizeof(double)*nInputs);

    int B;
    if(nNeurons>8){
        B = 8;
    }
    else{
        B = nNeurons;
    }
    int T = 128;

    //restore below if the size of network is large enough to justify the kernel launch

    gpu_setInputs_less_blocks<<<B,T>>>(gpu_neurons, inputs_a_Pinned, nNeurons,nInputs);

    //cudaDeviceSynchronize();
}

//added by luca
//seting inputs only at the layer level to reduce number of operations 

__host__ void Layer::setInputs_layer_level_only(double* _inputs){
    inputs = _inputs;
    inputs_a_Pageable = _inputs;
    memcpy(inputs_a_Pinned,inputs_a_Pageable,sizeof(double)*nInputs);
};


__host__ void Layer::propInputs(double* _gpu_InputOutputs) {
    
    int B;
    if(nNeurons*nInputs>8){
        B = 8;
    }
    else{
        B = nNeurons;
    }
    int T = 128;


    gpu_setInputs_less_blocks<<<B,T>>>(gpu_neurons, _gpu_InputOutputs, nNeurons,nInputs);
    cudaDeviceSynchronize();
}

__host__ void Layer::calcOutputs(double* inputs_next_layer,int nNeurons_next_layer){

    int B;
    if(nNeurons>8){
        B = 8;
    }
    else{
        B = nNeurons;
    }

    gpu_calcOutputs_less_blocks<<<B,threads_per_block>>>(gpu_neurons, h_bPinned, nNeurons, get_output_array_Pinned, nNeurons_next_layer, inputs_a_Pinned, inputs_next_layer,start_idx_for_reduction,number_of_concurrent_neurons_per_thread_block);
}

__host__ void Layer::calcOutputs_final_layer(){

    int B;
    if(nNeurons>8){
        B = 8;
    }
    else{
        B = nNeurons;
    }

    gpu_calcOutputs_less_blocks_final_layer<<<B,threads_per_block>>>(gpu_neurons,h_bPinned,nNeurons,get_output_array_Pinned,inputs_a_Pinned,start_idx_for_reduction,number_of_concurrent_neurons_per_thread_block);
    
}

__host__ void Layer::setSum_zero(){
    layer_set_sum_zero<<<nNeurons,1>>>(gpu_neurons);
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

//added by luca, funtion specific to LMS type applications, error is calculated from the input signal
__host__ double Layer::setBackwardError_LMS(double input_signal) {
    leadBackwardError = input_signal-get_output_array_Pinned[0];
    int B = std::ceil(float(nNeurons)/MAX_BLOCKSIZE);   // Total number of blocks required
    int T = MAX_BLOCKSIZE;
    if (nNeurons<MAX_BLOCKSIZE){
        T = nNeurons;
    }

    gpu_setBackwardError_lms<<<B,T>>>(gpu_neurons, leadBackwardError, nNeurons);

    return leadBackwardError;
}

__host__ double* Layer::calcErrorWeightProductSum(Neuron* neuron_previous_layer,double* inputs_previous_layer) { 

    int B = 8;
    int T = 128;

    if(nInputs<8){
        B = nInputs;
    }

    gpu_calcErrorWeightProductSum_less_blocks<<<B,T>>>(gpu_neurons,neuron_previous_layer, nNeurons, nInputs, gpu_sumlist,inputs_previous_layer);

    cudaDeviceSynchronize();
}

__host__ void Layer::propErrorBackward(double* _sumList) {
    int B;
    if(nNeurons>1024){
        B = 8;
    }
    else{
        B = 8 * std::ceil(nNeurons/128);
    }
    int T = 128;

    gpu_propErrorBackwards_less_blocks<<<B,T>>>(gpu_neurons, _sumList, nNeurons,inputs_a_Pinned);
    //cudaDeviceSynchronize();
}

//*************************************************************************************
//learning:
//*************************************************************************************

__host__ void Layer::updateWeights() {
    int B;
    if((nNeurons*nInputs)>1024){
        B = 8;
    }
    else{
        B = 8 * std::ceil((nNeurons*nInputs)/128);
    }
    int T = 128;

    gpu_updateWeights_less_blocks<<<B,T>>>(gpu_neurons, nNeurons,inputs_a_Pinned);
    //cudaDeviceSynchronize();
}


__host__ void Layer::updateWeights_first_layer() {
    int B;
    if((nNeurons*nInputs)>1024){
        B = 8;
    }
    else{
        B = 8 * std::ceil((nNeurons*nInputs)/128);
    }
    int T = 128;

    gpu_updateWeights_first_layer_less_blocks<<<B,T>>>(gpu_neurons, nNeurons,inputs_a_Pinned);
    //cudaDeviceSynchronize();
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

    int B;
    if((nNeurons)>1024){
        B = 8;
    }
    else{
        B = 8 * std::ceil((nNeurons)/128);
    }
    int T = 128;

    gpu_getOutputs<<<B,T>>>(gpu_neurons, get_output_array_Pinned, nNeurons);

    return get_output_array_Pinned;
}

__host__ double Layer::getOutput(int _neuronIndex) {
    return (neurons[_neuronIndex].getOutput());
}

//added by luca
__host__ double Layer::getOutput_no_memcpy(int _neuronIndex) {
    neurons[_neuronIndex].getOutput_no_memcpy(get_output_Pinned,get_output_Pageable);
    return(*get_output_Pageable);
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




