
#include "cldl/Neuron.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <stdio.h>


//check_cuda_function
inline
cudaError_t checkCuda(cudaError_t result)
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

//*************************************************************************************
// constructor de-constructor
//*************************************************************************************

__host__ Neuron::Neuron(int _nInputs)
{   
    //cudaMalloc(&array_for_dot_product_sum,128*sizeof(double));


    // initialisation
    gpu_allocateInt(&nInputs, _nInputs);
    gpu_allocateInt(&myLayerIndex, 0);
    gpu_allocateInt(&myNeuronIndex, 0);
    cudaMalloc((void**)&initialWeights, sizeof(double)*_nInputs);
    gpu_allocateDouble(&learningRate, 0.0);

    gpu_allocateInt(&iHaveReported, 0);


    // forward propagation of inputs
    checkCuda(cudaMallocHost((void**)&inputs, sizeof(double)*_nInputs));
    gpu_allocateDouble(&bias, 0.0);
    gpu_allocateDouble(&sum, 0.0);
    gpu_allocateDouble(&output, 0.0);

    // forward propagation of error
    cudaMalloc((void**)&inputErrors, sizeof(double)*_nInputs);
    gpu_allocateDouble(&forwardError, 0.0);
    gpu_allocateDouble(&calcForwardOutput,0.0);

    // back propagation of error
    gpu_allocateDouble(&backwardError, 0.0);
    cudaMalloc((void**)&ErrorWeightProducts, sizeof(double)*_nInputs);

    // mid propagation of error
    cudaMalloc((void**)&inputMidErrors, sizeof(double)*_nInputs);
    gpu_allocateDouble(&midError, 0.0);


    //
    // learning variables
    //
    gpu_allocateDouble(&backwardsCoeff, 0.0);
    gpu_allocateDouble(&midCoeff, 0.0);
    gpu_allocateDouble(&forwardCoeff, 0.0);
    gpu_allocateDouble(&globalCoeff, 0.0);

    cudaMalloc((void**)&weights, sizeof(double)*_nInputs);

    gpu_allocateDouble(&weightSum, 0.0);
    gpu_allocateDouble(&maxWeight, 1.0);
    gpu_allocateDouble(&minWeight, 1.0);
    gpu_allocateDouble(&weightChange, 0.0);
    gpu_allocateDouble(&weightsDifference, 0.0);
    gpu_allocateInt(&actMet, 0);

    // global setting
    gpu_allocateDouble(&globalError, 0.0);
    gpu_allocateDouble(&localError, 0.0);
    gpu_allocateDouble(&echoCoeff, 0.0);
    gpu_allocateDouble(&localCoeff, 0.0);

    gpu_allocateDouble(&overallError, 0.0);
    gpu_allocateDouble(&echoError, 0.0);
    cudaMalloc((void**)&echoErrors, sizeof(double)*_nInputs);

    //cout << "neuron" << endl;



}

__host__ Neuron::~Neuron(){

    //added by luca 
    //cudaFree(array_for_dot_product_sum);


    //initialisation
    cudaFree(nInputs);
    cudaFree(learningRate);
    cudaFree(myLayerIndex);
    cudaFree(initialWeights);
    cudaFree(myNeuronIndex);

    cudaFree(iHaveReported);

    // forward propagation of inputs
    cudaFreeHost(inputs);
    cudaFree(bias);
    cudaFree(sum);
    cudaFree(output);

    // forward propagation of error
    cudaFree(inputErrors);
    cudaFree(forwardError);
    cudaFree(calcForwardOutput);

    // back propagation of error
    cudaFree(backwardError);
    cudaFree(ErrorWeightProducts);

    // mid propagation of error
    cudaFree(inputMidErrors);
    cudaFree(midError);


    //learning
    cudaFree(backwardsCoeff);
    cudaFree(midCoeff);
    cudaFree(forwardCoeff);
    cudaFree(globalCoeff);
    cudaFree(weights);
    cudaFree(weightSum);
    cudaFree(maxWeight);
    cudaFree(minWeight);
    cudaFree(weightChange);
    cudaFree(weightsDifference);
    cudaFree(actMet);

    // global setting
    cudaFree(globalError);
    cudaFree(localError);
    cudaFree(echoCoeff);
    cudaFree(localCoeff);

    cudaFree(overallError);
    cudaFree(echoError);
    cudaFree(echoErrors);
}


//*************************************************************************************
//initialisation:
//*************************************************************************************

__host__ void Neuron::initNeuron(int _neuronIndex, int _layerIndex, weightInitMethod _wim, biasInitMethod _bim, actMethod _am){
    cudaMemcpy(myLayerIndex, &_layerIndex, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(myNeuronIndex, &_neuronIndex, sizeof(int), cudaMemcpyHostToDevice);
    switch(_wim) {
        case W_ZEROS:
            gpu_setValuesInArray<<<1,getNInputs()>>>(0, weights);
            break;
        case W_ONES:
            gpu_setValuesInArray<<<1,getNInputs()>>>(1, weights);
            break;
        case W_RANDOM:
            curandGenerator_t gen;
            curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
            curandSetPseudoRandomGeneratorSeed(gen, std::chrono::duration_cast<std::chrono::milliseconds>
                    (std::chrono::system_clock::now().time_since_epoch()).count());

            curandGenerateUniformDouble(gen, weights, getNInputs());
            break;
            /* rand function generates a random function between
             * 0 and 1, with the CUDA Random generator seed set
             * to current time from UNIX epoch (inherently unique)*/
    }
    cudaMemcpy(initialWeights, weights, sizeof(double)*getNInputs(), cudaMemcpyDeviceToDevice);

    gpu_setDouble<<<1,1>>>(weightSum, 0);
    gpu_getSumAndMaxMin<<<1,1>>>(weightSum, maxWeight, minWeight, weights, getNInputs());

    switch (_bim){
        case B_NONE:
            gpu_setDouble<<<1,1>>>(bias, 0.0);
            break;
        case B_RANDOM:
            gpu_setDouble<<<1,1>>>(bias, ((double)rand()/RAND_MAX));
            break;
    }
    switch(_am){
        case Act_Sigmoid:
            gpu_setInt<<<1,1>>>(actMet, 0);
            break;
        case Act_Tanh:
            gpu_setInt<<<1,1>>>(actMet, 1);
            break;
        case Act_NONE:
            gpu_setInt<<<1,1>>>(actMet, 2);
            break;
    }
}

__host__ void Neuron::setLearningRate(double _learningRate){
    gpu_setDouble<<<1,1>>>(learningRate, _learningRate);
}

__device__ void device_setLearningRate(Neuron* n, double _learningRate){
    *n->learningRate = _learningRate;
}

__host__ double Neuron::getLearningRate() {
    double _learningRate;
    cudaMemcpy(&_learningRate, learningRate, sizeof(double), cudaMemcpyDeviceToHost);
    return _learningRate;
}


//*************************************************************************************
//forward propagation of inputs:
//*************************************************************************************

__host__ void Neuron::setInput(int _index, double _value) {
    assert((_index>=0)&&(_index<getNInputs()));
    gpu_setValueInArray<<<1,1>>>(_value, _index, inputs);
}

__host__ double Neuron::getInput(int index) {
    double _input = 0.0;
    assert(index < getNInputs());

    double* input = inputs + index;
    cudaMemcpy(&_input, input, sizeof(double), cudaMemcpyDeviceToHost);
    return _input;
}

__host__ void Neuron::propInputs(int _index,  double _value){
    assert((_index>=0)&&(_index < getNInputs()));
    gpu_setValueInArray<<<1,1>>>(_value,_index, inputs);
}


/*
__device__ void device_calcOutput(Neuron* n, int* _layerHasReported){
    int nInputs = *(n->nInputs);
    __shared__ double _array_for_sum[128];
    device_dotProduct((*n).inputs, (*n).weights, (*n).sum, nInputs,_array_for_sum);
    __syncthreads();
        parallelReduction(n,_array_for_sum,nInputs);

    __syncthreads();
    if(threadIdx.x ==0){
    device_calcOutputCont(n, _layerHasReported);
    }
}
*/

__device__ void device_calcOutput_using_layer_level_inputs_no_prop(Neuron* n, int* _layerHasReported,double* inputs,double* outputs_current_layer,int neuron_index,int start_idx_for_reduction,const int threads_per_block,int nNeurons, double* _array_for_sum){
    
    int nInputs = *(n->nInputs);

    //calculating effective index
    
    int neuron_in_block_being_calculated = 0;

    //effective_idx
    int e_idx = threadIdx.x;

    if(start_idx_for_reduction != threads_per_block){

        //effectivly can ignore the added part from e_idx as it is removed by the int
        neuron_in_block_being_calculated = threadIdx.x/start_idx_for_reduction;

        //calculating effective idx
        e_idx = threadIdx.x - neuron_in_block_being_calculated * start_idx_for_reduction;
    }

    device_dotProduct(inputs, (*n).weights, (*n).sum, nInputs,_array_for_sum,e_idx);
    __syncthreads();

    //performing parallel reduction
    parallelReduction(n,_array_for_sum,start_idx_for_reduction,e_idx,neuron_in_block_being_calculated);
    __syncthreads();

    //needs updated for effective index
    if(e_idx == 0){
    device_calcOutputCont(n, _layerHasReported);
    outputs_current_layer[neuron_index] = *(*n).output;
    }
}

__device__ void device_calcOutput_using_layer_level_inputs(Neuron* n, int* _layerHasReported,double* inputs,double* next_layer_inputs,double * outputs_current_layer,int neuron_index,int start_idx_for_reduction,const int threads_per_block,int nNeurons,double* _array_for_sum){
    
    //printf("hi");

    //calculating effective index
    int neuron_in_block_being_calculated = 0;

    //effective_idx
    int e_idx = threadIdx.x;

    if(start_idx_for_reduction != threads_per_block){
        //effectivly can ignore the added part from e_idx as it is removed by the int
        neuron_in_block_being_calculated = threadIdx.x/start_idx_for_reduction;

        //calculating effective idx
        e_idx = threadIdx.x - neuron_in_block_being_calculated * start_idx_for_reduction;
    }

    //the size of this array is equal to the number of threads that are set per block 
    device_dotProduct(inputs, (*n).weights, (*n).sum, *(n->nInputs),_array_for_sum,e_idx);
    __syncthreads();


    //performing parallel reduction
    parallelReduction(n,_array_for_sum,start_idx_for_reduction,e_idx,neuron_in_block_being_calculated);
    __syncthreads();

    //needs updated for effective index
    if(e_idx == 0){
    device_calcOutputCont(n, _layerHasReported);
    outputs_current_layer[neuron_index] = *(*n).output;
    next_layer_inputs[neuron_index] = *(*n).output;

    //printf("Neuron Index: %i\noutput:%f\nnext_layer_inputs:%f\n\n",neuron_index,*(*n).output,next_layer_inputs[neuron_index]);
    }
}

__device__ void device_calcOutputCont(Neuron* n, int* _layerHasReported){
        if (*(*n).myLayerIndex == 0){
            *(*n).sum = *(*n).sum * 0.01;
        }
        *(*n).sum += *(*n).bias;
        device_doActivation((*n).output, (*n).sum, (*n).actMet);
        *(*n).iHaveReported = *_layerHasReported;
        if (*(*n).output > 0.49 && *(*n).iHaveReported == 0){
            *(*n).iHaveReported = 1;
        }
        *_layerHasReported = *(*n).iHaveReported;
}

//added by luca 

__device__ void parallelReduction(Neuron* n,double* _array_for_dot_sum,int start,int e_idx,int neuron_in_block_being_calculated){

    int idx = threadIdx.x;

    //if(e_idx != idx && e_idx + neuron_in_block_being_calculated*start > 1023){
    //printf("start:%i\nneuron_being_calculated: %i\nidx:%i\nEffective_idx:%i\nindex:%i\n\n",start,neuron_in_block_being_calculated,idx,e_idx,e_idx + neuron_in_block_being_calculated*start);}
    
    for (int s = start / 2; s > 0; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (e_idx < s) {
			_array_for_dot_sum[e_idx + neuron_in_block_being_calculated*start] += _array_for_dot_sum[e_idx + s + neuron_in_block_being_calculated*start];
		}
		__syncthreads();
	}
    __syncthreads();
    // Let the thread 0 for this block write its result to main memory
	// uses effective index
	if (e_idx == 0){

        //printf("idx:%i\n",idx);
		double total = _array_for_dot_sum[neuron_in_block_being_calculated * start];
        *(*n).sum = total;
	}
}

//added by luca, device function to deal with neurons block by block in calcWeightError


__device__ void device_calcErrorWeightProductSum_less_blocks(Neuron* n, int nNeurons, double* sumlist,int j,int start_idx_for_reduction,int number_of_concurrent_neurons_per_thread_block,int e_idx,double* _array_for_sum,int neuron_in_block_being_calculated){

    //want to update sumlist[nInputs] by the new error calculated for neuron[j]

    int idx = threadIdx.x;

    double temp_sum = 0.0;


    for(int i = e_idx ;i<nNeurons;i+= blockDim.x){
        //printf("e_idx:%i\nnNeurons:%i\n",threadIdx.x,nNeurons);
        n[i].ErrorWeightProducts[j] = n[i].weights[j] * (*n[i].backwardError);
        temp_sum += n[i].ErrorWeightProducts[j];
    }
   
   //store result in idx so that the same location isn't written to by different threads
    _array_for_sum[idx] = temp_sum;
    __syncthreads();

    if(e_idx == 0){
        parallelReduction_weights(j,_array_for_sum,sumlist,start_idx_for_reduction,e_idx,neuron_in_block_being_calculated);
        //printf("array_for_sum[10]: %f\n",_array_for_sum[10]);
    }
    __syncthreads();

}

__device__ void parallelReduction_weights(int j,double* _array_for_dot_sum,double * sumlist,int start,int e_idx,int neuron_in_block_being_calculated){

    for (int s = start / 2 ; s > 0; s >>= 1) {
		// Each thread does work unless it is further than the stride
        //printf("s: %i\n",start);
		if (e_idx < s) {
            //shifting along the array by the final index for e_idx
			_array_for_dot_sum[e_idx + neuron_in_block_being_calculated*start] += _array_for_dot_sum[e_idx + s + neuron_in_block_being_calculated*start];
		}
		__syncthreads();
	}
      __syncthreads();
    // Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (e_idx == 0) {
		double total = _array_for_dot_sum[neuron_in_block_being_calculated * start];
        sumlist[j] = total;
        //printf("sumlist[j]: %f\nindex: %i\n_array_for_sum[0]: %f\n",sumlist[j],neuron_in_block_being_calculated * start,_array_for_dot_sum[0]);
	}

}

__device__ double parallelReduction_updating_weights(double* weight_sum_array){

}


/*
__device__ void device_sum_tempArray(Neuron* n){

    //variable for final value
    double total = 0.0;

    //array of temp values
    double* array = (*n).array_for_dot_product_sum;

    //threadIdx
    int idx = threadIdx.x;

    //nInputs
    int nelements = 128;
    int nInputs = *(*n).nInputs;

    if(nInputs<128){
        nelements = nInputs;
    }

    for(int i = 0;i<nelements;i++){
        total +=  array[i]; 
    }  

    *(*n).sum = total;



}
*/


//int Neuron::calcOutput(int _layerHasReported){
//    sum=0;
//    for (int i=0; i<nInputs; i++){
//        sum += inputs[i] * weights[i];
//    }
//    sum += bias;
//    if (myLayerIndex == 0){
//        sum = sum * 0.01;
//    }
//    assert(std::isfinite(sum));
//    output = doActivation(sum);
//    iHaveReported = _layerHasReported;
//    if (output > 0.49 && iHaveReported == 0){
//        //cout << "I'm saturating, " << output << " layer: " << myLayerIndex << " neuron: " << myNeuronIndex << endl;
//        iHaveReported = 1;
//    }
//    assert(std::isfinite(output));
//    return iHaveReported;
//}


//*************************************************************************************
//forward propagation of error:
//*************************************************************************************

__host__ void Neuron::setForwardError(double _value) {
    gpu_setValuesInArray<<<1, getNInputs()>>>(_value, inputErrors);
}

__host__ double Neuron::getInputError(int _index) {
    double _inputError = 0.0;
    assert(_index < getNInputs());

    double* inputError = inputErrors + _index;
    cudaMemcpy(&_inputError, inputError, sizeof(double), cudaMemcpyDeviceToHost);
    return _inputError;
}

__host__ void Neuron::propErrorForward(int _index, double _value){
    assert((_index>=0)&&(_index<getNInputs()));
    gpu_setValueInArray<<<1,1>>>(_value, _index, inputErrors);
}


__device__ void device_calcForwardError(Neuron* n){
    __shared__ double _value[1024];
    int nInputs = *(n->nInputs);
    device_dotProduct((*n).inputErrors,(*n).weights, (*n).calcForwardOutput, nInputs,_value,threadIdx.x);
    device_doActivationPrime((*n).forwardError, (*n).sum, (*n).actMet);
    *(*n).forwardError = *(*n).forwardError * *(*n).calcForwardOutput;
}

__global__ void gpu_calcForwardError(Neuron* n){
    device_calcForwardError(n);
}

//__host__ void Neuron::calcForwardError() {
//    double* _value;
//    cudaMalloc((void**)&_value, sizeof(double)*getNInputs());
//    gpu_dotProduct<<<1, getNInputs()>>>(inputErrors, weights, _value, forwardError, getNInputs());
//
//    TODO forwardError must be multiplied with doActivationPrime(sum)
//    TODO assert forwardError isFinite
//}

__host__ double Neuron::getForwardError() {
    double _forwardError = 0.0;
    cudaMemcpy(&_forwardError, forwardError, sizeof(double), cudaMemcpyDeviceToHost);
    return _forwardError;
}


//*************************************************************************************
//back propagation of error
//*************************************************************************************

//TODO Fix setBackwardError
__host__ void Neuron::setBackwardError(double _leadError){
    gpu_doActivationPrime<<<1,1>>>(backwardError, sum, actMet);
    gpu_multiplication<<<1,1>>>(_leadError,backwardError);
}

__device__ void device_setBackwardError(double _leadError, Neuron* n){
    device_doActivationPrime((*n).backwardError, (*n).sum, (*n).actMet);
    *(*n).backwardError = *(*n).backwardError * _leadError;
}

__global__ void gpu_setBackwardError(double _leadError, Neuron* n){
    double leadError = _leadError;
    device_setBackwardError(leadError, n);
}

__device__ void device_propErrorBackward(double _nextSum, Neuron* n){
    device_doActivationPrime((*n).backwardError, (*n).sum, (*n).actMet);
    *(*n).backwardError = *(*n).backwardError * _nextSum;
}

__global__ void gpu_propErrorBackward(double _nextSum, Neuron* n){
    double nextSum = _nextSum;
    device_propErrorBackward(nextSum, n);
}

__host__ double Neuron::getBackwardError(){
    double _backwardError = 0.0;
    cudaMemcpy(&_backwardError, backwardError, sizeof(double), cudaMemcpyDeviceToHost);
    return _backwardError;
}

__host__ double Neuron::getErrorWeightProducts(int index) {
    double _ewProd = 0.0;

    double* ewProd = ErrorWeightProducts + index;
    cudaMemcpy(&_ewProd, ewProd, sizeof(double), cudaMemcpyDeviceToHost);
    return _ewProd;
}

__host__ double Neuron::getEchoError() {
    double _echoError = 0;
    cudaMemcpy(&_echoError, echoError, sizeof(double), cudaMemcpyDeviceToHost);
    return _echoError;
}

__device__ void echoErrorBackward(double _nextSum, Neuron* n) {
    device_doActivationPrime((*n).echoError,(*n).sum,(*n).actMet);
    *(*n).echoError = *(*n).echoError * _nextSum;
}

__global__ void gpu_echoErrorBackward(double _nextSum, Neuron* n){
    double nextSum = _nextSum;
    echoErrorBackward(nextSum, n);
}



//*************************************************************************************
//MID propagation of error
//*************************************************************************************

__host__ void Neuron::setMidError(double _leadMidError) {
    gpu_setValuesInArray<<<1, getNInputs()>>>(_leadMidError, inputMidErrors);
}

__host__ double Neuron::getInputMidErrors(int index) {
    double _inputMidError = 0.0;
    assert(index < getNInputs());

    double* inputMidError = inputMidErrors + index;
    cudaMemcpy(&_inputMidError, inputMidError, sizeof(double), cudaMemcpyDeviceToHost);
    return _inputMidError;
}

//TODO find how to do midError = midError * doActivationPrime(sum)
__host__ void Neuron::calcMidError() {
    double* _value;
    cudaMalloc((void**)&_value, sizeof(double)*getNInputs());
    gpu_dotProduct<<<1, getNInputs()>>>(inputMidErrors, weights, _value, midError, getNInputs());
    double output = getMidError();
    gpu_doActivationPrime<<<1,1>>>(midError, sum, actMet);
    gpu_multiplication<<<1, 1>>>(output, midError);

}


__host__ double Neuron::getMidError() {
    double _midError = 0.0;
    cudaMemcpy(&_midError, midError, sizeof(double), cudaMemcpyDeviceToHost);
    return _midError;
}

__host__ void Neuron::propMidErrorForward(int _index, double _value){
    assert((_index>=0)&&(_index<getNInputs()));
    gpu_setValueInArray<<<1,1>>>(_value, _index, inputMidErrors);
}




__host__ void Neuron::propMidErrorBackward(double _nextSum){
    //TODO needs test
    gpu_propError<<<1,1>>>(_nextSum, sum, actMet, midError);
}

//*************************************************************************************
//exploding/vanishing gradient:
//*************************************************************************************

//TODO getError

//*************************************************************************************
//learning
//*************************************************************************************

__host__ double Neuron::getBackwardsCoeff(){
    double _backwardsCoeff = 0.0;
    cudaMemcpy(&_backwardsCoeff, backwardsCoeff, sizeof(double), cudaMemcpyDeviceToHost);
    return _backwardsCoeff;
}

__host__ double Neuron::getWeight(int index) {
    double _weight = 0.0;

    double* weight = weights + index;
    cudaMemcpy(&_weight, weight, sizeof(double), cudaMemcpyDeviceToHost);
    return _weight;
}

//*************************************************************************************
//global settings
//*************************************************************************************

//TODO setGlobalError

//TODO getGlobalError

//TODO setEchoError

//TODO echoErrorForward

//TODO calcEchoError

//*************************************************************************************
//local backpropagation of error
//*************************************************************************************

//TODO setLocalError

//TODO propGlobalErrorBackwardLocally

//TODO getLocalError

//*************************************************************************************
// getters:
//*************************************************************************************

__host__ double Neuron::getOutput(){
    double _output=0.0;
    cudaMemcpy(&_output, output, sizeof(double), cudaMemcpyDeviceToHost);
    return _output;
}

//added by luca 

__host__ double Neuron::getOutput_no_memcpy(double* Pinned,double* Pageable){
    *Pinned = *output;
    printf("Pinned: %p",Pinned);
    //memcpy(&Pageable,Pinned,sizeof(double));
    return(0);
}

__host__ double Neuron::getSumOutput(){
    double _sum=0;
    cudaMemcpy(&_sum, sum, sizeof(double), cudaMemcpyDeviceToHost);
    return _sum;
}

__host__ double Neuron::getMaxWeight(){
    double _maxWeight=0;
    cudaMemcpy(&_maxWeight, maxWeight, sizeof(double), cudaMemcpyDeviceToHost);
    return _maxWeight;
}

__host__ double Neuron::getMinWeight(){
    double _minWeight=0;
    cudaMemcpy(&_minWeight, minWeight, sizeof(double), cudaMemcpyDeviceToHost);
    return _minWeight;
}

__host__ double Neuron::getSumWeight(){
    double _weightSum=0;
    cudaMemcpy(&_weightSum, weightSum, sizeof(double), cudaMemcpyDeviceToHost);
    return _weightSum;
}


//double Neuron::getWeightChange(){
//    weightsDifference = 0;
//    weightChange = 0;
//    for (int i=0; i<nInputs; i++){
//        weightsDifference = weights[i] - initialWeights[i];
//        weightChange += pow(weightsDifference,2);
//    }
//    return (weightChange);
//}

//TODO getWeightDistance

__host__ int Neuron::getNInputs(){
    int _nInputs=0;
    cudaMemcpy(&_nInputs, nInputs, sizeof(int), cudaMemcpyDeviceToHost);
    return _nInputs;
}


//TODO getWeights

//TODO getInitWeights

//*************************************************************************************
//saving and inspecting
//*************************************************************************************

//TODO saveWeights

//TODO printNeuron

//*************************************************************************************
//helper host functions:
//*************************************************************************************
__host__ void gpu_allocateInt(int** pointer, int value){
    cudaMalloc(pointer, sizeof(int));
    gpu_setInt<<<1,1>>>(*pointer, value);
}

__host__ void gpu_allocateDouble(double** pointer, double value){
    cudaMalloc(pointer, sizeof(double));
    gpu_setDouble<<<1,1>>>(*pointer, value);
}

//*************************************************************************************
//device CUDA kernels:
//*************************************************************************************

__device__ void device_propError(double _value, double* sum, int* actMet, double* errorLocation){
    double output = 0;
    device_doActivationPrime(&output, sum, actMet);
    *errorLocation = _value * output;
}

__device__ void device_doActivation(double* output, double* sum, int* actMet) {
    switch(*actMet){
        case 0:
            *output = (1/(1+(exp(-*sum)))) - 0.5;
            break;
        case 1:
            *output = tanh(*sum);
            break;
        case 2:
            *output = *sum;
            break;
    }
}

__device__ void device_doActivationPrime(double* output, double* input, int* actMet){
    switch(*actMet){
        case 0:
            device_doActivation(output, input, actMet);
            *output = 1 * (0.5 + *output) * (0.5 - *output); //exp(-_input) / pow((exp(-_input) + 1),2);
            break;
        case 1:
            *output = 1 - pow(tanh(*input), 2.0);
            break;
        case 2:
            *output = 1;
            break;
    }
}

//*************************************************************************************
//global CUDA kernels:
//*************************************************************************************
__global__ void gpu_propError(double _value, double* sum, int* actMet, double* errorLocation) {
    device_propError(_value, sum, actMet, errorLocation);
}


__global__ void gpu_setValuesInArray(double _value, double* list){
    list[threadIdx.x] = _value;
}

__global__ void gpu_setValueInArray(double _value, int index, double* list){
    list[index] = _value;
}

__global__ void gpu_getSumAndMaxMin(double* sum, double* max_list, double* list_min, double* list, int length){
    for (int i=0; i<length; i++){
        *sum = *sum + fabs(list[i]);
        *max_list = max(*max_list, list[i]);
        *list_min = min(*list_min, list[i]);
    }
}


__global__ void gpu_setInt(int* pointer, int value) {
    *pointer = value;
}

__global__ void gpu_setDouble(double* pointer, double value){
    *pointer = value;
}

__global__ void gpu_doActivation(double* output, double* sum, int* actMet) {
    device_doActivation(output, sum, actMet);
}

__global__ void gpu_doActivationPrime(double* output, double* input, int* actMet) {
    device_doActivationPrime(output, input, actMet);
}

__global__ void gpu_dotProduct(double* list1, double* list2, double* _value, double* _target, int arrayLength){
    int idx = threadIdx.x;
    int stride = blockDim.x;

    double target = 0.0;
    for (int i = idx; i < arrayLength; i+=stride){
        target += list1[i]*list2[i];
    }

    _value[idx] = target;
    __syncthreads();

    for (int size = stride/2; size>0; size/=2){
        if (idx < size){
            _value[idx] += _value[idx+size];
        }
        __syncthreads();
    }
    if (idx == 0){
        *_target = _value[0];
    }
}

__device__ void setSum_zero(Neuron* n){
    double* _sum = (*n).sum;
    double target = 0.0;
    *_sum = target;
}


__device__ void device_dotProduct(double* list1, double* list2, double* _target, int arrayLength, double* _storageArray,int e_idx){

    //double target = 0.0;

    int idx = threadIdx.x;

     _storageArray[idx] = 0.0;

    for (int i = e_idx; i < arrayLength; i+=blockDim.x){
        _storageArray[idx]+=list1[i]*list2[i];

    }
   
   
    //_storageArray[threadIdx.x] = target;

}

__global__ void gpu_multiplication(double value, double* output){
    *output = value * *output;
}
