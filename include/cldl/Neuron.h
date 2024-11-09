#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <chrono>

using namespace std;

/**
 * This is the class for creating neurons inside the Layer class.
 * This is the building block class of the network.
 */

class Neuron {
public:



    /**
     * Constructor for the Neuron class: it initialises a neuron with specific number fo inputs to that neuron
     * @param _nInputs
     */
    Neuron(int _nInputs);
    /**
     * Destructor
     * De-allocated any memory
     */
    ~Neuron();

    /**
        * Options for method of initialising biases
        * 0 for initialising all weights to zero
        * 1 for initialising all weights to one
        * 2 for initialising all weights to a random value between 0 and 1
        */
    enum biasInitMethod { B_NONE = 0, B_RANDOM = 1 };

    /**
     * Options for method of initialising weights
     * 0 for initialising all weights to zero
     * 1 for initialising all weights to one
     * 2 for initialising all weights to a random value between 0 and 1
     */
    enum weightInitMethod { W_ZEROS = 0, W_ONES = 1, W_RANDOM = 2 };

    /**
     * Options for activation functions of the neuron
     * 0 for using the logistic function
     * 1 for using the hyperbolic tan function
     * 2 for unity function (no activation)
     */
    enum actMethod {Act_Sigmoid = 0, Act_Tanh = 1, Act_NONE = 2};

    /**
     * Options for choosing an error to monitor the gradient of
     * 0 for monitoring the error that propagates backward
     * 1 for monitoring the error that propagates from the middle and bilaterally
     * 2 for monitoring the error that propagates forward
     */
    enum whichError {onBackwardError = 0, onMidError = 1, onForwardError = 2};

    /**
     * Initialises the neuron with the given methods for weight/bias initialisation and for activation function.
     * It also specifies the index of the neuron and the index of the layer that contains this neuron.
     * @param _neuronIndex The index of this neuron
     * @param _layerIndex The index of the layer that contains this neuron
     * @param _wim The method of initialising the weights, refer to weightInitMethod for more information
     * @param _bim The method of initialising the biases, refer to biasInitMethod for more information
     * @param _am The function used for activation of neurons, refer to actMethod for more information
     */
    __host__ void initNeuron(int _neuronIndex, int _layerIndex, weightInitMethod _wim, biasInitMethod _bim, actMethod _am);

    //Forward Propagation of inputs:
    __host__ void setInput(int _index,  double _value);
    __host__ void propInputs(int _index,  double _value);
    __host__ double getInput(int index);

    /** Sets the learning rate.
     * @param _learningRate Sets the learning rate for this neuron.
     **/
    __host__ void setLearningRate(double _learningRate);
    __host__ double getLearningRate();
    __host__ int getNInputs();

    /**
     * Sets the error of the neuron in the first hidden layer that is to be propagated forward
     * @param _value value of the error
     */
    //Forward Propagation of errors:
    __host__ void setForwardError(double _value);
    __host__ double getInputError(int _index);
    /**
     * Sets the forward propagating error of the neuron in layers other than the first hidden layer
     * @param _index index of the error
     * @param _value value of the error
     */
    __host__ void propErrorForward(int _index, double _value);
    __host__ double doActivation(double _sum);


    __host__ double getForwardError();
    __host__ void calcForwardError();

    //Back Propagation of errors:
    __host__ void setBackwardError(double _leadError);

    __host__ double getBackwardError();
    __host__ double getErrorWeightProducts(int index);
    __device__ void echoErrorBackward(double _nextSum, Neuron* n);
    __host__ double getBackwardsCoeff();
    __host__ double getEchoError();


    //Mid Propagation of errors:
    __host__ void setMidError(double _leadMidError);
    __host__ double getInputMidErrors(int index);
    __host__ void calcMidError();
    __host__ double getMidError();
    __host__ void propMidErrorForward(int index, double value);
    __host__ void propMidErrorBackward(double _nextSum);

    // Getters
    __host__ double getOutput();
    __host__ double getSumOutput();
    __host__ double getMaxWeight();
    __host__ double getMinWeight();
    __host__ double getSumWeight();
    __host__ double getWeightChange();
    __host__ double getWeight(int index);

     //added by luca, to remove memcpy from get output
    __host__ double getOutput_no_memcpy(double* Pinned,double* Pageable);


// initialisation:
public:

    //added by luca to store dot product results
    //double* array_for_dot_product_sum;

    int *nInputs;
    double *learningRate;
    int *myLayerIndex;
    int *myNeuronIndex;
    double *initialWeights;


    int *iHaveReported;

    //forward propagation of inputs:
    double *inputs;
    double *bias;
    double *sum;
    double *output;

    //forward propagation of error:
    double *inputErrors;
    double *forwardError;
    double *calcForwardOutput;

    //back propagation of error
    double *backwardError;
    double *ErrorWeightProducts;

    //mid propagation of error
    double *inputMidErrors;
    double *midError;

    //learning:
    double *backwardsCoeff;
    double *midCoeff;
    double *forwardCoeff;
    double *globalCoeff;
    double *weights;
    double *weightSum;
    double *maxWeight;
    double *minWeight;
    double *weightChange;
    double *weightsDifference;
    int *actMet;

    //global setting
    double *globalError;
    double *localError;
    double *echoCoeff;
    double *localCoeff;

    double *overallError;
    double *echoError;
    double *echoErrors;

private:


};


__device__ void device_setLearningRate(Neuron* n, double _learningRate);
__device__ void device_setBackwardError(double _leadError, Neuron* n);
__device__ void device_propErrorBackward(double _nextSum, Neuron* n);

//Cuda Kernels
__global__ void gpu_setValuesInArray(double _value, double* list);
__global__ void gpu_setValueInArray(double _value, int index, double* list);
__global__ void gpu_getSumAndMaxMin(double* sum, double* max_list, double* list_min, double* list, int length);

__global__ void gpu_propError(double _value, double* sum, int* actMet, double* errorLocation);
__device__ void device_propError(double _value, double* sum, int* actMet, double* errorLocation);


__host__ void gpu_allocateInt(int** pointer, int value);
__global__ void gpu_setInt(int* pointer, int value);

__host__ void gpu_allocateDouble(double** pointer, double value);
__global__ void gpu_setDouble(double* pointer, double value);

__global__ void gpu_doActivation(double *output, double *_sum, int *actMet);
__global__ void gpu_doActivationPrime(double *output, double *_input, int *actMet);


__device__ void device_doActivation(double* output, double *_sum, int* actMet);

__device__ void device_doActivationPrime(double* output, double *_sum, int* actMet);


__global__ void gpu_dotProduct(double* list1, double* list2, double* _value, double* _target, int arrayLength);
__device__ void device_dotProduct(double* list1, double* list2, double* _target, int arrayLength, double* _storageArray);

__device__ void device_calcForwardError(Neuron* n);
__global__ void gpu_calcForwardError(Neuron* n);

__device__ void echoErrorBackward(double _nextSum, Neuron* n);
__global__ void gpu_echoErrorBackward(double _nextSum, Neuron* n);

__device__ void propErrorBackward(double _nextSum, Neuron* n);
__global__ void gpu_propErrorBackward(double _nextSum, Neuron* n);

__global__ void gpu_setBackwardError(double _leadError, Neuron* n);

__device__ void device_calcMidError(Neuron *n);

__global__ void gpu_multiplication(double value, double* output);

__device__ void device_calcOutput(Neuron* n,int* _layerHasReported);
__device__ void device_calcOutputCont(Neuron* n, int* _layerHasReported);


//added by luca 
    //function to allow for sum to be set
    __device__ void setSum_zero(Neuron* n);

    //added by luca to sum the temp array from dot product
    //__device__ void device_sum_tempArray(Neuron* n);

    __device__ void parallelReduction(Neuron* n, double * _array_for_sum);

    __device__ void warpReduce(double* shmem_ptr, int t);

    //added by luca, functions for errorWeightProductSum

    __device__ void device_calcErrorWeightProductSum_less_blocks(Neuron* n, int nInputs, double* sumlist,int j);

    __device__ void parallelReduction_weights(int j,double* _array_for_dot_sum,double * sumlist);

   

    //end of added by luca





