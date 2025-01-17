
#pragma once

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

#include "Neuron.h"

/**
 * This is the class for creating layers that are contained inside the Net class.
 * The Layer instances in turn contain neurons.
 */

class Layer {
public:

    /**
     * Constructor for Layer: it initialises the neurons internally.
     * @param _nNeurons Total number of neurons in the layer
     * @param _nInputs Total number of inputs to that layer
     */
    Layer(int _nNeurons, int _nInputs);

    /**
     * Destructor
     * De-allocated any memory
     **/
    ~Layer();

    /**
     * Options for what gradient of a chosen error to monitor
     */
    enum whichGradient {exploding = 0, average = 1, vanishing = 2};

    /**
     * Initialises each layer with specific methods for weight/bias initialisation and activation function of neurons
     * @param _layerIndex The index that is assigned to this layer by the Net class
     * @param _wim weights initialisation method,
     * see Neuron::weightInitMethod for different options
     * @param _bim biases initialisation method,
     * see Neuron::biasInitMethod for different options
     * @param _am activation method,
     * see Neuron::actMethod for different options
     */
    __host__ void initLayer(int _layerIndex, Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am);

    //Added by luca, Pinned memory generation function

    cudaError_t checkCuda(cudaError_t result);

    __host__ int* generating_pinned_memory_address(int nInputs);

    __host__ void setSum_zero();
    
    //end of added by luca 

    //added by luca to remove memcpy

    __host__ double getOutput_no_memcpy(int _neuronIndex);

    //added by luca, function to update weights for the first layer
    __host__ void updateWeights_first_layer();


    /** Sets the learning rate.
     * @param _learningRate Sets the learning rate for all neurons.
     **/
    __host__ void setlearningRate(double _learningRate);

    /**
     * Sets the inputs to all neurons in the first hidden layer only
     * @param _inputs A pointer to an array of inputs
     */
    __host__ void setInputs(double *_inputs);

    //added by luca
    //seting inputs only at the layer level to reduce number of operations 

    __host__ void setInputs_layer_level_only(double* _inputs);

    /**
     * Sets the inputs to all neurons in the deeper layers (excluding the first hidden layer)
     * @param _index The index of the input
     * @param _value The value of the input
     */
    __host__ void propInputs(double *_gpu_InputOutputs);

    /**
     * Demands that all neurons in this layer calculate their output
     */
    __host__ void calcOutputs(double* inputs_next_layer,int nNeurons_next_layer);

    /**
     * Sets the error to be propagated forward to all neurons in the first hidden layer only
     * @param _leadForwardError the error to be propagated forward
     */
    __host__ void setForwardError(double _leadForwardError);

    /**
     * Sets the error to be propagated forwards to all neurons in deeper layers
     * @param _index Index of input where the error originates form
     * @param _value The value of the error
     */
    __host__ void propErrorForward(int _index, double _value);

    /**
     * calculates the forward error by doing a weighed sum of forward errors and the weights
     */
    __host__ void calcForwardError();

    /**
     * Allows for accessing the forward error of a specific neuron.
     * @param _neuronIndex Index of the neuron to request the error from
     * @return Returns the forward error from the chosen neuron
     */
    __host__ double getForwardError(int _neuronIndex);

    /**
     * Sets the error to be propagated backward at all neurons in the output layer only.
     * @param _leadError the error to be propagated backward
     */
    __host__ void setBackwardError(double _leadError);

    //added by luca, setBackward error to input - output
    __host__ double setBackwardError_LMS(double input_signal);

    __host__ double* calcErrorWeightProductSum(Neuron* neuron_previous_layer,double* inputs_previous_layer);

    //updated version of set backward error to use on the final layer to prop and supply in a single launch
    __host__ double calcErrorWeightProductSum_LMS(Neuron* neuron_previous_layer,double* inputs_previous_layer,double input_signal);

    /**
     * Sets the error to be propagated backward at all neurons, except those in the output layer.
     * @param _neuronIndex The index of the neuron receiving the weighted sum of errors
     * @param _nextSum The weighted sum of propagating error
     */
    __host__ void propErrorBackward(double* _sumList);

    /**
     * Allows for accessing the error that propagates backward in the network
     * @param _neuronIndex The index from which the error is requested
     * @return Returns the error of the chosen neuron
     */
    __host__ double getBackwardError(int _neuronIndex);

    /**
     * Sets the middle error in all neurons in the chosen layer by Net
     * @param _leadMidError The error to be propagated bilaterally
     */
    __host__ void setMidError(double _leadMidError);

    /**
     * calculates the error to be propagated bilaterally
     */
    __host__ void calcMidError();

    /**
     * Allows for accessing the error that propagates bilaterally
     * @param _neuronIndex The index of the neuron that the error is requested from
     * @return Returns the mid error
     */
    __host__ double getMidError(int _neuronIndex);

    /**
     * Sets the mid error in all neurons of a chosen layer by Net
     * @param _index Index of the mid error
     * @param _value Value of the mid error
     */
    __host__ void propMidErrorForward(int _index, double _value);

    /**
     * Sets the mid error in all neurons of a specific layer chosen by Net
     * @param _neuronIndex The index of the neuron to receive the error
     * @param _nextSum The weighted sum of errors
     */
    __host__ void propMidErrorBackward(int _neuronIndex, double _nextSum);

    /**
     * It provides a measure of the magnitude of the error in this layer
     * to alarm for vanishing or exploding gradients.
     * \param _whichError choose what error to monitor, for more information see Neuron::whichError
     * \param _whichGradient choose what gradient of the chosen error to monitor,
     * for more information see Layer::whichGradient
     * @return Returns the chosen gradient in this layer
     */
    __host__ double getGradient(Neuron::whichError _whichError, whichGradient _whichGradient);

    /**
     * Sets the coefficient of the errors used for learning
     * @param _globalCoeff coefficient of the global error
     * @param _backwardsCoeff coefficient of the error propagating backward
     * @param _midCoeff coefficient of the error propagating bilaterally
     * @param _forwardCoeff coefficient of the error propagating forward
     * @param _localCoeff coefficient of the error propagating locally
     * @param _echoCoeff coefficient of the error resonating back and forth
     */
    __host__ void setErrorCoeff(double _globalCoeff, double _backwardsCoeff,
                                double _midCoeff, double _forwardCoeff,
                                double _localCoeff, double  _echoCoeff);

    /**
     * Requests that all neurons perform one iteration of learning
     */
    __host__ void updateWeights();

    //this method is for testing only
    __host__ void setWeights(double* _weightsList);

    /**
     * Sets the global error, all neurons will have access to this error
     * @param _globalError The global error
     */
    __host__ void setGlobalError(double _globalError);

    /**
     * Sets the local error at all neurons
     * @param _leadLocalError The error to be propagated locally only
     */
    __host__ void setLocalError(double _leadLocalError);

    /**
     * sets the error that propagates backwards and locally (for one layer only) for all neurons
     */
    __host__ void propGlobalErrorBackwardLocally(int _neuronIndex, double _nextSum);

    /**
     * Allows for accessing the local error of a specific neuron
     * @param _neuronIndex The index of the neuron to request the local error from
     * @return Returns the local error
     */
    __host__ double getLocalError(int _neuronIndex);

    /**
     * Sets the error to be resonated back and forth at all neurons
     * @param _echoError the resonating error
     */
    __host__ void setEchoError(double _clError);

    /**
     * Sets the resonating error for a specific neuron
     * @param _neuronIndex Index of the neurons receiving the error
     * @param _nextSum The weighted sum of propagating errors
     */
    __host__ void echoErrorBackward(int _neuronIndex, double _nextSum);

    /**
     * Allows for accessing the resonating error of a specific neuron
     * @param _neuronIndex The index of the neuron to reuquest the error form.
     * @return Returns the resonating error of the neuron
     */
    __host__ double getEchoError(int _neuronIndex);

    /**
     * Sets the resonating error for a specific neuron
     * @param _index the index of the incoming error
     * @param _value The value of the incoming error
     */
    __host__ void echoErrorForward(int _index, double _value);

    /**
     * Demands that all neurons calculate their resonating error
     */
    __host__ void calcEchoError();

    /**
     * Allows access to a specific neuron
     * @param _neuronIndex The index of the neuron to access
     * @return A pointer to that neuron
     */
    Neuron* getNeuron(int _neuronIndex);

    /**
     * Reports the number of neurons in this layer
     * @return The total number of neurons in this layer
     */
    __host__ int getnNeurons();

    /**
     * Allows for accessing the activation of a specific neuron
     * @param _neuronIndex The index of the neuron
     * @return the activation of that neuron
     */
    __host__ double* getOutputs();

    __host__ double getOutput(int _neuronIndex);

    __host__ double getErrorWeightProductSum(int index);

    /**
     * Allows for accessing the sum output of any specific neuron
     * @param _neuronIndex The index of the neuron to access
     * @return Returns the weighted sum of the inputs to that neuron
     */
    __host__ double getSumOutput(int _neuronIndex);

    /**
     * Allows for accessing any specific weights in the layer
     * @param _neuronIndex The index of the neuron containing that weight
     * @param _weightIndex The index of the input to which that weight is assigned
     * @return Returns the chosen weight
     */
    __host__ double getWeights(int _neuronIndex, int _weightIndex);

    /**
     * Accesses the total sum of weight changes of all the neurons in this layer
     * @return sum of weight changes all neurons
     */
    __host__ double getWeightChange();

    /**
     * Performs squared root on the weight change
     * @return The sqr of the weight changes
     */
    __host__ double getWeightDistance();

    /**
     * Reports the global error that is assigned to a specific neuron in this layer
     * @param _neuronIndex the neuron index
     * @return the value of the global error
     */
    __host__ double getGlobalError(int _neuronIndex);

    /**
     * Reports the initial value that was assigned to a specific weight at the initialisatin of the network
     * @param _neuronIndex Index of the neuron containing the weight
     * @param _weightIndex Index of the weight
     * @return
     */
    __host__ double getInitWeight(int _neuronIndex, int _weightIndex);

    /**
     * Saves the temporal weight change of all weights in all neurons into files
     */
    __host__ void saveWeights();

    __host__ void printWeights(FILE*);

    /**
     * Snaps the final distribution of weights in a specific layer,
     * this is overwritten every time the function is called
     */
    __host__ void snapWeights();

    /**
     * Prints on the console a full tree of this layer with the values of all weights and outputs for all neurons
     */
    __host__ void printLayer();

    //added by luca 
    //calcOutputs function that doesn't include propagation to the next layer
    __host__ void calcOutputs_final_layer();

    __host__ double single_block_launch(    Layer** layers,
                                            int nLayers,
                                            double input_signal,
                                            double* error,
                                            int* nNeurons_array,
                                            int* nNeurons_per_block_calcOut,
                                            int* nNeurons_per_block_calcErrorWeight,
                                            int* start_idx_calc_out,
                                            int* start_idx_calcErrorWeight,
                                            Neuron** gpu_neuron_pointers);
    

public:
    /**Added by Luca
    *constant variable to store the pageable and pinned memory locations
    */

    int *h_aPageable, *h_bPageable;   
    int *h_aPinned, *h_bPinned;

    //pinned and pageable for the inputs 

    double* inputs_a_Pageable;
    double* inputs_a_Pinned;

    //pinned and pageable for get output 
    double* get_output_Pageable;
    double* get_output_Pinned;

    //pinned and pageable for output array
    double* get_output_array_Pageable;
    double* get_output_array_Pinned;



   

    //constant global variables 
    const int threads_per_block = 128;
    const int max_blocks = 8;


     //variables for use in calculating how many neurons should be calculated per thread block 
    int start_idx_for_reduction;
    int number_of_concurrent_neurons_per_thread_block;

    //calc weight product sum variables
    int start_idx_for_reduction_calcWeightProduct_sum;
    int number_of_concurrent_neurons_per_thread_block_calcWeight_Product_sum;







    // initialisation:
    int nNeurons = 0;
    int nInputs = 0;
    double learningRate = 0;
    int myLayerIndex = 0;
    Neuron *neurons;
    Neuron *gpu_neurons;
    double* gpu_sumlist;
    
    int layerHasReported = 0;

    //forward propagation of inputs:
    double *inputs;
    double *gpu_inputs;
    double *gpu_weights;

    //forward propagation of error:
    double leadForwardError = 0;

    //back propagation of error:
    double leadBackwardError = 0;

    //mid propagation of error:
    double leadMidError = 0;

    //global settings
    double globalError = 0;
    double leadLocalError =0;

    //exploding vanishing gradient:
    double averageError = 0;
    double maxError = 0;
    double minError = 0;

    //learning:
    double weightChange=0;
};
