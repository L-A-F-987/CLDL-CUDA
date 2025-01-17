
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

#include "Layer.h"

/** Net is the main class used to set up a neural network used for
 * Closed-Loop Deep Learning. It initialises all the layers and the
 * neurons internally.
 *
 * (C) 2019,2020, Bernd Porr <bernd@glasgowneuro.tech>
 * (C) 2019,2020, Sama Daryanavard <2089166d@student.gla.ac.uk>
 *
 * GNU GENERAL PUBLIC LICENSE
 */
class Net {

public:

    /**
     * Constructor: The neural network that performs the learning.
     * \param _nLayers Total number of hidden layers, excluding the input layer
     * \param _nNeurons A pointer to an int array with number of
     * neurons for all layers need to have the length of _nLayers.
     * \param _nInputs Number of Inputs to the network
     */
    Net(int _nLayers, int* _nNeurons, int _nInputs);

    /**
     * Destructor
     * De-allocated any memory
     */
    ~Net();

    /**
     * Dictates the initialisation of the weights and biases
     * and determines the activation function of the neurons.
     * \param _wim weights initialisation method,
     * see Neuron::weightInitMethod for different options
     * \param _bim biases initialisation method,
     * see Neuron::biasInitMethod for different options
     * \param _am activation method,
     * see Neuron::actMethod for different options
     */


    __host__ void initNetwork(Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am);

    /**
     * Sets the learning rate.
     * \param _learningRate Sets the learning rate for
     * all layers and neurons.
     */
    __host__ void setLearningRate(double _learningRate);


    //added by luca 
    //combining prop and update inputs for LMS mode 
    __host__ double single_block_lms(double* _inputs,double input_signal);

    /**
     * Sets the inputs to the network in each iteration
     * of learning, needs to be placed in an infinite loop.
     * @param _inputs A pointer to the array of inputs
     */
    __host__ void setInputs(double *_inputs);

    /**
     * It propagates the inputs forward through the network.
     */
    __host__ void propInputs();

    /**
     * Sets the error at the input layer to be propagated forward.
     * @param _leadForwardError The closed-loop error for learning
     */
    __host__ void setForwardError(double _leadForwardError);

    /**
     * Propagates the _leadForwardError forward through the network.
     */
    __host__ void propErrorForward();

    /**
     * Sets the error at the output layer to be propagated backward.
     * @param _leadError The closed-loop error for learning
     */
    __host__ void setBackwardError(double _leadError);

    /**
     * Propagates the _leadError backward through the network.
     */
    __host__ void propErrorBackward();

    /**
     * Sets the close-loop error to the a chosen layer to be propagated bilaterally.
     * @param _layerIndex The index of the layer at which to inject the error
     * @param _leadMidError The closed-loop error for learning
     */
    __host__ void setMidError(int _layerIndex, double _leadMidError);

    /**
     * Propagates the _leadMidError from the chosen layer forward to the output layer.
     */
    __host__ void propMidErrorForward();

    /**
     * Propagates the _leadMidError from the chosen layer backward to the input layer.
     */
    __host__ void propMidErrorBackward();

    /**
     * It provides a measure of how the magnitude of the error changes through the layers
     * to alarm for vanishing or exploding gradients.
     * \param _whichError choose what error to monitor, for more information see Neuron::whichError
     * \param _whichGradient choose what gradient of the chosen error to monitor,
     * for more information see Layer::whichGradient
     * @return Returns the ratio of the chosen gradient in the last layer to the the first layer
     */
    __host__ double getGradient(Neuron::whichError _whichError, Layer::whichGradient _whichGradient);

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
     * Requests that all layers perform one iteration of learning
     */
    __host__ void updateWeights();

    // this is only for testing
    __host__ void setWeights(double* _weightsList);

    /**
     * Sets the global error, all layers and neurons will have access to this error
     * @param _globalError The global error
     */
    __host__ void setGlobalError(double _globalError);

    /**
     * Sets the error to be resonated back and forth in the network
     * @param _echoError the resonating error
     */
    __host__ void setEchoError(double _echoError);

    /**
     * Propagates the resonating error backward through the network
     */
    __host__ void echoErrorBackward();

    /**
     * propagates the resonating error forward through the network
     */
    __host__ void echoErrorForward();

    /**
     * It propagates the resonating error back and forth through the network
     * using the echoErrorBackward and echoErrorForward until the residue error is zero
     * @param _theError The error used for resonating
     */
    __host__ void doEchoError(double _theError);

    /**
     * Sets the local error at every layer
     * @param _leadLocalError The error to be propagated locally only
     */
    __host__ void setLocalError(double _leadLocalError);

    /**
     * propagates the local error backwards and locally (for one layer only)
     */
    __host__ void propGlobalErrorBackwardLocally();

    /**
     * Allows Net to access each layer
     * @param _layerIndex the index of the chosen layer
     * @return A pointer to the chosen Layer
     */
    __host__ Layer *getLayer(int _layerIndex);

    /**
     * Allows the user to access the activation output of a specific neuron in the output layer only
     * @param _neuronIndex The index of the chosen neuron
     * @return The value at the output of the chosen neuron
     */
    __host__ double getOutput(int _neuronIndex);

    __host__ double getOutput_no_memcpy(int _neuronIndex);

    /**
     * Allows the user to access the weighted sum output of a specific neuron in output layer only
     * @param _neuronIndex The index of the chosen neuron
     * @return The value at the sum output of the chosen neuron
     */
    __host__ double getSumOutput(int _neuronIndex);

    /**
     * Informs on the total number of hidden layers (excluding the input layer)
     * @return Total number of hidden layers in the network
     */
    __host__ int getnLayers();

    /**
     * Informs on the total number of inputs to the network
     * @return Total number of inputs
     */
    __host__ int getnInputs();

    /**
     * Informs on the total number of outputs from the network
     * @return Total number of outputs
     */
    __host__ int getnOutputs();

    /**
     * Allows for monitoring the overall weight change of the network.
     * @return returns the Euclidean wight distance of all neurons in the network from their initial value
     */
    __host__ double getWeightDistance();

    /**
     * Allows for monitoring the weight change in a specific layer of the network.
     * @param _layerIndex The index of the chosen layer
     * @return returns the Euclidean wight distance of neurons in the chosen layer from their initial value
     */
    __host__ double getLayerWeightDistance(int _layerIndex);

    /**
     * Grants access to a specific weight in the network
     * @param _layerIndex Index of the layer that contains the chosen weight
     * @param _neuronIndex Index of the neuron in the chosen layer that contains the chosen weight
     * @param _weightIndex Index of the input to which the chosen weight is assigned
     * @return returns the value of the chosen weight
     */
    __host__ double getWeights(int _layerIndex, int _neuronIndex, int _weightIndex);

    /**
     * Informs on the total number of neurons in the network
     * @return The total number of neurons
     */
    __host__ int getnNeurons();

    /**
     * Saves the temporal changes of all weights in all neurons into files
     */
    __host__ void saveWeights();

    __host__ void printInitialWeights();

    __host__ void printWeights();

    /**
     * Snaps the final distribution of all weights in a specific layer,
     * this is overwritten every time the function is called
     */
    __host__ void snapWeights();

    /**
     * Prints on the console a full tree of the network with the values of all weights and outputs for all neurons
     */
    __host__ void printNetwork();


    //added by luca 
    __host__ double setBackwardError_LMS(double _input_signal);


    __host__ double set_and_propErrorBackward_lms(double _input_signal) ;

private:

    //added by luca, storing the number of neurons in each layer in an array so it can be passed to the single block function 
    int* nNeurons_array;
    int* nNeurons_array_pageable;

    Neuron** gpu_neuron_pointers;

    //storing the start indexes and n_neurons per block 

    int* nNeurons_per_block_calcOut;
    int* nNeurons_per_block_calcErrorWeight;

    int* start_idx_calcOut;
    int* start_idx_calcErrorWeight;

    /**
     * Total number of hidden layers
     */
    int nLayers = 0;
    /**
     * total number of neurons
     */
    int nNeurons = 0;
    /**
     * total number of weights
     */
    int nWeights = 0;
    /**
     * total number of inputs
     */
    int nInputs = 0;
    /**
     * total number of outputs
     */
    int nOutputs = 0;
    /**
     * the learning rate
     */
    double learningRate = 0;
    /**
     * A double pointer to the layers in the network
     */
    Layer **layers = 0;
    /**
     * A pointer to the inputs of the network
     */
    double *inputs = 0;
    /**
     * The error to be propagated forward
     */
    double leadForwardError = 0;
    /**
     * The error to be propagated backward
     */
    double theLeadError = 0;
    /**
     * Index of the layer at which the mid error is injected
     */
    int midLayerIndex = 0;
    /**
     * The error to be propagated bilaterally
     */
    double theLeadMidError = 0;
    /**
     * A pointer to the gradient of the error
     */
    double *errorGradient = NULL;
    /**
     * The global error that is passed to every neuron
     */
    double globalError = 0;
    /**
     * The error to be propagated back and forth through the network
     */
    double echoError = 0;
    /**
     * The error to be propagated locally
     */
    double theLeadLocalError = 0;

};
