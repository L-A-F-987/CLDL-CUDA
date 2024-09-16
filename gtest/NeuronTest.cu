#include "cldl/Neuron.h"
#include "gtest/gtest.h"
#include <cuda_runtime.h>
using namespace std;


__global__ void changeNInputs(Neuron* n){
    *(*n).nInputs = 2;
}


TEST(CUDATest, testObjectPointerCalls){
    Neuron* n = new Neuron(1);
    Neuron* d_n;

    cudaMalloc((void**) &d_n, sizeof(Neuron));

    cudaMemcpy(d_n, n, sizeof(Neuron), cudaMemcpyHostToDevice);

    changeNInputs<<<1,1>>>(d_n);
    ASSERT_EQ(n->getNInputs(), 2);

}

TEST(CUDATest, testCudaMalloc){
    double *d_a;
    double a = 2.0;
    cudaMallocManaged(&d_a, sizeof(double));
    gpu_setDouble<<<1,1>>>(d_a, 1.0);
    cudaMemcpy(&a, d_a, sizeof(double), cudaMemcpyDeviceToHost);

    ASSERT_EQ(a, 1.0);

}

TEST(NeuronTest, testNeuronIntialisationAndNInputs){
    Neuron *n;
    n = new Neuron(1);
    ASSERT_EQ(n->getNInputs(), 1);

}



TEST(NeuronTest, testSetLearningRate){
    Neuron *n;
    n = new Neuron(1);
    n->setLearningRate(2.0);
    ASSERT_EQ(n->getLearningRate(), 2.0);
}

TEST(NeuronTest, testSetInputs){
    Neuron *n;
    n = new Neuron(3);
    n->setInput(0,2.0);
    n->setInput(1,4.0);
    n->setInput(2,6.0);

    n->propInputs(1,3.0);

    ASSERT_EQ(n->getInput(0), 2.0);
    ASSERT_EQ(n->getInput(1), 3.0);
    ASSERT_EQ(n->getInput(2), 6.0);
}

TEST(NeuronTest, testSetInputErrors) {
    Neuron *n;
    n = new Neuron(4);
    n->setForwardError(2.0);
    n->propErrorForward(2, 4.0);
    ASSERT_EQ(n->getInputError(1), 2.0);
    ASSERT_EQ(n->getInputError(2), 4.0);

}


TEST(NeuronTest, testSumAndMaxMin){
    double *sum, *d_sum, *max, *d_max, *min, *d_min, *list, *d_list;

    gpu_allocateDouble(&d_sum, 0.0);
    gpu_allocateDouble(&d_max, 1.0);
    gpu_allocateDouble(&d_min, 1.0);


    cudaMalloc((void**)&d_list, sizeof(double)*4);

    list = new double[4];
    list[0] = 0.5;
    list[1] = 1.0;
    list[2] = 1.5;
    list[3] = 2.0;


    sum = (double*)malloc(sizeof(double));
    max = (double*)malloc(sizeof(double));
    min = (double*)malloc(sizeof(double));

    cudaMemcpy(d_list, list, sizeof(double)*4, cudaMemcpyHostToDevice);

    gpu_getSumAndMaxMin<<<1,1>>>(d_sum, d_max, d_min, d_list, 4);
    cudaMemcpy(sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(max, d_max, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(min, d_min, sizeof(double), cudaMemcpyDeviceToHost);

    ASSERT_EQ(*sum, 5.0);
    ASSERT_EQ(*max, 2.0);
    ASSERT_EQ(*min, 0.5);

}

TEST(NeuronTest, TestDoActivation){
    double *d_result, *d_sum;
    double result = 0;
    int *actMet;

    gpu_allocateDouble(&d_result, 0.0);
    gpu_allocateDouble(&d_sum, 2.0);
    gpu_allocateInt(&actMet, 1);

    gpu_doActivation<<<1,1>>>(d_result, d_sum, actMet);
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_DOUBLE_EQ(result, 0.9640275800758169);

    gpu_setDouble<<<1,1>>>(d_result, 0.0);
    gpu_setDouble<<<1,1>>>(d_sum, 2.0);
    gpu_setInt<<<1,1>>>(actMet, 2);

    gpu_doActivation<<<1,1>>>(d_result, d_sum, actMet);
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_DOUBLE_EQ(result, 2.0);

    gpu_setDouble<<<1,1>>>(d_result, 0.0);
    gpu_setDouble<<<1,1>>>(d_sum, 2.0);
    gpu_setInt<<<1,1>>>(actMet, 0);

    gpu_doActivation<<<1,1>>>(d_result, d_sum, actMet);
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_DOUBLE_EQ(result, 0.38079707797788231);

}

TEST(NeuronTest, TestDoActivationPrime){
    double *d_result, *d_sum;
    double result = 0;
    int *actMet;

    gpu_allocateDouble(&d_result, 0.0);
    gpu_allocateInt(&actMet, 0);
    gpu_allocateDouble(&d_sum, 2.0);

    gpu_doActivationPrime<<<1,1>>>(d_result, d_sum, actMet);
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_DOUBLE_EQ(result, 0.10499358540350662);

    gpu_setDouble<<<1,1>>>(d_result, 0.0);
    gpu_setDouble<<<1,1>>>(d_sum, 2.0);
    gpu_setInt<<<1,1>>>(actMet, 1);

    gpu_doActivationPrime<<<1,1>>>(d_result, d_sum, actMet);
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_DOUBLE_EQ(result, 0.070650824853164429);

    gpu_setDouble<<<1,1>>>(d_result, 0.0);
    gpu_setDouble<<<1,1>>>(d_sum, 2.0);
    gpu_setInt<<<1,1>>>(actMet, 2);

    gpu_doActivationPrime<<<1,1>>>(d_result, d_sum, actMet);
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_DOUBLE_EQ(result, 1.0);

}
TEST(NeuronTest, testDotProduct){
    double *d_list1, *d_list2, *list1, *list2, *d_value, *d_target, *target;

    gpu_allocateDouble(&d_target, 0.0);
    cudaMalloc((void**)&d_list1, sizeof(double)*4);
    cudaMalloc((void**)&d_list2, sizeof(double)*4);
    cudaMalloc((void**)&d_value, sizeof(double)*4);

    list1 = new double[4];
    list1[0] = 1.0;
    list1[1] = 1.0;
    list1[2] = 1.0;
    list1[3] = 1.0;

    list2 = new double[4];
    list2[0] = 1.0;
    list2[1] = 2.0;
    list2[2] = 3.0;
    list2[3] = 4.0;

    target = (double*)malloc(sizeof(double));

    cudaMemcpy(d_list1, list1, sizeof(double)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_list2, list2, sizeof(double)*4,cudaMemcpyHostToDevice);

    gpu_dotProduct<<<1,2>>>(d_list1, d_list2, d_value, d_target, 4);

    cudaMemcpy(target, d_target, sizeof(double), cudaMemcpyDeviceToHost);

    ASSERT_EQ(*target, 10.0);
}

TEST(NeuronTest, testSetMidError){
    Neuron *n;
    n = new Neuron(4);
    n->setMidError(2.0);
    ASSERT_EQ(n->getInputMidErrors(1), 2.0);
    n->setMidError(3.0);
    ASSERT_EQ(n->getInputMidErrors(1), 3.0);
}

TEST(NeuronTest, testCalcMidError){
    Neuron *n;
    n = new Neuron(4);
    n->setMidError(2.0);
    n->calcMidError();
    ASSERT_EQ(n->getMidError(), 0.0);
}

TEST(NeuronTest, testSetAndGetBackwardError){
    Neuron* n = new Neuron(4);
    Neuron* d_n;

    cudaMalloc((void**) &d_n, sizeof(Neuron));

    cudaMemcpy(d_n, n, sizeof(Neuron), cudaMemcpyHostToDevice);

    gpu_setBackwardError<<<1,4>>>(2.0, d_n);

    ASSERT_EQ(n->getBackwardError(),0.5);
}

TEST(NeuronTest, testGPUMultiplication){
    double *output;
    double result = 0;
    gpu_allocateDouble(&output, 1.0);
    gpu_multiplication<<<1,1>>>(2.0, output);
    cudaMemcpy(&result, output, sizeof(double), cudaMemcpyDeviceToHost);

    ASSERT_EQ(result, 2.0);


}


TEST(NeuronTest, testEchoErrorBackward){
    Neuron* n = new Neuron(1);
    Neuron* d_n;

    cudaMalloc((void**) &d_n, sizeof(Neuron));

    cudaMemcpy(d_n, n, sizeof(Neuron), cudaMemcpyHostToDevice);

    gpu_echoErrorBackward<<<1,1>>>(2.0, d_n);

    ASSERT_EQ(n->getEchoError(),0.5);

}

TEST(NeuronTest, testPropErrorBackward){
    Neuron* n = new Neuron(1);
    Neuron* d_n;

    cudaMalloc((void**) &d_n, sizeof(Neuron));

    cudaMemcpy(d_n, n, sizeof(Neuron), cudaMemcpyHostToDevice);

    gpu_propErrorBackward<<<1,1>>>(2.0, d_n);

    ASSERT_EQ(n->getBackwardError(),0.5);

}

TEST(NeuronTest, testNeuronInit){
    Neuron* n = new Neuron(2);
    Neuron* d_n;
    cudaMalloc((void**) &d_n, sizeof(Neuron));
    cudaMemcpy(d_n, n, sizeof(Neuron), cudaMemcpyHostToDevice);

    n->initNeuron(0,0, Neuron::W_RANDOM, Neuron::B_NONE, Neuron::Act_NONE);
    ASSERT_FALSE(n->getWeight(0) == n->getWeight(1));
    printf("%f\n", n->getWeight(0));
    printf("%f\n", n->getWeight(1));
    ASSERT_TRUE(n->getWeight(0)<=1);
    ASSERT_TRUE(n->getWeight(1)<=1);
    ASSERT_TRUE(n->getWeight(0)>0);
    ASSERT_TRUE(n->getWeight(1)>0);

}
TEST(NeuronTest, testCalcForwardError){
    Neuron* n = new Neuron(4);
    Neuron* d_n;

    cudaMalloc((void**) &d_n, sizeof(Neuron));
    cudaMemcpy(d_n, n, sizeof(Neuron), cudaMemcpyHostToDevice);

    n->setForwardError(2.0);
    gpu_calcForwardError<<<1,n->getNInputs()>>>(d_n);
    ASSERT_EQ(n->getForwardError(), 0);
}
