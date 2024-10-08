#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>


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

int main(){

  std::cout<<"This program is written to test the processing time of the cublas library\n\n\n";
  

  for(int i = 100;i<30000;i+=500){

    std::cout<<"Performing dot product of two vectors of length:  "<<i<<"\n";

    double *d_a, *d_b;

    const int ds = i;
    
    cudaMalloc(&d_a, sizeof(d_a[0])*ds);
    cudaMalloc(&d_b, sizeof(d_b[0])*ds);
    
    double *h = new double[ds];
    for (int i = 0; i < ds; i++) h[i] = 5;
    cudaMemcpy(d_a, h, sizeof(d_a[0])*ds, cudaMemcpyHostToDevice);
    for (int i = 0; i < ds; i++) h[i] = 2;
    cudaMemcpy(d_b, h, sizeof(d_b[0])*ds, cudaMemcpyHostToDevice);
    
    cublasHandle_t hd;
    cublasStatus_t stat = cublasCreate(&hd);

    int incy = 1;
    int incx = 1;

    double* result;
    double *res = new double;

    cudaMalloc(&result,sizeof(res));
    cudaMemcpy(result,res,sizeof(result),cudaMemcpyHostToDevice);
    
  //timing the dot product function 
    auto start = std::chrono::high_resolution_clock::now();
    cublasDdot(hd, ds ,d_a ,incx ,d_b ,incy ,result);
    auto total = std::chrono::high_resolution_clock::now()- start;

  //printing the time taken to perform the dot product
    float total_printable = std::chrono::duration_cast<std::chrono::microseconds>(
        total).count();

    float total_in_ms = total_printable/1000;
    float total_in_seconds = total_in_ms/(1000);

    int num_per_sec = 1/total_in_seconds;

    std::cout<<"Time To Calculate Dot--->"<<total_in_ms<<"ms\n";
    std::cout<<"Number Calculable Per Second--->"<<num_per_sec<<"\n\n";

 

    
    
  }


  //for (int i = 0; i < ds; i++) std::cout << h[i] << std::endl;
}

