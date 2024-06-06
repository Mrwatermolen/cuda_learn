#include <cuda_runtime.h>
#include <stdio.h>

void sumArrays(float *a, float *b, float *res, const int size) {
  for (int i = 0; i < size; i += 4) {
    res[i] = a[i] + b[i];
    res[i + 1] = a[i + 1] + b[i + 1];
    res[i + 2] = a[i + 2] + b[i + 2];
    res[i + 3] = a[i + 3] + b[i + 3];
  }
}
__global__ void sumArraysGPU(float *a, float *b, float *res) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  res[i] = a[i] + b[i];
}

int main(int argc, char **argv) {
  int dev = 0;
  cudaSetDevice(dev);

  int nElem = 1 << 14;
  printf("Vector size:%d\n", nElem);
  int nByte = sizeof(float) * nElem;
  float *a_h = (float *)malloc(nByte);
  float *b_h = (float *)malloc(nByte);
  float *res_h = (float *)malloc(nByte);
  float *res_from_gpu_h = (float *)malloc(nByte);
  memset(res_h, 0, nByte);
  memset(res_from_gpu_h, 0, nByte);

  float *a_d, *b_d, *res_d;
  // pine memory malloc
  (cudaMallocHost(&a_d, nByte));
  (cudaMallocHost(&b_d, nByte));
  (cudaMallocHost(&res_d, nByte));

  for (int i = 0; i < nElem; i++) {
    a_h[i] = i;
    b_h[i] = i * 2;
  }

  (cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
  (cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));

  dim3 block(1024);
  dim3 grid(nElem / block.x);
  sumArraysGPU<<<grid, block>>>(a_d, b_d, res_d);
  printf("Execution configuration<<<%d,%d>>>\n", grid.x, block.x);

  (cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));
  sumArrays(a_h, b_h, res_h, nElem);

  // print result
  for (int i = 0; i < nElem; i++) {
    // if (res_h[i] != res_from_gpu_h[i]) {
    //   printf("mismatch at pos %d, host:%f, gpu:%f\n", i, res_h[i],
    //          res_from_gpu_h[i]);
    //   break;
    // }
    printf("host:%f, gpu:%f\n", res_h[i], res_from_gpu_h[i]);
  }

  cudaFreeHost(a_d);
  cudaFreeHost(b_d);
  cudaFreeHost(res_d);

  free(a_h);
  free(b_h);
  free(res_h);
  free(res_from_gpu_h);

  return 0;
}