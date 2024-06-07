#include <cstdio>
#include <fz/array.cuh>

template <typename T, std::size_t S>
__global__ void kernelTestArrayHD(fz::cuda::Array<T, S> *a) {
  auto thread_id = threadIdx.x * blockDim.x + threadIdx.y;

  if (thread_id == 0) {
    printf("Running in the device\n");
    printf("==========Array Basic Info==========\n");
    printf("Array size: %lu\n", a->size());
    printf("==========End of Array Basic Info==========\n");

    printf("Data from the host\n");

    printf("==========Array Data==========\n");
    for (const auto &item : *a) {
      printf("%f ", item);
    }
    printf("\n");
    printf("==========End of Array Data==========\n");
  }

  __syncthreads();

  auto size = a->size() / (blockDim.x * blockDim.y);
  for (std::size_t i = thread_id * size; i < (thread_id + 1) * size; ++i) {
    (*a)[i] = (*a)[i] - thread_id;
  }

  __syncthreads();

  if (thread_id == 0) {
    printf("Data changed in the device\n");
    printf("==========Array Data==========\n");
    for (const auto &item : *a) {
      printf("%f ", item);
    }
    printf("\n");
    printf("==========End of Array Data==========\n");

    printf("End of the device\n");
  }
}

auto testArrayHD() {
  printf(
      "=======================TEST ARRAY HOST DEVICE=======================\n");

  printf("Running in the host\n");

  auto hd = fz::cuda::ArrayHD<float, 8>();
  hd.allocateHost();
  hd.allocateDevice();
  auto host = hd.host();
  auto device = hd.device();

  // init Host Array
  for (std::size_t i = 0; i < host->size(); ++i) {
    (*host)[i] = host->size() - i;
  }

  printf("==========Array Data==========\n");
  for (const auto &item : *host) {
    printf("%f ", item);
  }
  printf("\n");
  printf("==========End of Array Data==========\n");

  printf("\n");
  auto grid = dim3(1, 1, 1);
  auto block = dim3(2, 2, 1);
  hd.copyHostToDevice();
  kernelTestArrayHD<<<grid, block>>>(device);
  cudaDeviceSynchronize();
  printf("\n");

  printf("Back to Host\n");

  printf("==========Host Array Data==========\n");
  for (const auto &item : *host) {
    printf("%f ", item);
  }
  printf("\n");
  printf("==========End of Host Array Data==========\n");

  printf("Copy Device to Host\n");
  hd.copyDeviceToHost();
  printf("==========Host Array Data==========\n");
  for (const auto &item : *host) {
    printf("%f ", item);
  }
  printf("\n");
  printf("==========End of Host Array Data==========\n");

  printf(
      "=======================END OF TEST ARRAY HOST "
      "DEVICE=======================\n");
}

int main(int argc, char *argv[]) {
  testArrayHD();

  return 0;
}
