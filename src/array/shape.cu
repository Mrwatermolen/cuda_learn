#include <ffz/array.cuh>
#include <iostream>

__device__ void testShape() {
  auto shape = ffz::cuda::FixedShape<2, 3, 4>{};

  printf("Size: %lu\n", shape.size());
  printf("Total size: %lu\n", shape.capacity());
  printf("Stride 0: %lu\n", shape.stride(0));
  printf("Stride 1: %lu\n", shape.stride(1));
  printf("Stride 2: %lu\n", shape.stride(2));
  printf("Shape 0: %lu\n", shape.get(0));
  printf("Shape 1: %lu\n", shape.get(1));
  printf("Shape 2: %lu\n", shape.get(2));

  auto d = new float[shape.capacity()];
  d[0] = 1;
  d[1] = 2;
  printf("Data 0: %f\n", d[0]);
}

__global__ void kenrel() { testShape(); }

int main() {

  std::cout << "Executing kernel...\n";
  kenrel<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}
