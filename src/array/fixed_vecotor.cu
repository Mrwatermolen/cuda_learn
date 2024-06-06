#include <cstdio>
#include <ffz/fixed_vector.cuh>
#include <iostream>

__device__ void testArray() {
  ffz::cuda::FixedVector<float, ffz::cuda::FixedShape<2, 3, 4>> a;

  // array info
  printf("==========Array Basic Info==========\n");
  printf("Array dim: %lu\n", a.dim());
  printf("Array size: %lu\n", a.size());
  printf("Array shape: ");
  for (std::size_t i = 0; i < a.dim(); ++i) {
    printf("%lu ", a.shape(i));
  }
  printf("\n");
  printf("Array stride: ");
  for (std::size_t i = 0; i < a.dim(); ++i) {
    printf("%lu ", a.stride(i));
  }
  printf("\n");
  printf("==========End of Array Basic Info==========\n");

  // array data
  printf("==========Array Data==========\n");
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = i;
  }
  printf("Flatten print\n");
  for (const auto &item : a) {
    printf("%f ", item);
  }
  printf("\n");
  printf("3D print\n");
  for (std::size_t i = 0; i < a.shape(0); ++i) {
    for (std::size_t j = 0; j < a.shape(1); ++j) {
      for (std::size_t k = 0; k < a.shape(2); ++k) {
        printf("%f ", a(i, j, k));
      }
      printf("\n");
    }
    printf("\n");
  }

  printf("\n");
  printf("==========End of Array Data==========\n");
}

__global__ void kernel() { testArray(); }

int main() {
  std::cout << "Run in the device\n";

  kernel<<<1, 1>>>();

  cudaDeviceSynchronize();

  printf("\n");
}
