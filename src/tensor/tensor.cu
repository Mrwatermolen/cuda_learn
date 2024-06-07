#include <cstdio>
#include <fz/tensor.cuh>

__global__ void kenerl() {
  auto t = fz::cuda::Tensor<float, 3>({2, 3, 4});
  printf("Tensor Shape: ");
  for (const auto& s : t.shape()) {
    printf("%lu ", s);
  }
  printf("\n");
  printf("Tensor Size: %lu", t.size());
  printf("\n");
  printf("Tensor Stride: ");
  for (const auto& s : t.stride()) {
    printf("%lu ", s);
  }
  printf("\n");
  printf("Tensor Data: \n");
  for (const auto& v : t) {
    printf("%.2f ", v);
  }
  printf("\n");
  printf("Print as tensor: \n");
  for (std::size_t i = 0; i < t.shape()[0]; ++i) {
    printf("\n");
    for (std::size_t j = 0; j < t.shape()[1]; ++j) {
      for (std::size_t k = 0; k < t.shape()[2]; ++k) {
        printf("%.2f ", t(i, j, k));
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n");

  auto delta = 0.2;
  for (std::size_t i = 0; i < t.size(); ++i) {
    t[i] = i * delta;
  }

  printf("Tensor Data: \n");
  for (const auto& v : t) {
    printf("%.2f ", v);
  }
  printf("\n");
  printf("Print as tensor: \n");
  for (std::size_t i = 0; i < t.shape()[0]; ++i) {
    printf("\n");
    for (std::size_t j = 0; j < t.shape()[1]; ++j) {
      for (std::size_t k = 0; k < t.shape()[2]; ++k) {
        printf("%.2f ", t(i, j, k));
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n");
}

int main() {
  auto grid_dim = dim3{1, 1, 1};
  auto block_dim = dim3{1, 1, 1};
  kenerl<<<grid_dim, block_dim>>>();
  cudaDeviceSynchronize();
}
