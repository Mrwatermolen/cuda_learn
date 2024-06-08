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

  auto t2 = t;
  printf("Tensor Data: \n");
  for (const auto& v : t2) {
    printf("%.2f ", v);
  }
  printf("\n");
  printf("Print as tensor: \n");
  for (std::size_t i = 0; i < t2.shape()[0]; ++i) {
    printf("\n");
    for (std::size_t j = 0; j < t2.shape()[1]; ++j) {
      for (std::size_t k = 0; k < t2.shape()[2]; ++k) {
        printf("%.2f ", t2(i, j, k));
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n");
  printf("Reshape Tensor: \n");
  t2.reshape({4, 3, 2});
  printf("Tensor Shape: ");
  for (const auto& s : t2.shape()) {
    printf("%lu ", s);
  }
  printf("\n");
  printf("Tensor Size: %lu", t2.size());
  printf("\n");
  printf("Tensor Stride: ");
  for (const auto& s : t2.stride()) {
    printf("%lu ", s);
  }
  printf("\n");
  printf("Tensor Data: \n");
  for (const auto& v : t2) {
    printf("%.2f ", v);
  }
  printf("\n");
  printf("Print as tensor: \n");
  for (std::size_t i = 0; i < t2.shape()[0]; ++i) {
    printf("\n");
    for (std::size_t j = 0; j < t2.shape()[1]; ++j) {
      for (std::size_t k = 0; k < t2.shape()[2]; ++k) {
        printf("%.2f ", t2(i, j, k));
      }
      printf("\n");
    }
    printf("\n");
  }

  printf("T1: \n");
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
}

template <typename T, fz::cuda::SizeType N>
__global__ void kernelTestTensorHD(fz::cuda::Tensor<T, N>* t) {
  auto thread_id = threadIdx.x * blockDim.x + threadIdx.y;

  if (thread_id == 0) {
    printf("Running in the device\n");
    printf("==========Tensor Basic Info==========\n");
    printf("Tensor size: %lu\n", t->size());
    printf("Tensor shape: ");
    for (const auto& s : t->shape()) {
      printf("%lu ", s);
    }
    printf("\n");
    printf("Tensor stride: ");
    for (const auto& s : t->stride()) {
      printf("%lu ", s);
    }
    printf("\n");
    printf("==========End of Tensor Basic Info==========\n");

    printf("Data from the host\n");

    printf("==========Tensor Data==========\n");
    printf("Data address: %p\n", t->begin());
    for (const auto& item : *t) {
      printf("%f ", item);
    }
    printf("\n");
    printf("Data as tensor: \n");
    for (std::size_t i = 0; i < t->shape()[0]; ++i) {
      printf("\n");
      for (std::size_t j = 0; j < t->shape()[1]; ++j) {
        for (std::size_t k = 0; k < t->shape()[2]; ++k) {
          printf("%.2f ", t->at(i, j, k));
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("==========End of Tensor Data==========\n");
  }

  __syncthreads();

  auto size = t->size() / (blockDim.x * blockDim.y);
  for (std::size_t i = thread_id * size; i < (thread_id + 1) * size; ++i) {
    (*t)[i] = (*t)[i] - thread_id;
  }

  __syncthreads();

  if (thread_id == 0) {
    printf("Data changed in the device\n");
    printf("==========Tensor Data==========\n");
    for (const auto& item : *t) {
      printf("%f ", item);
    }
    printf("\n");
    printf("Data as tensor: \n");
    for (std::size_t i = 0; i < t->shape()[0]; ++i) {
      printf("\n");
      for (std::size_t j = 0; j < t->shape()[1]; ++j) {
        for (std::size_t k = 0; k < t->shape()[2]; ++k) {
          printf("%.2f ", t->at(i, j, k));
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("==========End of Tensor Data==========\n");

    printf("Reshape Tensor: \n");
    t->reshape({4, 3, 2});
    printf("Tensor Shape: ");
    for (const auto& s : t->shape()) {
      printf("%lu ", s);
    }
    printf("\n");
    printf("Tensor Size: %lu", t->size());
    printf("\n");
    printf("Tensor Stride: ");
    for (const auto& s : t->stride()) {
      printf("%lu ", s);
    }
    printf("\n");
    printf("Tensor Data: \n");
    for (const auto& v : *t) {
      printf("%.2f ", v);
    }
    printf("\n");
    printf("Print as tensor: \n");
    for (std::size_t i = 0; i < t->shape()[0]; ++i) {
      printf("\n");
      for (std::size_t j = 0; j < t->shape()[1]; ++j) {
        for (std::size_t k = 0; k < t->shape()[2]; ++k) {
          printf("%.2f ", t->at(i, j, k));
        }
        printf("\n");
      }
      printf("\n");
    }

    printf("End of the device\n");
  }
}

void testTensorHD() {
  printf(
      "=======================TEST TENSOR HOST "
      "DEVICE=======================\n");

  printf("Running in the host\n");

  auto hd = fz::cuda::TensorHD<float, 3>({2, 3, 4});
  auto host = hd.host();

  // init Host Array
  for (std::size_t i = 0; i < host->size(); ++i) {
    (*host)[i] = host->size() - i;
  }

  printf("==========Tensor Data==========\n");
  for (const auto& item : *host) {
    printf("%f ", item);
  }
  printf("\n");
  printf("Data as tensor: \n");
  for (std::size_t i = 0; i < host->shape()[0]; ++i) {
    printf("\n");
    for (std::size_t j = 0; j < host->shape()[1]; ++j) {
      for (std::size_t k = 0; k < host->shape()[2]; ++k) {
        printf("%.2f ", host->at(i, j, k));
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("==========End of Tensor Data==========\n");

  printf("\n");
  auto grid = dim3(1, 1, 1);
  auto block = dim3(2, 2, 1);
  hd.copyHostToDevice();
  kernelTestTensorHD<float, 3><<<grid, block>>>(hd.device());
  cudaDeviceSynchronize();
  printf("\n");
  printf("Back to Host\n");
  printf("==========Tensor Data==========\n");
  for (const auto& item : *host) {
    printf("%f ", item);
  }
  printf("\n");
  printf("Data as tensor: \n");
  for (std::size_t i = 0; i < host->shape()[0]; ++i) {
    printf("\n");
    for (std::size_t j = 0; j < host->shape()[1]; ++j) {
      for (std::size_t k = 0; k < host->shape()[2]; ++k) {
        printf("%.2f ", host->at(i, j, k));
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("==========End of Tensor Data==========\n");

  hd.copyDeviceToHost();

  host = hd.host();
  printf("Copy Device to Host\n");
  printf("==========Tensor Data==========\n");
  for (const auto& item : *host) {
    printf("%f ", item);
  }
  printf("\n");
  printf("Data as tensor: \n");
  for (std::size_t i = 0; i < host->shape()[0]; ++i) {
    printf("\n");
    for (std::size_t j = 0; j < host->shape()[1]; ++j) {
      for (std::size_t k = 0; k < host->shape()[2]; ++k) {
        printf("%.2f ", host->at(i, j, k));
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("==========End of Tensor Data==========\n");

  printf(
      "=======================END TEST TENSOR HOST "
      "DEVICE=======================\n");
}

int main() {
  auto grid_dim = dim3{1, 1, 1};
  auto block_dim = dim3{1, 1, 1};
  kenerl<<<grid_dim, block_dim>>>();
  cudaDeviceSynchronize();
  testTensorHD();
}
