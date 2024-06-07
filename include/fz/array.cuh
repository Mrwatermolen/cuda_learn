#ifndef __FZ_ARRAY_CUH__
#define __FZ_ARRAY_CUH__

#include <fz/shape.cuh>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace fz::cuda {

/**
 * @brief Same as std::array
 *
 * @tparam T
 * @tparam S
 */
template <typename T, SizeType S>
class Array {
 public:
  FZ_CUDA_DUAL Array() {}

  FZ_CUDA_DUAL Array(std::initializer_list<T> list) {
    SizeType i = 0;
    for (auto &item : list) {
      _data[i++] = item;
    }
  }

  FZ_CUDA_DUAL Array(const Array &other) {
    for (SizeType i = 0; i < size(); ++i) {
      _data[i] = other._data[i];
    }
  }

  FZ_CUDA_DUAL Array(Array &&other) noexcept {
    for (SizeType i = 0; i < size(); ++i) {
      _data[i] = std::move(other._data[i]);
    }
  }

  FZ_CUDA_DUAL auto operator=(Array &&other) noexcept -> Array & {
    if (this != &other) {
      for (SizeType i = 0; i < size(); ++i) {
        _data[i] = std::move(other._data[i]);
      }
    }
    return *this;
  }

  FZ_CUDA_DUAL auto operator=(const Array &other) -> Array & {
    if (this != &other) {
      for (SizeType i = 0; i < size(); ++i) {
        _data[i] = other._data[i];
      }
    }
    return *this;
  }

  FZ_CUDA_DUAL auto operator[](SizeType index) -> T & { return _data[index]; }

  FZ_CUDA_DUAL auto operator[](SizeType index) const -> const T & {
    return _data[index];
  }

  FZ_CUDA_DUAL static constexpr auto size() -> SizeType { return S; }

  FZ_CUDA_DUAL auto begin() -> T * { return _data; }

  FZ_CUDA_DUAL auto end() -> T * { return _data + size(); }

  FZ_CUDA_DUAL auto cbegin() const -> const T * { return _data; }

  FZ_CUDA_DUAL auto cend() const -> const T * { return _data + size(); }

 private:
  T _data[S] = {};
};

template <typename T>
__global__ inline auto __kernelDestroyDeviceObject(T *device) -> void {
  printf("Destroying device object\n");
  (*device).~T();
}

template <typename T>
inline auto __destroyDeviceObject(T *device) -> void {
  constexpr auto has_trivial_destructor = std::is_trivially_destructible_v<T>;
  if constexpr (!has_trivial_destructor) {
    __kernelDestroyDeviceObject<<<1, 1>>>(device);
  }
}

template <typename T, SizeType S>
class ArrayHD {
 public:
  using DeviceArray = Array<T, S>;
  using HostArray = Array<T, S>;

  ArrayHD() {}

  ~ArrayHD() {
    if (_device) {
      __destroyDeviceObject(_device);
      cudaFree(_device);
      _device = nullptr;
    }

    delete _host;
    _host = nullptr;
  }

  FZ_CUDA_DUAL auto device() -> DeviceArray * { return _device; }

  auto host() -> HostArray * { return _host; }

  FZ_CUDA_DUAL auto device() const -> const DeviceArray * { return _device; }

  auto host() const -> const HostArray * { return _host; }

  auto allocateDevice() -> void {
    if (_device) {
      throw std::runtime_error("Device memory is already allocated");
    }

    auto err = cudaMalloc(&_device, sizeof(DeviceArray));
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to allocate device memory" +
                               std::string(cudaGetErrorString(err)));
    }
  }

  auto allocateHost() -> void {
    if (_host) {
      throw std::runtime_error("Host memory is already allocated");
    }

    _host = new HostArray();
  }

  auto copyHostToDevice() -> void {
    if (!_host) {
      throw std::runtime_error("Host memory is not allocated");
    }
    if (!_device) {
      allocateDevice();
    }

    auto err =
        cudaMemcpy(_device, _host, sizeof(HostArray), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to copy to device memory" +
                               std::string(cudaGetErrorString(err)));
    }
  }

  auto copyDeviceToHost() -> void {
    if (!_device) {
      throw std::runtime_error("Device memory is not allocated");
    }
    if (!_host) {
      allocateHost();
    }

    auto err =
        cudaMemcpy(_host, _device, sizeof(DeviceArray), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to copy to host memory" +
                               std::string(cudaGetErrorString(err)));
    }
  }

 protected:
 private:
  DeviceArray *_device{};
  HostArray *_host{};
};

}  // namespace fz::cuda

#endif  // __FZ_ARRAY_CUH__
