#ifndef __FZ_TENSOR_CUH__
#define __FZ_TENSOR_CUH__

#include <cassert>
#include <fz/array.cuh>
#include <fz/common.cuh>
#include <fz/memory.cuh>
#include <iostream>
#include <stdexcept>

namespace fz::cuda {

template <typename T, SizeType N>
class TensorHD;

template <typename T, SizeType N>
class Tensor {
 public:
  friend class TensorHD<T, N>;

 public:
  using DimArray = Array<SizeType, N>;

 public:
  FZ_CUDA_DUAL Tensor() {};

  FZ_CUDA_DUAL Tensor(DimArray shape)
      : _shape{shape},
        _size{makeSize(shape)},
        _stride{makeStride(shape)},
        _data{new T[size()]} {};

  FZ_CUDA_DUAL Tensor(const Tensor &other)
      : _shape{other._shape}, _size{other._size}, _stride{other._stride} {
    _data = new T[size()];
    for (SizeType i = 0; i < size(); ++i) {
      _data[i] = other._data[i];
    }
  };

  FZ_CUDA_DUAL Tensor(Tensor &&other) noexcept
      : _shape{std::move(other._shape)},
        _size{std::move(other._size)},
        _stride{std::move(other._stride)},
        _data{std::move(other._data)} {
    other._data = nullptr;
  };

  FZ_CUDA_DUAL ~Tensor() {
    delete[] _data;
    _data = nullptr;
  }

  FZ_CUDA_DUAL auto operator()(const Tensor &other) {
    if (this == &other) {
      return *this;
    }

    _shape = other._shape;
    _size = other._size;
    _stride = other._stride;
    _data = new T[size()];
    for (SizeType i = 0; i < size(); ++i) {
      _data[i] = other._data[i];
    }

    return *this;
  }

  FZ_CUDA_DUAL auto operator()(Tensor &&other) noexcept {
    if (this == &other) {
      return *this;
    }

    _shape = std::move(other._shape);
    _size = std::move(other._size);
    _stride = std::move(other._stride);
    _data = std::move(other._data);

    return *this;
  }

  FZ_CUDA_DUAL constexpr static auto dim() { return N; }

  FZ_CUDA_DUAL auto shape() const -> DimArray { return _shape; }

  FZ_CUDA_DUAL auto size() const { return _size; }

  FZ_CUDA_DUAL auto stride() const -> DimArray { return _stride; }

  FZ_CUDA_DUAL auto operator[](SizeType index) -> T & { return _data[index]; }

  FZ_CUDA_DUAL auto operator[](SizeType index) const -> const T & {
    return _data[index];
  }

  template <typename... Args>
  FZ_CUDA_DUAL auto operator()(Args &&...args) -> T & {
    auto offset = dataOffset(_stride.data(), std::forward<Args>(args)...);
    return _data[offset];
  }

  template <typename... Args>
  FZ_CUDA_DUAL auto operator()(Args &&...args) const -> const T & {
    auto offset = dataOffset(_stride.data(), std::forward<Args>(args)...);
    return _data[offset];
  }

  template <typename... Args>
  FZ_CUDA_DUAL auto at(Args &&...args) -> T & {
    auto offset = dataOffset(_stride.data(), std::forward<Args>(args)...);
    if (offset >= size()) {
#ifdef __CUDA_ARCH__
      printf("Tensor index out of range");
      asm("trap;");
#else
      throw std::out_of_range("Tensor index out of range");
#endif
    }

    return _data[offset];
  }

  template <typename... Args>
  FZ_CUDA_DUAL auto at(Args &&...args) const -> const T & {
    auto offset = dataOffset(_stride.data(), std::forward<Args>(args)...);
    if (offset >= size()) {
#ifdef __CUDA_ARCH__
      printf("Tensor index out of range");
      asm("trap;");
#else
      throw std::out_of_range("Tensor index out of range");
#endif
    }

    return _data[offset];
  }

  FZ_CUDA_DUAL auto begin() { return _data; }

  FZ_CUDA_DUAL auto end() { return _data + size(); }

  FZ_CUDA_DUAL auto cbegin() { return _data; }

  FZ_CUDA_DUAL auto cend() { return _data + size(); }

  // FZ_CUDA_DUAL auto resize(SizeType size) {}

  FZ_CUDA_DUAL auto reshape(DimArray shape) {
#ifdef __CUDA_ARCH__
    if (size() != makeSize(shape)) {
      printf("Cannot reshape tensor to different size");
      asm("trap;");
    }
#else
    if (size() != makeSize(shape)) {
      throw std::invalid_argument("Cannot reshape tensor to different size");
    }
#endif

    _shape = shape;
    _stride = makeStride(shape);
  }

  /**
   * @brief Only __kernelSetDeviceArrayData can call this function
   *
   * @param data
   */
  __device__ auto setData(T *data) { _data = data; }

 private:
  DimArray _shape{};
  SizeType _size{};
  DimArray _stride{};

  T *_data{};

  FZ_CUDA_DUAL auto makeSize(const DimArray &shape) const -> SizeType {
    if (shape.size() == 0) {
      return 0;
    }

    SizeType size = 1;
    for (const auto &s : shape) {
      size *= s;
    }

    return size;
  }

  FZ_CUDA_DUAL auto makeStride(const DimArray &shape) const {
    DimArray stirde{};
    stirde[stirde.size() - 1] = 1;
    for (SizeType i = 1; i < shape.size(); ++i) {
      stirde[stirde.size() - i - 1] =
          stirde[stirde.size() - i] * shape[stirde.size() - i];
    }

    return stirde;
  }

  template <SizeType dim>
  FZ_CUDA_DUAL static auto rawOffset(const SizeType strides[]) -> SizeType {
    return 0;
  }

  template <SizeType dim, typename Arg, typename... Args>
  FZ_CUDA_DUAL static auto rawOffset(const SizeType strides[], Arg &&arg,
                                     Args &&...args) -> SizeType {
    return static_cast<SizeType>(arg) * strides[dim] +
           rawOffset<dim + 1>(strides, std::forward<Args>(args)...);
  }

  template <typename Arg, typename... Args>
  FZ_CUDA_DUAL static auto dataOffset(const SizeType stride[], Arg &&arg,
                                      Args &&...args) -> SizeType {
    constexpr SizeType nargs = sizeof...(Args) + 1;
    if constexpr (nargs == dim()) {
      return rawOffset<static_cast<SizeType>(0)>(stride, std::forward<Arg>(arg),
                                                 std::forward<Args>(args)...);
    }
  }
};

template <typename T, SizeType N>
class TensorHD {
 public:
  using DeviceTensor = Tensor<T, N>;
  using HostTensor = Tensor<T, N>;

  TensorHD(Array<SizeType, N> shape) : _host{new HostTensor{shape}} {}

  ~TensorHD() {
    if (_device) {
      // don't call destructor to avoid double free
      // __destroyDeviceObject(_device);
      {
        auto err = cudaFree(_device);
        if (err != cudaSuccess) {
          std::cerr << "Failed to free device memory \n";
          std::abort();
        }
      }
      _device = nullptr;
    }

    if (_device_data) {
      {
        auto err = cudaFree(_device_data);
        if (err != cudaSuccess) {
          std::cerr << "Failed to free device memory \n";
          std::abort();
        }
      }
      _device_data = nullptr;
    }

    delete _host;
    _host = nullptr;
  }

  FZ_CUDA_DUAL auto device() -> DeviceTensor * { return _device; }

  auto host() -> HostTensor * { return _host; }

  FZ_CUDA_DUAL auto device() const -> const DeviceTensor * { return _device; }

  auto host() const -> const HostTensor * { return _host; }

  auto copyHostToDevice() -> void {
    if (!_host) {
      throw std::runtime_error("Host memory is not allocated");
    }
    if (_device) {
      // Free previous device memory
      auto err = cudaFree(_device);
      if (err != cudaSuccess) {
        throw std::runtime_error("Failed to free device memory" +
                                 std::string(cudaGetErrorString(err)));
      }
      _device = nullptr;
    }

    if (_device_data != nullptr) {
      auto err = cudaFree(_device_data);
      if (err != cudaSuccess) {
        throw std::runtime_error("Failed to free device memory" +
                                 std::string(cudaGetErrorString(err)));
      }
    }

    {
      // Allocate device memory
      auto err = cudaMalloc(&_device, sizeof(DeviceTensor));
      if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory" +
                                 std::string(cudaGetErrorString(err)));
      }
    }

    {
      // Copy tensor metadata
      auto err = cudaMemcpy(_device, _host, sizeof(HostTensor),
                            cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy to device memory" +
                                 std::string(cudaGetErrorString(err)));
      }
    }

    {
      // Malloc device data
      auto err = cudaMalloc(&_device_data, _host->size() * sizeof(T));
      if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory" +
                                 std::string(cudaGetErrorString(err)));
      }
    }

    {
      // Copy tensor data
      auto err = cudaMemcpy(_device_data, _host->_data,
                            _host->size() * sizeof(T), cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy to device memory" +
                                 std::string(cudaGetErrorString(err)));
      }
    }

    __kernelSetDeviceArrayData<<<1, 1>>>(_device, _device_data);
  }

  auto copyDeviceToHost() -> void {
    if (!_device || !_device_data) {
      throw std::runtime_error("Device memory is not allocated");
    }

    {
      auto err = cudaMemcpy(_host, _device, sizeof(DeviceTensor),
                            cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy to host memory" +
                                 std::string(cudaGetErrorString(err)));
      }
    }

    assert(_device_data == _host->_data);
    _host->_data = new T[_host->size()];

    {
      auto err = cudaMemcpy(_host->_data, _device_data,
                            _host->size() * sizeof(T), cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy data to host memory" +
                                 std::string(cudaGetErrorString(err)));
      }
    }

    // auto new_host = new HostTensor();

    // {
    //   auto err = cudaMemcpy(new_host, _device, sizeof(DeviceTensor),
    //                         cudaMemcpyDeviceToHost);
    //   if (err != cudaSuccess) {
    //     throw std::runtime_error("Failed to copy to host memory" +
    //                              std::string(cudaGetErrorString(err)));
    //   }
    // }

    // assert(_device_data == new_host->_data);
    // new_host->_data = new T[new_host->size()];

    // {
    //   auto err =
    //       cudaMemcpy(new_host->_data, _device_data,
    //                  new_host->size() * sizeof(T), cudaMemcpyDeviceToHost);
    //   if (err != cudaSuccess) {
    //     throw std::runtime_error("Failed to copy data to host memory" +
    //                              std::string(cudaGetErrorString(err)));
    //   }
    // }

    // delete _host;
    // _host = new_host;
  }

 protected:
 private:
  DeviceTensor *_device{};
  HostTensor *_host{};
  T *_device_data{};
};

}  // namespace fz::cuda

#endif  // __FZ_TENSOR_CUH__
