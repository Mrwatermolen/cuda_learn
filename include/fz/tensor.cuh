#ifndef __FZ_TENSOR_CUH__
#define __FZ_TENSOR_CUH__

#include <fz/common.cuh>

#include "fz/array.cuh"

namespace fz::cuda {

template <typename T, SizeType N>
class Tensor {
 public:
  using DimArray = Array<SizeType, N>;

 public:
  FZ_CUDA_DUAL Tensor() {};

  FZ_CUDA_DUAL Tensor(DimArray shape)
      : _shape{shape},
        _size{makeSize(shape)},
        _stride{makeStride(shape)},
        _data{new T[size()]} {};

  FZ_CUDA_DUAL Tensor(const Tensor &) {};

  FZ_CUDA_DUAL Tensor(Tensor &&) {};

  FZ_CUDA_DUAL ~Tensor() {
    delete[] _data;
    _data = nullptr;
  }

  FZ_CUDA_DUAL auto operator()(const Tensor &) { return *this; }

  FZ_CUDA_DUAL auto operator()(Tensor &&) { return *this; }

  FZ_CUDA_DUAL constexpr static auto dim() { return N; }

  FZ_CUDA_DUAL auto size() const { return _size; }

  FZ_CUDA_DUAL auto shape() const -> DimArray { return _shape; }

  FZ_CUDA_DUAL auto stride() const -> DimArray { return _stride; }

  FZ_CUDA_DUAL auto operator[](SizeType index) -> T & { return _data[index]; }

  FZ_CUDA_DUAL auto operator[](SizeType index) const -> const T & {
    return _data[index];
  }

  template <typename... Args>
  FZ_CUDA_DUAL auto operator()(Args &&...args) {
    auto offset = dataOffset(_stride.data(), std::forward<Args>(args)...);
    return _data[offset];
  }

  template <typename... Args>
  FZ_CUDA_DUAL auto operator()(Args &&...args) const {
    auto offset = dataOffset(_stride.data(), std::forward<Args>(args)...);
    return _data[offset];
  }

  FZ_CUDA_DUAL auto begin() { return _data; }

  FZ_CUDA_DUAL auto end() { return _data + size(); }

  FZ_CUDA_DUAL auto cbegin() { return _data; }

  FZ_CUDA_DUAL auto cend() { return _data + size(); }

  // FZ_CUDA_DUAL auto resize(SizeType size) {}

  FZ_CUDA_DUAL auto reshape(DimArray shape) {}

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

}  // namespace fz::cuda

#endif  // __FZ_TENSOR_CUH__
