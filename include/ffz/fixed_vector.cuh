#ifndef __FFZ_FIXED_VECTOR_CUH__
#define __FFZ_FIXED_VECTOR_CUH__

#include "ffz/shape.cuh"

namespace ffz::cuda {

/**
 * @brief FixedVector is a fixed size vector with fixed shape and stride
 *
 * @tparam T
 * @tparam S
 */
template <typename T, typename S> class FixedVector {
public:
  FZ_CUDA_DUAL FixedVector() {
    for (SizeType i = 0; i < dim(); ++i) {
      makeShape();
      makeStride();
    }
  };

  FZ_CUDA_DUAL FixedVector(const FixedVector &other) : _shape(other._shape) {
    for (SizeType i = 0; i < size(); ++i) {
      _data[i] = other._data[i];
    }
  }

  FZ_CUDA_DUAL auto operator=(const FixedVector &other) -> FixedVector & {
    if (this != &other) {
      for (SizeType i = 0; i < size(); ++i) {
        _data[i] = other._data[i];
      }
      _shape = other._shape;
    }
    return *this;
  }

  FZ_CUDA_DUAL auto operator[](SizeType index) -> T & { return _data[index]; }

  FZ_CUDA_DUAL auto operator[](SizeType index) const -> const T & {
    return _data[index];
  }

  template <typename... Args>
  FZ_CUDA_DUAL auto operator()(Args &&...args) -> T & {
    auto offset = dataOffset(_stride, std::forward<Args>(args)...);
    return _data[dataOffset(_stride, std::forward<Args>(args)...)];
  }

  FZ_CUDA_DUAL static constexpr auto dim() -> SizeType { return S::size(); }

  FZ_CUDA_DUAL static constexpr auto size() -> SizeType {
    return S::capacity();
  }

  FZ_CUDA_DUAL auto shape() const { return _shape; }

  FZ_CUDA_DUAL auto stride() const { return _stride; }

  FZ_CUDA_DUAL auto shape(SizeType i) const -> SizeType { return _shape[i]; }

  FZ_CUDA_DUAL auto stride(SizeType i) const -> SizeType { return _stride[i]; }

  FZ_CUDA_DUAL void fill(T value) {
    for (SizeType i = 0; i < size(); ++i) {
      _data[i] = value;
    }
  }

  FZ_CUDA_DUAL auto begin() -> T * { return _data; }

  FZ_CUDA_DUAL auto end() -> T * { return _data + size(); }

  FZ_CUDA_DUAL auto cbegin() const -> const T * { return _data; }

  FZ_CUDA_DUAL auto cend() const -> const T * { return _data + size(); }

private:
  T _data[size()] = {};
  SizeType _shape[dim()];
  SizeType _stride[dim()];

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

  FZ_CUDA_DUAL auto makeStride() {
    if constexpr (dim() == 0) {
      return;
    } else {
      makeStride<0>();
    }
  }

  template <SizeType I> FZ_CUDA_DUAL auto makeStride() -> void {
    if constexpr (I == dim() - 1) {
      _stride[I] = S::stride(I);
    } else {
      _stride[I] = S::stride(I);
      makeStride<I + 1>();
    }
  }

  FZ_CUDA_DUAL auto makeShape() -> void {
    if constexpr (dim() == 0) {
      return;
    } else {
      makeShape<dim() - 1>();
    }
  }

  template <SizeType I> FZ_CUDA_DUAL auto makeShape() -> void {
    _shape[I] = S::get(I);
    if constexpr (I == 0) {
      return;
    } else {
      makeShape<I - 1>();
    }
  }
};

} // namespace ffz::cuda

#endif // __FFZ_FIXED_VECTOR_CUH__
