#ifndef __FFZ_SHAPE_CUH__
#define __FFZ_SHAPE_CUH__

namespace ffz::cuda {

using SizeType = std::size_t;

#define FZ_CUDA_DUAL __host__ __device__

/**
 * @brief A fixed shape class with compile-time size.
 *
 * @tparam X
 */
template <SizeType... X> class FixedShape {
public:
  FZ_CUDA_DUAL FixedShape() {}

  FZ_CUDA_DUAL static constexpr auto size() -> SizeType { return sizeof...(X); }

  FZ_CUDA_DUAL static constexpr auto capacity() -> SizeType {
    return (X * ...);
  }

  template <SizeType I> FZ_CUDA_DUAL static constexpr auto get() -> SizeType {
    return get(I);
  }

  template <SizeType I> FZ_CUDA_DUAL static constexpr auto stride() -> SizeType {
    return stride(I);
  }

  FZ_CUDA_DUAL static constexpr auto get(SizeType I) -> SizeType {
    return _data[I];
  }

  FZ_CUDA_DUAL static constexpr auto stride(SizeType I) -> SizeType {
    if (I == size() - 1) {
      return 1;
    } else {
      return _data[I + 1] * stride(I + 1);
    }
  }

private:
  constexpr static SizeType _data[size()] = {X...};
};

} // namespace ffz::cuda

#endif // __FFZ_SHAPE_CUH__
