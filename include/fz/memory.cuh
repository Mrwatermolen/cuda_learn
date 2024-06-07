#ifndef __FZ_MEMORY_CUH__
#define __FZ_MEMORY_CUH__

#include <type_traits>

template <typename T>
__global__ inline auto __kernelDestroyDeviceObject(T *device) -> void {
  (*device).~T();
}

template <typename T>
inline auto __destroyDeviceObject(T *device) -> void {
  constexpr auto has_trivial_destructor = std::is_trivially_destructible_v<T>;
  if constexpr (!has_trivial_destructor) {
    __kernelDestroyDeviceObject<<<1, 1>>>(device);
  }
}

#endif  // __FZ_MEMORY_CUH__
