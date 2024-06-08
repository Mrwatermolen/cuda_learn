#ifndef __FZ_MEMORY_CUH__
#define __FZ_MEMORY_CUH__

#include <type_traits>

template <typename T>
__global__ auto __kernelDestroyDeviceObject(T *device) -> void {
  (*device).~T();
}

template <typename T>
auto __destroyDeviceObject(T *device) -> void {
  constexpr auto has_trivial_destructor = std::is_trivially_destructible_v<T>;
  if constexpr (!has_trivial_destructor) {
    __kernelDestroyDeviceObject<<<1, 1>>>(device);
  }
}

// template <typename T>
// __global__ void __kernelConstructDeviceObject(T *device) {
//   constexpr auto has_trivial_constructor =
//       std::is_trivially_default_constructible_v<T>;
//   if constexpr (!has_trivial_constructor) {
//     (*device) = T();
//   }
// }

template <typename D, typename T>
__global__ auto __kernelSetDeviceArrayData(D *device, T *data) -> void {
  device->setData(data);
}

#endif  // __FZ_MEMORY_CUH__
