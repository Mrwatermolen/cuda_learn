#ifndef __CUDA_LEARN_FDTD_UPDATE_SCHEME_H__
#define __CUDA_LEARN_FDTD_UPDATE_SCHEME_H__

#include "fz/common.cuh"
#define FDTD_UPDATE_SCHEME_CUDA_DUAL __host__ __device__

namespace one_dimensional {

template <typename T, typename E, typename H, typename I>
FDTD_UPDATE_SCHEME_CUDA_DUAL inline auto updateE(const T& ce2e, const T& ch2e,
                                                 E&& e, const H& h,
                                                 const I& start, const I& end) {
#if defined(FZ_FDTD_TEST_INDEX_OUT_OF_RANGE)
  for (auto i = start + 1; i < end; ++i) {
    e.at(i) = ce2e.at(i) * e.at(i) + ch2e.at(i) * (h.at(i) - h.at(i - 1));
  }
#else

  for (auto i = start + 1; i < end; ++i) {
    e[i] = ce2e[i] * e[i] + ch2e[i] * (h[i] - h[i - 1]);
  }
#endif
}

template <typename T, typename E, typename H, typename I>
FDTD_UPDATE_SCHEME_CUDA_DUAL inline auto updateELeft(const T& ce2e,
                                                     const T& ch2e, E&& e,
                                                     const H& h,
                                                     const I& start) {
  if (start == 0) {
    return;
  }
#if defined(FZ_FDTD_TEST_INDEX_OUT_OF_RANGE)
  e.at(start) = ce2e.at(start) * e.at(start) +
                ch2e.at(start) * (h.at(start) - h.at(start - 1));
#else

  e[start] = ce2e[start] * e[start] + ch2e[start] * (h[start] - h[start - 1]);

#endif
}

template <typename T, typename H, typename E, typename I>
FDTD_UPDATE_SCHEME_CUDA_DUAL inline auto updateH(const T& ch2h, const T& ce2h,
                                                 H&& h, const E& e,
                                                 const I& start, const I& end) {
#if defined(FZ_FDTD_TEST_INDEX_OUT_OF_RANGE)
  for (auto i = start; i < end; ++i) {
    h.at(i) = ch2h.at(i) * h.at(i) + ce2h.at(i) * (e.at(i + 1) - e.at(i));
  }
#else

  for (auto i = start; i < end; ++i) {
    h[i] = ch2h[i] * h[i] + ce2h[i] * (e[i + 1] - e[i]);
  }
#endif
}

}  // namespace one_dimensional

namespace utl {

enum class Axis { X, Y, Z };

enum class Direction { Forward, Backward };

enum class Field { E, H };

FZ_CUDA_DUAL inline constexpr auto axisA(Axis c) {
  // a cross b = c
  switch (c) {
    case Axis::X:
      return Axis::Y;
    case Axis::Y:
      return Axis::Z;
    case Axis::Z:
      return Axis::X;
  }
}

FZ_CUDA_DUAL inline constexpr auto axisB(Axis c) {
  // a cross b = c
  switch (c) {
    case Axis::X:
      return Axis::Z;
    case Axis::Y:
      return Axis::X;
    case Axis::Z:
      return Axis::Y;
  }
}

FZ_CUDA_DUAL inline constexpr auto oppositeDirection(Direction d) {
  return d == Direction::Forward ? Direction::Backward : Direction::Forward;
}

FZ_CUDA_DUAL inline constexpr auto dualField(Field f) {
  return f == Field::E ? Field::H : Field::E;
}

}  // namespace utl

namespace two_dimensional {}

#endif  // __CUDA_LEARN_FDTD_UPDATE_SCHEME_H__
