add_executable(array array.cu)
target_compile_options(array PRIVATE --expt-relaxed-constexpr)
add_executable(shape shape.cu)
target_compile_options(shape PRIVATE --expt-relaxed-constexpr)
