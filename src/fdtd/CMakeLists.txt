add_executable(fdtd_1d_fixed 1d_fixed.cu)

option(FZ_FDTD_TEST_INDEX_OUT_OF_RANGE "Test index out of range" OFF)

if(FZ_FDTD_TEST_INDEX_OUT_OF_RANGE)
    target_compile_definitions(fdtd_1d_fixed PRIVATE FZ_FDTD_TEST_INDEX_OUT_OF_RANGE)
endif()

add_executable(fdtd_1d 1d.cu)

add_executable(fdtd_2d 2d.cu)
