cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA QUIET)
message(STATUS "FoundCUDA : ${CUDA_FOUND}")

if(BOOTSTRAP_DEPS)
else(BOOTSTRAP_DEPS)

  if(BUILD_MCAS_SERVER)
    add_subdirectory(cpp_basic)
    add_subdirectory(cpp_bench)
    add_subdirectory(python_basic)
    add_subdirectory(personalities)
    add_subdirectory(configurations)

    if(CUDA_FOUND)
      add_subdirectory(gpu_direct)
    endif(CUDA_FOUND)
  endif()

endif(BOOTSTRAP_DEPS)
