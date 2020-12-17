cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(libmcasapi-test C)



# include_directories(${CMAKE_SOURCE_DIR}/src/lib/libnupm/include)
# include_directories(${CMAKE_SOURCE_DIR}/src/lib/libpmem/include)
# include_directories(${CMAKE_SOURCE_DIR}/src/lib/libccpm/include)
# include_directories(../src)

# link_directories(${CMAKE_SOURCE_DIR}/src/lib/libnupm/)
# link_directories(${CMAKE_SOURCE_DIR}/src/lib/libpmem/)
# link_directories(${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}) # ndctl
link_directories(${CMAKE_INSTALL_PREFIX}/lib) # ndctl

add_compile_options(-g -fPIC)
add_executable(libmcasapi-test test1.c)
target_link_libraries(libmcasapi-test common mcasapi numa) # numa ndctl gtest pthread dl nupm pmem z gcov 
