cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(libmcasapi-test C)

link_directories(${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

add_compile_options(-g -fPIC)
add_executable(libmcasapi-test test1.c)
target_link_libraries(libmcasapi-test common numa pthread dl mcasapi)
