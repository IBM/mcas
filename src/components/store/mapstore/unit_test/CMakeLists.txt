cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(mapstore-tests CXX)

include_directories(${CMAKE_SOURCE_DIR}/src/components)
include_directories(${CMAKE_INSTALL_PREFIX}/include)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)
link_directories(${CMAKE_INSTALL_PREFIX}/lib64)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  set(GTEST_LIB gtest)
else()
  set(GTEST_LIB gtestd)
endif()

add_compile_options(-O2 -fPIC)
add_executable(mapstore-test1 test1.cpp)
target_link_libraries(mapstore-test1 ${ASAN_LIB} common numa ${GTEST_LIB} pthread dl)

