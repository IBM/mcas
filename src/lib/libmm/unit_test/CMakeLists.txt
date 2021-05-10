cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(libmm-unit-test CXX)

include_directories(${CMAKE_SOURCE_DIR}/src/lib/common/include)
include_directories(${CMAKE_SOURCE_DIR}/src/mm)
include_directories(${CMAKE_INSTALL_PREFIX}/include)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)
link_directories(${CMAKE_INSTALL_PREFIX}/lib64)

add_compile_options(-g -fPIC -DCONFIG_DEBUG)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  set(GTEST_LIB gtest)
else()
  set(GTEST_LIB gtestd)
endif()


add_executable(${PROJECT_NAME} test1.cpp)

target_link_libraries(${PROJECT_NAME} ${GTEST_LIB} common pthread dl ${GTEST})
