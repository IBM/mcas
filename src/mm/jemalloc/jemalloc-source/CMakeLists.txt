cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

set(JEMALLOC_PROJECT jemalloc-5.2.1)
set(JEMALLOC_CONFIGURE_FLAGS "--disable-prof-gcc --enable-prof-libunwind --disable-initial-exec-tls --with-jemalloc-prefix=jel_")

include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

message("Building ${CMAKE_CURRENT_SOURCE_DIR}/${JEMALLOC_PROJECT}")
ExternalProject_Add(
  jemalloc-lib
  BUILD_IN_SOURCE 1
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/jemalloc
  URL https://github.com/jemalloc/jemalloc/archive/refs/tags/5.2.1.tar.gz
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${JEMALLOC_PROJECT}
  CONFIGURE_COMMAND ./autogen.sh
  BUILD_COMMAND ./configure --disable-prof-gcc --enable-prof-libunwind --disable-initial-exec-tls --with-jemalloc-prefix=jel_ --prefix ${CMAKE_CURRENT_BINARY_DIR}/build COMMAND make 
  INSTALL_COMMAND make install_include COMMAND make install_lib_static
  )

