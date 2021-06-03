cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

# deps.. czmq-devel

if(BUILD_MCAS_SERVER)
  add_subdirectory (zyre)
endif()
