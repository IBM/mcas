
cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

CHECK_INCLUDE_FILES(infiniband/verbs.h HAVE_INFINIBAND_HEADERS)

if(BUILD_MCAS_SERVER AND HAVE_INFINIBAND_HEADERS)
  add_subdirectory (fabric)
endif()

