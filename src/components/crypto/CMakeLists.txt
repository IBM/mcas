cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

if(BUILD_MCAS_SERVER AND BUILD_COMPONENT_CRYPTO)
  add_subdirectory(tls)
endif()
