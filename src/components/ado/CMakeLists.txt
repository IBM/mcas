cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

if(BUILD_MCAS_SERVER)
  add_subdirectory(mgr_proxy)
  add_subdirectory(ado_proxy)
endif()
