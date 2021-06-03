cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

if (BUILD_PYTHON_SUPPORT)

  if (BUILD_MCAS_SERVER)
    add_subdirectory(mcas-python-api)
    add_subdirectory(python-personality)
  endif()
  
  add_subdirectory(pymm)

endif(BUILD_PYTHON_SUPPORT)


