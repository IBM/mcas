cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)

if(BOOTSTRAP_DEPS)
else(BOOTSTRAP_DEPS)
add_subdirectory(libnop)
add_subdirectory(regression)

if(BUILD_MCAS_SERVER AND BUILD_MCAS_CLIENT)
  add_subdirectory(ado-test)
  add_subdirectory(kv-test)
endif()

add_subdirectory(eastl-test)

if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/mcas-testing")
  add_subdirectory(mcas-testing)
endif()

endif(BOOTSTRAP_DEPS)
