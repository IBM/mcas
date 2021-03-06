cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(rust-mcasapi)

include(ExternalProject)

configure_file(build.rs.in ${CMAKE_CURRENT_SOURCE_DIR}/build.rs)

find_program(CARGO cargo)
find_program(BINDGEN bindgen)

# wrapper generation
add_custom_target(mcasapi_wrapper.rs
  COMMAND ${BINDGEN} --no-layout-tests ${CMAKE_SOURCE_DIR}/src/lib/libmcasapi/src/mcas_api_wrapper.h -o ${CMAKE_CURRENT_SOURCE_DIR}/src/mcasapi_wrapper.rs
  DEPENDS ${CMAKE_SOURCE_DIR}/src/lib/libmcasapi/src/mcas_api_wrapper.h
  BYPRODUCTS ${CMAKE_CURRENT_SOURCE_DIR}/src/mcasapi_wrapper.rs
  )

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/debug/libmcasapi.rlib
  COMMAND ${CARGO} build --lib --target-dir=${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS mcasapi_wrapper.rs
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

add_custom_target(${PROJECT_NAME} ALL SOURCES ${CMAKE_CURRENT_BINARY_DIR}/debug/libmcasapi.rlib)

