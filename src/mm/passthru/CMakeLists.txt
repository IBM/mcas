cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(mm-plugin-passthru C)

set(SOURCES src/mm_plugin_passthru.c)

add_compile_options("-fvisibility=hidden")

include_directories(${CMAKE_SOURCE_DIR}/src/lib/common/include)
include_directories(${CMAKE_SOURCE_DIR}/src/mm)

add_library(${PROJECT_NAME} SHARED ${SOURCES})
  
target_compile_options(${PROJECT_NAME} PUBLIC "-D_GNU_SOURCE -fPIC")

install (TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
