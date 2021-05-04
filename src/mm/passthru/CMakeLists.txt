cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(mm-plugin-passthru C)

set(SOURCES src/mm_plugin_passthru.c)

add_compile_options("--std=c99")

include_directories(${CMAKE_SOURCE_DIR}/src/lib/common/include)
include_directories(${CMAKE_SOURCE_DIR}/src/mm)

add_library(${PROJECT_NAME} SHARED ${SOURCES})
  
target_compile_options(${PROJECT_NAME} PUBLIC "-D_GNU_SOURCE -fPIC")

# wrap malloc and free, and other things that may call malloc from GLIBC */
set(CMAKE_SHARED_LINKER_FLAGS "-Bsymbolic -Bsymbolic-functions \
-Wl,--wrap=malloc \
-Wl,--wrap=realloc \
-Wl,--wrap=free \
-Wl,--wrap=puts \
-Wl,--wrap=calloc \
-Wl,--wrap=malloc_usable_size \
-Wl,--wrap=memalign")

install (TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
