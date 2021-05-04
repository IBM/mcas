cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(pmem C)
enable_language(C)

execute_process( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )
message(FATAL ${ARCHITECTURE})

find_library(LIB_NDCTL libndctl.so HINTS ${CMAKE_INSTALL_PREFIX}/lib)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories("./")
include_directories("./x86_64")
include_directories("./include")
include_directories("./common")
include_directories("../common/include")
include_directories(${CMAKE_INSTALL_PREFIX}/include) # ndctl

link_directories(${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}) # tbb tbbmalloc
link_directories(${CMAKE_INSTALL_PREFIX}/lib) # tbb tbbmalloc

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

add_definitions(-DCONFIG_DEBUG)

#set(SOURCES libpmem.c memops_generic.c  pmem.c  pmem.h  pmem_posix.c)
file(GLOB_RECURSE SOURCES ./*.c)

add_library(${PROJECT_NAME} SHARED ${SOURCES})
add_library(${PROJECT_NAME}-static STATIC ${SOURCES})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx512f")

target_compile_options(${PROJECT_NAME} PUBLIC "-fPIC")
target_compile_options(${PROJECT_NAME}-static PUBLIC "-fPIC")

#target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Debug>:-O0>")
target_link_libraries(${PROJECT_NAME} stdc++fs common numa pthread ${LIB_NDCTL} dl gcov)

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
install(TARGETS ${PROJECT_NAME}-static ARCHIVE DESTINATION lib)
