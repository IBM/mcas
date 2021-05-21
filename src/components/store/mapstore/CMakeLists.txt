cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(component-mapstore CXX)

# for serious debugging
#add_compile_options("$<$<CONFIG:Debug>:-O0>")

add_definitions(-DCONFIG_DEBUG)
add_definitions(-DDM_REGION_LOG_GRAIN_SIZE=${DM_REGION_LOG_GRAIN_SIZE})

add_subdirectory(./unit_test)

include_directories(${CMAKE_SOURCE_DIR}/src/lib/common/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/GSL/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libnupm/include)
include_directories(${CMAKE_SOURCE_DIR}/src/components)
include_directories(${CMAKE_SOURCE_DIR}/src/mm)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/cityhash/cityhash/src)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

link_directories(${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
link_directories(${CMAKE_INSTALL_PREFIX}/lib) # cityhash

file(GLOB SOURCES src/*.c*)
configure_file(src/map_store_env.h.in ${CMAKE_CURRENT_BINARY_DIR}/map_store_env.h)

add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_compile_options(${PROJECT_NAME} PUBLIC "-fPIC")

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
target_link_libraries(${PROJECT_NAME} common numa dl rt pthread cityhash)

set_target_properties(${PROJECT_NAME} PROPERTIES
  INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

install (TARGETS ${PROJECT_NAME}
    LIBRARY
    DESTINATION lib)

