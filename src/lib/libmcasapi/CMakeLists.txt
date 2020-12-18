cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(mcasapi CXX C)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-pedantic -Wall -Wextra -Werror -Wcast-align -Wcast-qual -Wconversion -Wformat=2 -Wredundant-decls -Wshadow -Wtype-limits -Wunused-parameter -Wwrite-strings)
add_definitions(-DDM_REGION_LOG_GRAIN_SIZE=${DM_REGION_LOG_GRAIN_SIZE})

include(../../../mk/clang-dev-tools.cmake)

# find_library(LIB_JEMALLOC libjemalloc.so HINTS ${CMAKE_INSTALL_PREFIX}/lib)
#find_library(LIB_NDCTL libndctl.so HINTS ${CMAKE_INSTALL_PREFIX}/lib)

include_directories(${CMAKE_SOURCE_DIR}/src/components)
# include_directories(${CMAKE_SOURCE_DIR}/src/lib/libpmem/common)
# include_directories(${CMAKE_SOURCE_DIR}/src/lib/libpmem)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories("../common/include")
# include_directories(${CMAKE_SOURCE_DIR}/src/lib/cityhash/cityhash/src)
 include_directories(${CMAKE_INSTALL_PREFIX}/include) # daxctl

# link_directories(${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}) # tbb tbbmalloc
# link_directories(${CMAKE_INSTALL_PREFIX}/lib) # tbb tbbmalloc

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fopenmp")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

add_definitions(-DCONFIG_DEBUG)

file(GLOB SOURCES src/*.c*)

add_library(mcasapi SHARED ${SOURCES})

target_compile_options(${PROJECT_NAME} PUBLIC $<$<CONFIG:Debug>:-O0>)
target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Weffc++ -Wold-style-cast>)
target_link_libraries(${PROJECT_NAME} common dl )

add_subdirectory(unit_test)

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
install(DIRECTORY "src/" DESTINATION "include" FILES_MATCHING PATTERN "mcas_api_wrapper.h")
