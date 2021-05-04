cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(nupm CXX C)

set(CMAKE_CXX_STANDARD 17)
include(${CMAKE_SOURCE_DIR}/mk/avoid_boost_type_to_string_bug.cmake)

add_compile_options(-pedantic -Wall -Wextra -Werror -Wcast-align -Wcast-qual -Wconversion -Wformat=2 -Wredundant-decls -Wshadow -Wtype-limits -Wunused-parameter -Wwrite-strings)
add_definitions(-DDM_REGION_LOG_GRAIN_SIZE=${DM_REGION_LOG_GRAIN_SIZE})

include(../../../mk/clang-dev-tools.cmake)

# find_library(LIB_JEMALLOC libjemalloc.so HINTS ${CMAKE_INSTALL_PREFIX}/lib)
find_library(LIB_NDCTL libndctl.so HINTS ${CMAKE_INSTALL_PREFIX}/lib)

include_directories(./include/nupm)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libpmem/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libpmem/common)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libpmem)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories("../common/include")
include_directories(${CMAKE_SOURCE_DIR}/src/lib/cityhash/cityhash/src)
include_directories(${CMAKE_INSTALL_PREFIX}/include) # daxctl

link_directories(${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}) # tbb tbbmalloc
link_directories(${CMAKE_INSTALL_PREFIX}/lib) # tbb tbbmalloc

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fopenmp")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

add_definitions(-DCONFIG_DEBUG)

file(GLOB SOURCES src/*.c*)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

target_compile_options(${PROJECT_NAME} PUBLIC $<$<CONFIG:Debug>:-O0>)
target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Weffc++ -Wold-style-cast>)
target_compile_options(${PROJECT_NAME} PUBLIC "-fPIC")
target_link_libraries(${PROJECT_NAME} stdc++fs common numa ${LIB_NDCTL} dl tbb tbbmalloc pmem cityhash gcov)

add_subdirectory(unit_test)

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)

install(DIRECTORY "src/" DESTINATION "include/nupm" FILES_MATCHING PATTERN "./include/nupm/*")
#install(DIRECTORY "include/nupm" DESTINATION include FILES_MATCHING PATTERN "*.h*")


# # force early install because others depend on this.  this seems a hack. is there
# # a better way?
# #
# add_custom_command(
#   TARGET ${PROJECT_NAME} POST_BUILD
#   COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/include/nupm
#   COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/lib
#   COMMAND cp ${PROJECT_SOURCE_DIR}/src/*.h ${CMAKE_INSTALL_PREFIX}/include/nupm
#   COMMAND cp -r ${PROJECT_SOURCE_DIR}/src/x86_64 ${CMAKE_INSTALL_PREFIX}/include/nupm
#   COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/libnupm.so ${CMAKE_INSTALL_PREFIX}/lib
#   )

