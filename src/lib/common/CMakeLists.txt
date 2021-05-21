cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(common CXX)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-g -pedantic -Wall -Wextra -Werror -Wcast-align -Wcast-qual -Wconversion -Weffc++ -Wold-style-cast -Wredundant-decls -Wshadow -Wtype-limits -Wunused-parameter -Wwrite-strings -Wformat=2)

add_subdirectory(unit_test)

include(../../../mk/clang-dev-tools.cmake)

configure_file(include/config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/config.h)

execute_process( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )

include_directories(./include)
include_directories(../GSL/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include) # daxctl

link_directories(${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
link_directories(${CMAKE_INSTALL_PREFIX}/lib)

set(CMAKE_INCLUDE_CURRENT_DIR true)
set(CMAKE_INSTALL_RPATH "/usr/lib64")

if( ${ARCHITECTURE} STREQUAL "x86_64" )
  add_compile_options(-msse4.2)
endif()

add_definitions(-DCONFIG_DEBUG)

file(GLOB SOURCES src/*.c* perf/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES})
add_library(${PROJECT_NAME}-static STATIC ${SOURCES})

target_compile_options(${PROJECT_NAME} PUBLIC "-fPIC")
target_compile_options(${PROJECT_NAME}-static PUBLIC "-fPIC")

target_link_libraries(${PROJECT_NAME} dl gcov numa)

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
install(TARGETS ${PROJECT_NAME}-static ARCHIVE DESTINATION lib)
install(DIRECTORY "include/common" DESTINATION include FILES_MATCHING PATTERN "*.h*")
install(DIRECTORY "include/common/perf" DESTINATION include FILES_MATCHING PATTERN "*.h*")
install(DIRECTORY "include/component" DESTINATION include FILES_MATCHING PATTERN "*.h*")
