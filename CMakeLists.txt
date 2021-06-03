cmake_minimum_required(VERSION 3.5)

option(FORCE_OTT_BUILD "force out of tree build" ON)
option(BOOTSTRAP_DEPS "bootstrap to build deps" ON)

# build scope options
option(BUILD_MCAS_SERVER "build MCAS server and component dependencies" ON)
option(BUILD_MCAS_CLIENT "build MCAS client libraries" ON)
option(BUILD_KERNEL_SUPPORT "build kernel module dependencies" ON)
option(BUILD_COMPONENT_CRYPTO "build crypto component" ON)
option(BUILD_PYTHON_SUPPORT "build Python support" ON)
option(BUILD_RUST "enabled Rust-based dependencies" ON)
option(BUILD_EXAMPLES_PMDK "build PMDK-based ADO examples (requires PMDK installed)" OFF)
option(BUILD_MPI_APPS "build mcas-mpi-bench and any other MPI dependent applications" OFF)


# features
option(FEATURE_TIMESTAMPS "feature: enable timestamps in hstore map" ON)
option(FEATURE_POOL_ACL "feature: enable pool access control" OFF)

# debug options
option(TEST_HSTORE_PERISHABLE "hstore perishable testing enabled" OFF)
option(CODE_COVERAGE "enable gcc code coverage" OFF)
option(SPAN_USES_GSL "span data uses gsl::span, not iovec" OFF)
option(BYTE_USES_STD "byte data uses std::byte, not gsl::byte" OFF)
option(INVASIVE_PERF "enable invasive performance trace" OFF)
option(PROFILE "profile the code" OFF)

set(DM_REGION_LOG_GRAIN_SIZE 25 CACHE STRING "log base 2 of the device_manager allocation granularity")

if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR AND FORCE_OTT_BUILD)
  message(FATAL_ERROR "Cannot use in-source build ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}. You should delete any CMakeCache.txt and CMakeFiles and then try out-of-tree build")
endif(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR AND FORCE_OTT_BUILD)

# check for Rust installation
find_program(RUST_COMPILER rustc ~/.cargo/bin/)
find_program(RUST_CARGO cargo ~/.cargo/bin/)
if(NOT ${RUST_COMPILER} MATCHES "RUST_COMPILER-NOTFOUND")
  message("-- Found Rust compiler: ${RUST_COMPILER}")
  message("-- Found Rust Cargo: ${RUST_CARGO}")
else()
  set(BUILD_RUST OFF)
endif()

#add_compile_options(-shared-libgcc) # for exception support when using gcc (not g++) to link
add_compile_options(-g)
add_definitions(-DTBB_USE_THREADING_TOOLS -DEASTL_STD_ITERATOR_CATEGORY_ENABLED)

if(SPAN_USES_GSL)
 add_definitions(-DMCAS_SPAN_USES_GSL=1)
endif()
if(BYTE_USES_STD)
 add_definitions(-DMCAS_BYTE_USES_STD=1)
endif()

if(INVASIVE_PERF)
  add_definitions(-DMCAS_TM_ENABLED=1)
else()
  add_definitions(-DMCAS_TM_ENABLED=0)
endif()

if(FEATURE_TIMESTAMPS)
  add_definitions(-DENABLE_TIMESTAMPS=1)
else()
  add_definitions(-DENABLE_TIMESTAMPS=0)
endif()

set(ENV{MCAS_HOME} ${CMAKE_CURRENT_SOURCE_DIR})

include(CheckIncludeFiles)
find_package(PkgConfig)
include(${CMAKE_CURRENT_SOURCE_DIR}/mk/common.cmake)
project(comanche)

# locate FindXXX.cmake
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/mk)
message("CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

if(PROFILE)
  find_library(PROFILER_LIB NAMES profiler PATHS /usr/local/lib,/usr/local/lib64,/usr/lib,/usr/lib64)
  if ("${PROFILER_LIB}" STREQUAL "PROFILER_LIB-NOTFOUND")
    set(PROFILER "")
    message("profiler not found")
  else()
    set(PROFILER "profiler")
    add_definitions(-DHAS_PROFILER=1)
    add_definitions(-DPROFILE=1) # mcas server expectation
    message("profiler found")
  endif()
else()
  message("profiler not requested")
endif()

# run 'make bootstrap' to  make comanche deps and core
# then run 'make' to build components/tests
add_custom_target(bootstrap
  bash ${CMAKE_SOURCE_DIR}/mk/bootstrap.sh ${CMAKE_SOURCE_DIR}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

# bootstrap build should be performed one off
add_subdirectory(src)
add_subdirectory(testing)
add_subdirectory(examples)
add_subdirectory(deploy)


