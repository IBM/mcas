cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(personality-python CXX)

add_subdirectory(pymcas)

include(${CMAKE_SOURCE_DIR}/mk/avoid_boost_type_to_string_bug.cmake)

find_program(PYTHON NAMES "python3.7" "python3.6")
find_program(FLATC NAMES flatc PATHS "${CMAKE_INSTALL_PREFIX}/bin" NO_DEFAULT_PATH)

if (NOT FLATC)
  message(FATAL_ERROR ": Flatc compiler not found")
endif()

if (NOT PYTHON)
  message(FATAL_ERROR ": Python not found")  
endif()

get_filename_component(PYTHON_VER ${PYTHON} NAME)                                                                                
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# get python locations
execute_process ( COMMAND ${PYTHON} -m site --user-site OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process ( COMMAND ${PYTHON} -c "from sysconfig import get_paths as gp; print(gp()['include'])" OUTPUT_VARIABLE PYTHON_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process ( COMMAND ${PYTHON} -c "from sysconfig import get_config_vars as gc; print(gc()['LIBPL'])" OUTPUT_VARIABLE PYTHON_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process ( COMMAND ${PYTHON} -c "from sysconfig import get_config_vars as gc; print(gc()['BLDLIBRARY'])" OUTPUT_VARIABLE PYTHON_BLDLIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)

message("-- Python    site : ${PYTHON_SITE_PACKAGES}")
message("--         stdlib : ${PYTHON_LIB_DIR}")
message("--        include : ${PYTHON_INCLUDE_DIR}")

set(GENERATED ${CMAKE_CURRENT_BINARY_DIR}/generated/pp_generated.h)

set(PLUGIN_SOURCES ${GENERATED} src/pp_plugin.cpp pymcas/src/pymcas_core_module.cc pymcas/src/ndarray_helpers.cc)
set(CLIENT_SOURCES ${GENERATED} )

set(TEST_SOURCES ./src/pp_test.cpp)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/generated)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/common/include)
include_directories(${CMAKE_SOURCE_DIR}/src/components)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/flatbuffers/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libnupm/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libpmem/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libccpm/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libpmem/common)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libadoproto/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include) # EASTL
include_directories(${PYTHON_INCLUDE_DIR})

# uses local site install
include_directories(${PYTHON_SITE_PACKAGES}/numpy/core/include)
include_directories(${CMAKE_SOURCE_DIR}/src/python/mcas-python-api/src/)
  
link_directories(${CMAKE_BINARY_DIR}/src/lib/flatbuffers/)
link_directories(${PYTHON_LIB_DIR})
link_directories(${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
link_directories(${CMAKE_INSTALL_PREFIX}/lib)
  
add_custom_command(
  OUTPUT ${GENERATED}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/pp.fbs
  COMMAND ${FLATC} --cpp -o ${CMAKE_CURRENT_BINARY_DIR}/generated/ -c ${CMAKE_CURRENT_SOURCE_DIR}/src/pp.fbs

  COMMAND touch ${GENERATED}
)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} ${FLAG_DUMP_CLASS} -DCONFIG_DEBUG)
add_compile_options("$<$<CONFIG:Debug>:-O0>")
add_compile_options(-g -Wall -Wextra -Wcast-align -Wcast-qual -Wconversion -Wredundant-decls -Wshadow -Wtype-limits -Wno-unused-parameter -Wwrite-strings)

add_library(${PROJECT_NAME} SHARED ${PLUGIN_SOURCES})

target_link_libraries(${PROJECT_NAME} common pthread numa dl rt flatbuffers ${PYTHON_BLDLIB_DIR})

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH
  ${CMAKE_BINARY_DIR}/src/lib/flatbuffers/:${CMAKE_INSTALL_PREFIX}/lib:${PYTHON_LIB_DIR})

#configure_file(pp-ccpm.conf.in ${CMAKE_CURRENT_BINARY_DIR}/pp-ccpm.conf)
#configure_file(src/pp_ccpm_test.sh.in ${CMAKE_CURRENT_BINARY_DIR}/pp_ccpm_test.sh)

install (TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
  DESTINATION conf
  FILES_MATCHING
  PATTERN "*.conf"
  PATTERN CMakeFiles EXCLUDE
  PATTERN generated EXCLUDE
  PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_WRITE GROUP_WRITE WORLD_WRITE)

# install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
#   DESTINATION lib
#   FILES_MATCHING
#   PATTERN "*.py"
#   PATTERN CMakeFiles EXCLUDE
#   PATTERN generated EXCLUDE
#   PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE)

#install(FILES src/pp_ccpm_test.py
#  DESTINATION lib)

#install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pp_ccpm_test.sh
#  PERMISSIONS OWNER_EXECUTE OWNER_READ WORLD_EXECUTE WORLD_READ
#  DESTINATION bin)

#install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated/Proto
#  DESTINATION lib64/python)



