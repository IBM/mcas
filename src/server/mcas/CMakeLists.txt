cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include(../../../mk/common.cmake)

# use this to disable optimizations, e.g. for debugging or profiling
add_compile_options("$<$<CONFIG:Debug>:-O0>")

project(mcas CXX)

if(FEATURE_POOL_ACL)
  add_definitions(-DFEATURE_POOL_ACL)
endif(FEATURE_POOL_ACL)

include(../../../mk/clang-dev-tools.cmake)

include_directories(${CMAKE_SOURCE_DIR}/src/lib/common/include/)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libpmem/include/)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libnupm/include/)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libadoproto/include/)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libthreadipc/include/)
include_directories(${CMAKE_SOURCE_DIR}/src/components/)
include_directories(${CMAKE_SOURCE_DIR}/src/kernel/modules/xpmem/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include) # rapidjson


link_directories(${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}) # tbb (via nupm)
link_directories(${CMAKE_INSTALL_PREFIX}/lib) # tbb (via nupm)

add_definitions(-DCONFIG_DEBUG)
add_compile_options("$<$<CONFIG:Release>:-DNDEBUG>")

configure_file(example-ado.conf.in ${CMAKE_CURRENT_SOURCE_DIR}/example-ado.conf)
configure_file(src/env.h.in ${CMAKE_CURRENT_SOURCE_DIR}/src/env.h)

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

add_executable(mcas src/main.cpp src/shard.cpp src/shard_ado.cpp
src/connection_handler.cpp src/ado_manager.cpp src/security.cpp
src/fabric_connection_base.cpp src/fabric_transport.cpp
src/config_file.cpp src/tls_session.cpp)

target_link_libraries(mcas ${ASAN_LIB} threadipc common numa pthread
dl nupm boost_program_options crypto z ado-proto xpmem gnutls
${PROFILER} )

#add_subdirectory(unit_test)

set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_INSTALL_PREFIX}/lib)

add_dependencies(${PROJECT_NAME} threadipc common nupm)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(FILES "example-ado.conf" DESTINATION bin)
install(FILES "mcas-logged.sh" DESTINATION bin
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)
