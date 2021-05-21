cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 14)

project(mm-plugin-rcalb C CXX)

add_definitions(-DCONFIG_DEBUG)
add_compile_options("-fvisibility=hidden")

include_directories(${CMAKE_SOURCE_DIR}/src/lib/common/include)
include_directories(${CMAKE_SOURCE_DIR}/src/components)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libmm/src)
include_directories(${CMAKE_INSTALL_PREFIX}/include)
  
file(GLOB SOURCES src/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

#set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions -Wl,--wrap=malloc -Wl,--wrap=free -Wl,--wrap=calloc -Wl,--wrap=realloc -Wl,--wrap=memalign")

target_compile_options(${PROJECT_NAME} PUBLIC "-fPIC")

target_link_libraries(${PROJECT_NAME} -static-libgcc -static-libstdc++ common-static dl pthread)

# set the linkage in the install/lib
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

install(TARGETS ${PROJECT_NAME}  LIBRARY DESTINATION lib)


