cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(lib)

if(BOOTSTRAP_DEPS)

add_subdirectory(kernel/modules)

else(BOOTSTRAP_DEPS)

# see https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html
add_compile_options($<$<BOOL:${CODE_COVERAGE}>:--coverage>)
add_subdirectory(components)
add_subdirectory(mm)
add_subdirectory(server)
add_subdirectory(python)
add_subdirectory(apps)
# EASTL test build is normally OFF. To build EASTL tests, cmake with -DEASTL_BUILD_TESTS=1
add_subdirectory(ado)

if(BUILD_RUST)
  add_subdirectory(rust/mcasapi)
endif(BUILD_RUST)

if(BUILD_KERNEL_SUPPORT)
  add_subdirectory(kernel/modules/mcas/unit_test)
endif(BUILD_KERNEL_SUPPORT)

endif(BOOTSTRAP_DEPS)
