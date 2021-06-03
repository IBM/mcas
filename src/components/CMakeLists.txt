cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

add_compile_options(-pedantic -Wall -Wextra -Werror -Wcast-align -Wcast-qual -Wconversion -Weffc++ -Wformat=2 -Wold-style-cast -Wredundant-decls -Wshadow -Wtype-limits -Wunused-parameter -Wwrite-strings)

if(BOOTSTRAP_DEPS)
else(BOOTSTRAP_DEPS)
  add_subdirectory(api)
  add_subdirectory(net)
  add_subdirectory(store)
  add_subdirectory(ado)
  add_subdirectory(client)
  add_subdirectory(index)  
  add_subdirectory(cluster)
  add_subdirectory(crypto)
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/enterprise-only")
  add_subdirectory(enterprise-only)
endif()

endif(BOOTSTRAP_DEPS)
