#include <stdlib.h>
#include <unistd.h>
#include "mm_rcalb.h"


/**
 * Factory entry point
 *
 */
extern "C" void* factory_createInstance(component::uuid_t component_id)
{
  if (component_id == Rca_LB_memory_manager_factory::component_id()) {
    return static_cast<void*>(new Rca_LB_memory_manager_factory());
  }
  else
    return NULL;
}
