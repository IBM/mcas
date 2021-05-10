#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "../../mm_plugin_itf.h"
#include "rc_alloc_lb.h"
#include "logging.h"
//#define DEBUG /* enable log output */

namespace global
{
static unsigned debug_level = 3;
}

using Heap = Rca_LB;

PUBLIC status_t mm_plugin_init()
{
  return S_OK;
}

PUBLIC status_t mm_plugin_create(const char * params, void * root_ptr, mm_plugin_heap_t * out_heap)
{
  PPLOG("mm_plugin_create (%s)", params);

  auto new_heap = new Heap(global::debug_level);
  *out_heap = reinterpret_cast<mm_plugin_heap_t>(new_heap);

  return S_OK;
}

PUBLIC status_t mm_plugin_destroy(mm_plugin_heap_t heap)
{
  auto h = reinterpret_cast<Heap*>(heap);
  delete h;
  return S_OK;
}

PUBLIC status_t mm_plugin_add_managed_region(mm_plugin_heap_t heap,
                                      void * region_base,
                                      size_t region_size)
{
  PPNOTICE("%s base=%p size=%lu",__func__, region_base, region_size);
  auto h = reinterpret_cast<Heap*>(heap);
  memset(region_base, 0, region_size);
  h->add_managed_region(region_base, region_size, 0 /* numa node */);
  return S_OK;
}

PUBLIC status_t mm_plugin_query_managed_region(mm_plugin_heap_t heap,
                                        unsigned region_id,
                                        void** out_region_base,
                                        size_t* out_region_size)
{
  PPLOG("%s",__func__);
  asm("int3");
  return E_NOT_IMPL;
}

PUBLIC status_t mm_plugin_register_callback_request_memory(mm_plugin_heap_t heap,
                                                    request_memory_callback_t callback,
                                                    void * param)
{
  PPLOG("%s",__func__);
  asm("int3");
  return E_NOT_IMPL;
}

PUBLIC status_t mm_plugin_allocate(mm_plugin_heap_t heap, size_t n, void ** out_ptr)
{
  auto h = reinterpret_cast<Heap*>(heap);
  *out_ptr = h->alloc(n, 0);
  PPLOG("%s (%lu) -> %p",__func__, n, *out_ptr);
  return S_OK;
}

PUBLIC status_t mm_plugin_aligned_allocate(mm_plugin_heap_t heap, size_t n, size_t alignment, void ** out_ptr)
{
  PPLOG("%s (%lu,%lu)",__func__, n, alignment);
  auto h = reinterpret_cast<Heap*>(heap);
  *out_ptr = h->alloc(n, 0, alignment);
  assert(*out_ptr);
  return S_OK;
}

PUBLIC status_t mm_plugin_aligned_allocate_offset(mm_plugin_heap_t heap, size_t n, size_t alignment, size_t offset, void ** out_ptr)
{
  asm("int3");  
  return E_NOT_IMPL;
}

PUBLIC status_t mm_plugin_deallocate(mm_plugin_heap_t heap, void * ptr, size_t n)
{
  if(ptr == nullptr || n == 0) return S_OK;
  PPLOG("%s (%p, %lu)",__func__, ptr, n);
  auto h = reinterpret_cast<Heap*>(heap);
  h->free(ptr, 0); // should use size   h->free(ptr, 0);
  return S_OK;
}

PUBLIC status_t mm_plugin_deallocate_without_size(mm_plugin_heap_t heap, void * ptr)
{
  PPLOG("%s (%p)",__func__, ptr);
  auto h = reinterpret_cast<Heap*>(heap);
  h->free(ptr, 0);
  return S_OK;
}

PUBLIC status_t mm_plugin_callocate(mm_plugin_heap_t heap, size_t n, void ** out_ptr)
{
  if(out_ptr == nullptr) return E_INVAL;

  auto h = reinterpret_cast<Heap*>(heap);
  *out_ptr = h->alloc(n, 0);
  ::memset(*out_ptr, 0, n);
  PPLOG("%s (%lu)",__func__, n);
  return S_OK;
}

PUBLIC status_t mm_plugin_reallocate(mm_plugin_heap_t heap, void * ptr, size_t n, void ** out_ptr)
{
  if(ptr == nullptr)
    return mm_plugin_allocate(heap, n, out_ptr);

  if(n == 0 && ptr != nullptr)
    return mm_plugin_deallocate_without_size(heap, ptr);

  PPLOG("%s (%p, %lu)",__func__, ptr, n);
  *out_ptr = nullptr;
  return E_NOT_IMPL; /* we don't support reallocation */
}

PUBLIC status_t mm_plugin_usable_size(mm_plugin_heap_t heap, void * ptr, size_t * out_size)
{
  asm("int3");
  return E_NOT_IMPL;
}

PUBLIC status_t mm_plugin_inject_allocation(mm_plugin_heap_t heap, void * ptr, size_t size)
{
  auto h = reinterpret_cast<Heap*>(heap);
  h->inject_allocation(ptr, size, 0);
  return S_OK;
}

PUBLIC void mm_plugin_debug(mm_plugin_heap_t heap)
{
}

#pragma GCC diagnostic pop
