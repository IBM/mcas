#include <assert.h>
#include <stdlib.h>
#include <malloc.h>

#include <common/logging.h>
#include <common/errors.h>

#include "../../mm_plugin_itf.h"

#define DEBUG

#include "safe_print.h"

struct heap_t {
  size_t _alloc_count;
  size_t _free_count;
};

/** 
 * Initialize mm library
 * 
 */
PUBLIC status_t mm_plugin_init()
{
  return S_OK;
}

/** 
 * Create a heap instance (pass through doesn't understand this)
 * 
 * @param params Constructor parameters (e.g., JSON)
 * @param out_heap Heap context 
 * 
 * @return 
 */
PUBLIC status_t mm_plugin_create(const char * params, void * root_ptr, mm_plugin_heap_t * out_heap)
{
  struct heap_t * inst = malloc(sizeof(struct heap_t));
  memset(inst, 0, sizeof(struct heap_t));
  
  *out_heap = inst;
  
  return S_OK;
}

PUBLIC status_t mm_plugin_destroy(mm_plugin_heap_t heap)
{
  return E_NOT_IMPL;
}
  
  
/** 
 * Add (slab) region of memory to fuel the allocator.  The passthru
 * allocator does not need this as it takes directly from the OS.
 * 
 * @param heap Heap context
 * @param region_base Pointer to beginning of region
 * @param region_size Region length in bytes
 * 
 * @return E_NOT_IMPL, S_OK, E_FAIL, E_INVAL
 */
PUBLIC status_t mm_plugin_add_managed_region(mm_plugin_heap_t heap,
                                      void * region_base,
                                      size_t region_size)
{
  return E_NOT_IMPL;
}
  
  
/** 
 * Query memory regions being used by the allocator
 * 
 * @param heap Heap context
 * @param region_id Index counting from 0
 * @param [out] Base address of region
 * @param [out] Size of region in bytes
 * 
 * @return S_MORE (continue to next region id), S_OK (done), E_INVAL
 */  
PUBLIC status_t mm_plugin_query_managed_region(mm_plugin_heap_t heap,
                                               unsigned region_id,
                                               void** out_region_base,
                                               size_t* out_region_size)
{
  return E_NOT_IMPL;
}
  
  
/** 
 * Register callback the allocator can use to request more memory
 * 
 * @param heap Heap context
 * @param callback Call back function pointer
 * @param param Optional parameter which will be pass to callback function
 * 
 * @return E_NOT_IMPL, S_OK
 */
PUBLIC status_t mm_plugin_register_callback_request_memory(mm_plugin_heap_t heap,
                                                           request_memory_callback_t callback,
                                                           void * param)
{
  return E_NOT_IMPL;
}
  
  
/** 
 * Allocate a region of memory without alignment or hint
 * 
 * @param heap Heap context
 * @param Length in bytes
 * @param [out] Pointer to allocated region
 *
 * @return S_OK or E_FAIL
 */
PUBLIC status_t mm_plugin_allocate(mm_plugin_heap_t heap, size_t n, void ** out_ptr)
{
  assert(out_ptr);    
  *out_ptr = malloc(n);

  struct heap_t* h = heap;
  h->_alloc_count++;
  SAFE_PRINT("MM [%lu]: PASSTHRU\t - ALLOC(%lu) --> %p",
             h->_alloc_count, n, *out_ptr);

  return S_OK;
}

  
/** 
 * Allocation region of memory that is aligned
 * 
 * @param heap Heap context
 * @param n Size of region in bytes
 * @param alignment Alignment in bytes
 * @param out_ptr [out] Pointer to allocated region
 * 
 * @return S_OK, E_FAIL, E_INVAL (depending on implementation)
 */
PUBLIC status_t mm_plugin_aligned_allocate(mm_plugin_heap_t heap, size_t n, size_t alignment, void ** out_ptr)
{
  assert(out_ptr);
  /* don't use aligned_alloc because its statically defined */
  *out_ptr = memalign(alignment, n);
  struct heap_t * h = heap;
  h->_alloc_count++;
  SAFE_PRINT("MM [%lu]: PASSTHRU\t - ALIGNED_ALLOC(%lu, %lu) --> %p",
             ((struct heap_t*)heap)->_alloc_count, n, alignment, *out_ptr);

  return S_OK;
}

  
/** 
 * Special case for EASTL
 * 
 * @param heap Heap context
 * @param n Size of region in bytes
 * @param alignment Alignment in bytes
 * @param offset Unknown
 * 
 * @return 
 */
PUBLIC status_t mm_plugin_aligned_allocate_offset(mm_plugin_heap_t heap, size_t n, size_t alignment, size_t offset, void ** out_ptr)
{
  return S_OK;
}


/** 
 * Free a previously allocated region of memory with length known
 * 
 * @param heap Heap context
 * @param ptr Pointer to previously allocated region
 * @param size Length of region in bytes
 *
 * @return S_OK or E_INVAL;
 */
PUBLIC status_t mm_plugin_deallocate(mm_plugin_heap_t heap, void * ptr, size_t size)
{
  free(ptr);

  ((struct heap_t*)heap)->_free_count++;
  return S_OK;
}
  

/** 
 * Free previously allocated region without known length
 * 
 * @param heap Heap context
 * @param ptr Pointer to region
 * 
 * @return S_OK
 */
PUBLIC status_t mm_plugin_deallocate_without_size(mm_plugin_heap_t heap, void * ptr)
{
  free(ptr);
  ((struct heap_t*)heap)->_free_count++;
  return S_OK;
}


/** 
 * Allocate region and zero memory
 * 
 * @param heap Heap context
 * @param size Size of region in bytes
 * @param ptr [out] Pointer to allocated region
 * 
 * @return S_OK
 */
PUBLIC status_t mm_plugin_callocate(mm_plugin_heap_t heap, size_t n, void ** out_ptr)
{
  /* for some reason calling calloc here breaks the interposition */
  void  * p = malloc(n);
  __builtin_memset(p, 0, n);
  *out_ptr = p;
  return S_OK;
}
  

/** 
 * Resize an existing allocation
 * 
 * @param heap Heap context
 * @param ptr Pointer to existing allocated region
 * @param size New size in bytes
 * @param ptr [out] New reallocated region or null on unable to reallocate
 * 
 * @return S_OK
 */
PUBLIC status_t mm_plugin_reallocate(mm_plugin_heap_t heap, void * ptr, size_t size, void ** out_ptr)
{
  *out_ptr = realloc(ptr, size);
  return S_OK;
}


/** 
 * Get the number of usable bytes in block pointed to by ptr.  The
 * allocator *may* not have this information and should then return
 * E_NOT_IMPL. Returned size may be larger than requested allocation
 * 
 * @param heap Heap context
 * @param ptr Pointer to block base
 * 
 * @return Number of bytes in allocated block
 */
PUBLIC status_t mm_plugin_usable_size(mm_plugin_heap_t heap, void * ptr, size_t * out_size)
{
  *out_size = malloc_usable_size(ptr);
  return S_OK;
}
  

/** 
 * Get debugging information
 * 
 * @param heap Heap context
 */
PUBLIC void mm_plugin_debug(mm_plugin_heap_t heap)
{
}

