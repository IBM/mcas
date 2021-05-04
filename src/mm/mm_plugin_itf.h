#ifndef __MM_PLUGIN_ITF_H__
#define __MM_PLUGIN_ITF_H__

#include <stdlib.h>

#if defined(__cplusplus)
extern "C"
{
#endif

  typedef int status_t;
  typedef void * mm_plugin_heap_t;
  typedef void (*request_memory_callback_t)(void * param, size_t alignment, size_t size_hint, void * addr_hint);
  typedef void (*release_memory_callback_t)(void * param, void * addr, size_t size);

  /** 
   * Initialize mm library
   * 
   * @return S_OK, E_FAIL
   */
  status_t mm_plugin_init();

  /** 
   * Create a heap instance
   * 
   * @param params Constructor parameters (e.g., JSON)
   * @param out_heap Heap context 
   * 
   * @return S_OK, E_FAIL
   */
  status_t mm_plugin_create(const char * params, mm_plugin_heap_t * out_heap);

  /** 
   * Delete a heap instance
   * 
   * @param heap Heap context to destroy
   * 
   * @return S_OK, E_INVAL, E_FAIL
   */
  status_t mm_plugin_destroy(mm_plugin_heap_t heap);
  
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
  status_t mm_plugin_add_managed_region(mm_plugin_heap_t heap,
                                        void * region_base,
                                        size_t region_size);
  
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
  status_t mm_plugin_query_managed_region(mm_plugin_heap_t heap,
                                          unsigned region_id,
                                          void** out_region_base,
                                          size_t* out_region_size);
  
  /** 
   * Register callback the allocator can use to request more memory
   * 
   * @param heap Heap context
   * @param callback Call back function pointer
   * @param param Optional parameter which will be pass to callback function
   * 
   * @return E_NOT_IMPL, S_OK
   */
  status_t mm_plugin_register_callback_request_memory(mm_plugin_heap_t heap,
                                                      request_memory_callback_t callback,
                                                      void * param);
  
  /** 
   * Allocate a region of memory without alignment or hint
   * 
   * @param heap Heap context
   * @param Length in bytes
   * @param [out] Pointer to allocated region
   *
   * @return S_OK or E_FAIL
   */
  status_t mm_plugin_allocate(mm_plugin_heap_t heap, size_t n, void ** out_ptr);
  
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
  status_t mm_plugin_aligned_allocate(mm_plugin_heap_t heap, size_t n, size_t alignment, void ** out_ptr);
  
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
  status_t mm_plugin_aligned_allocate_offset(mm_plugin_heap_t heap, size_t n, size_t alignment, size_t offset, void ** out_ptr);

  /** 
   * Free a previously allocated region of memory with length known
   * 
   * @param heap Heap context
   * @param ptr Pointer to previously allocated region
   * @param size Length of region in bytes
   *
   * @return S_OK or E_INVAL;
   */
  status_t mm_plugin_deallocate(mm_plugin_heap_t heap, void * ptr, size_t size);

  /** 
   * Free previously allocated region without known length
   * 
   * @param heap Heap context
   * @param ptr Pointer to region
   * 
   * @return S_OK
   */
  status_t mm_plugin_deallocate_without_size(mm_plugin_heap_t heap, void * ptr);

  /** 
   * Allocate region and zero memory
   * 
   * @param heap Heap context
   * @param size Size of region in bytes
   * @param ptr [out] Pointer to allocated region
   * 
   * @return S_OK
   */
  status_t mm_plugin_callocate(mm_plugin_heap_t heap, size_t n, void ** out_ptr);

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
  status_t mm_plugin_reallocate(mm_plugin_heap_t heap, void * ptr, size_t size, void ** out_ptr);

  /** 
   * Get the number of usable bytes in block pointed to by ptr.  The
   * allocator *may* not have this information and should then return
   * E_NOT_IMPL. Returned size may be larger than requested allocation
   * 
   * @param heap Heap context
   * @param ptr Pointer to block base
   * @param out_ptr [out] Number of bytes in allocated block
   * 
   * @return S_OK, E_NOT_IMPL
   */
  status_t mm_plugin_usable_size(mm_plugin_heap_t heap, void * ptr, size_t * out_size);


  /** 
   * Inject an allocation back into the allocator (reconstituing)
   * 
   * @param heap Heap context
   * @param ptr Pointer to region of memory to mark allocated
   * @param size Size of region in bytes
   * 
   * @return S_OK, E_NOT_IMPL
   */
  status_t mm_plugin_inject_allocation(mm_plugin_heap_t heap, void * ptr, size_t size);

  /** 
   * Get debugging information
   * 
   * @param heap Heap context
   */
  void mm_plugin_debug(mm_plugin_heap_t heap);



  /** 
   * Function pointer table for all methods
   * 
   */
  typedef struct tag_mm_plugin_function_table_t
  {
    status_t (*mm_plugin_init)();
    status_t (*mm_plugin_create)(const char * params, mm_plugin_heap_t * out_heap);
    status_t (*mm_plugin_destroy)(mm_plugin_heap_t heap);
    status_t (*mm_plugin_add_managed_region)(mm_plugin_heap_t heap,
                                             void * region_base,
                                             size_t region_size);
    status_t (*mm_plugin_query_managed_region)(mm_plugin_heap_t heap,
                                               unsigned region_id,
                                               void** out_region_base,
                                               size_t* out_region_size);
    status_t (*mm_plugin_register_callback_request_memory)(mm_plugin_heap_t heap,
                                                           request_memory_callback_t callback,
                                                           void * param);
    status_t (*mm_plugin_allocate)(mm_plugin_heap_t heap, size_t n, void ** out_ptr);
    status_t (*mm_plugin_aligned_allocate)(mm_plugin_heap_t heap, size_t n, size_t alignment, void ** out_ptr);
    status_t (*mm_plugin_aligned_allocate_offset)(mm_plugin_heap_t heap, size_t n, size_t alignment, size_t offset, void ** out_ptr);
    status_t (*mm_plugin_deallocate)(mm_plugin_heap_t heap, void * ptr, size_t size);
    status_t (*mm_plugin_deallocate_without_size)(mm_plugin_heap_t heap, void * ptr);
    status_t (*mm_plugin_callocate)(mm_plugin_heap_t heap, size_t n, void ** out_ptr);
    status_t (*mm_plugin_reallocate)(mm_plugin_heap_t heap, void * ptr, size_t size, void ** out_ptr);
    status_t (*mm_plugin_usable_size)(mm_plugin_heap_t heap, void * ptr, size_t * out_size);
    void     (*mm_plugin_debug)(mm_plugin_heap_t heap);
    status_t (*mm_plugin_inject_allocation)(mm_plugin_heap_t heap, void * ptr, size_t size);
  } mm_plugin_function_table_t;

#if defined(__cplusplus)
}
#endif  


#endif // __MM_PLUGIN_ITF_H__
