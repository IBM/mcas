/*
   Copyright [2021] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef __MM_PLUGIN_ITF_H__
#define __MM_PLUGIN_ITF_H__

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"

#define PUBLIC __attribute__((__visibility__("default")))

#if defined(__cplusplus)
#pragma GCC diagnostic ignored "-Weffc++"
#endif

#include <stdlib.h>
#include <assert.h>

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
   * @param root_ptr Root point for persistent heaps
   * @param out_heap Heap context 
   * 
   * @return S_OK, E_FAIL
   */
  status_t mm_plugin_create(const char * params,
                            void * root_ptr,
                            mm_plugin_heap_t * out_heap);

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

  /*
    The POSIX realloc() function changes the size of the memory block pointed
    to by ptr to size bytes.  The contents will be unchanged in the
    range from the start of the region up to the minimum of the old and
    new sizes.  If the new size is larger than the old size, the added
    memory will not be initialized.  If ptr is NULL, then the call is
    equivalent to malloc(size), for all values of size; if size is equal
    to zero, and ptr is not NULL, then the call is equivalent to
    free(ptr).  Unless ptr is NULL, it must have been returned by an
    earlier call to malloc(), calloc(), or realloc().  If the area
    pointed to was moved, a free(ptr) is done.
  */

  /** 
   * Resize an existing allocation
   * 
   * @param heap Heap context
   * @param ptr Pointer to existing allocated region
   * @param size New size in bytes
   * @param ptr [out] New reallocated region or null on unable to reallocate
   *
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
    status_t (*mm_plugin_create)(const char * params, void * root_ptr, mm_plugin_heap_t * out_heap);
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


#if defined(__cplusplus)

#include <dlfcn.h>
#include <string>
#include <stdio.h>
#include <stdexcept>

#define LOAD_SYMBOL(X) _ft.X = reinterpret_cast<typeof(_ft.X)>(dlsym(_module, # X)); assert(_ft.X)

/** 
 * C++ wrapper on C-based plugin API
 * 
 */
class MM_plugin_wrapper
{
public:
    
  MM_plugin_wrapper(const std::string& plugin_path,
                    const std::string& config = "",
                    void * root_ptr = nullptr) {
    assert(plugin_path.empty() == false);
    _module = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_NODELETE); // RTLD_DEEPBIND | 

    if(_module == nullptr) {
      char err[1024];
      sprintf(err, "%s\n", dlerror());
      throw std::invalid_argument(err);
    }

    LOAD_SYMBOL(mm_plugin_init);
    LOAD_SYMBOL(mm_plugin_create);
    LOAD_SYMBOL(mm_plugin_add_managed_region);
    LOAD_SYMBOL(mm_plugin_query_managed_region);
    LOAD_SYMBOL(mm_plugin_register_callback_request_memory);
    LOAD_SYMBOL(mm_plugin_allocate);
    LOAD_SYMBOL(mm_plugin_aligned_allocate);
    LOAD_SYMBOL(mm_plugin_aligned_allocate_offset);
    LOAD_SYMBOL(mm_plugin_deallocate);
    LOAD_SYMBOL(mm_plugin_deallocate_without_size);
    LOAD_SYMBOL(mm_plugin_callocate);
    LOAD_SYMBOL(mm_plugin_reallocate);
    LOAD_SYMBOL(mm_plugin_usable_size);
    LOAD_SYMBOL(mm_plugin_debug);
    LOAD_SYMBOL(mm_plugin_destroy);

    //      dlclose(_module);
      
    /* create heap instance */      
    _ft.mm_plugin_create(config.c_str(), root_ptr, &_heap);
  }

  virtual ~MM_plugin_wrapper() noexcept {
    _ft.mm_plugin_destroy(_heap);
  }

  /* forwarding in liners */
  inline status_t init() noexcept {
    return _ft.mm_plugin_init();
  }
    

  inline status_t add_managed_region(void * region_base, size_t region_size) noexcept {
    return _ft.mm_plugin_add_managed_region(_heap, region_base, region_size);
  }

  inline status_t query_managed_region(unsigned region_id, void** out_region_base, size_t* out_region_size) noexcept {
    return _ft.mm_plugin_query_managed_region(_heap, region_id, out_region_base, out_region_size);
  }
    
  inline status_t register_callback_request_memory(request_memory_callback_t callback, void * param) noexcept {
    return _ft.mm_plugin_register_callback_request_memory(_heap, callback, param);
  }
    
  inline status_t allocate(size_t n, void ** out_ptr) noexcept {
    return _ft.mm_plugin_allocate(_heap, n, out_ptr);
  }
    
  inline status_t aligned_allocate(size_t n, size_t alignment, void ** out_ptr) noexcept {
    return _ft.mm_plugin_aligned_allocate(_heap, n, alignment, out_ptr);
  }
    
  inline status_t aligned_allocate_offset(size_t n, size_t alignment, size_t offset, void ** out_ptr) noexcept {
    return _ft.mm_plugin_aligned_allocate_offset(_heap, n, alignment, offset, out_ptr);
  }
    
  inline status_t deallocate(void * ptr, size_t size) noexcept {
    return _ft.mm_plugin_deallocate(_heap, ptr, size);
  }
    
  inline status_t deallocate_without_size(void * ptr) noexcept {
    return _ft.mm_plugin_deallocate_without_size(_heap, ptr);
  }
    
  inline status_t callocate(size_t n, void ** out_ptr) noexcept {
    return _ft.mm_plugin_callocate(_heap, n, out_ptr);
  }
    
  inline status_t reallocate(void * ptr, size_t size, void ** out_ptr) noexcept {
    return _ft.mm_plugin_reallocate(_heap, ptr, size, out_ptr);
  }
    
  inline status_t usable_size(void * ptr, size_t * out_size) noexcept {
    return _ft.mm_plugin_usable_size(_heap, ptr, out_size);
  }
    
  inline void debug(mm_plugin_heap_t heap) noexcept {
    return _ft.mm_plugin_debug(_heap);
  }
    
  inline status_t inject_allocation(void * ptr, size_t size) noexcept {
    return _ft.mm_plugin_inject_allocation(_heap, ptr, size);
  }

private:
  void *                     _module;
  mm_plugin_function_table_t _ft;
  mm_plugin_heap_t           _heap;
};

#include <limits>
#include <new>

/** 
 * Standard C++ allocator wrapper
 * 
 */
template <class T>
class MM_plugin_cxx_allocator
{
public:
  using value_type    = T;
  using pointer       = value_type*;
  using const_pointer = typename std::pointer_traits<pointer>::template
    rebind<value_type const>;
  using void_pointer       = typename std::pointer_traits<pointer>::template
    rebind<void>;
  using const_void_pointer = typename std::pointer_traits<pointer>::template
    rebind<const void>;
  
  using difference_type = typename std::pointer_traits<pointer>::difference_type;
  using size_type       = std::make_unsigned_t<difference_type>;
  
  template <class U> struct rebind {typedef MM_plugin_cxx_allocator<U> other;};

  MM_plugin_cxx_allocator(MM_plugin_wrapper& wrapper) noexcept : _wrapper(wrapper)
  {
  }
  
  template <class U>
  MM_plugin_cxx_allocator(MM_plugin_cxx_allocator<U> const& a_) noexcept : _wrapper(a_._wrapper)  {}

  pointer allocate(std::size_t n)
  {
    pointer p = nullptr;
    auto status = _wrapper.allocate(n*sizeof(value_type), reinterpret_cast<void**>(&p));
    if(status != 0) throw std::bad_alloc();
    return p;
  }

  void deallocate(pointer p, std::size_t n) noexcept
  {
    _wrapper.deallocate(p, n);
  }

  pointer allocate(std::size_t n, const_void_pointer)
  {
    return allocate(n);
  }

  template <class U, class ...Args>
  void construct(U* p, Args&& ...args)
  {
    ::new(p) U(std::forward<Args>(args)...);
  }

  template <class U>
  void destroy(U* p) noexcept
  {
    p->~U();
  }

  std::size_t max_size() const noexcept
  {
    return std::numeric_limits<size_type>::max();
  }

  MM_plugin_cxx_allocator select_on_container_copy_construction() const
  {
    return *this;
  }

  using propagate_on_container_copy_assignment = std::false_type;
  using propagate_on_container_move_assignment = std::false_type;
  using propagate_on_container_swap            = std::false_type;
  using is_always_equal                        = std::is_empty<MM_plugin_cxx_allocator>;
  
  MM_plugin_wrapper& _wrapper;
};

template <class T, class U>
bool
operator==(MM_plugin_cxx_allocator<T> const&, MM_plugin_cxx_allocator<U> const&) noexcept
{
  return true;
}

template <class T, class U>
bool
operator!=(MM_plugin_cxx_allocator<T> const& x, MM_plugin_cxx_allocator<U> const& y) noexcept
{
  return !(x == y);
}


#undef LOAD_SYMBOL
#endif


#pragma GCC diagnostic pop

#endif // __MM_PLUGIN_ITF_H__
