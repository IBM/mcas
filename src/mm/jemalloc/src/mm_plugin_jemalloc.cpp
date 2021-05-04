#include <stdio.h>
#include <assert.h>
#include <common/errors.h>
#include <common/utils.h>
#include <jemalloc/jemalloc.h>
#include <stdarg.h>
#include <fstream>
#include <vector>
#include "avl_malloc.h"
#include "logging.h"

#include "../../mm_plugin_itf.h"

#define LOG_TO_FILE
#undef  DEBUG_EXTENTS
#define DEBUG_ALLOCS

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
#pragma GCC diagnostic ignored "-Wunused-parameter"

#define CAST_HEAP(X) (reinterpret_cast<Heap*>(X))
// forward decls
static void hook_extents(unsigned arena_id);
  
static unsigned next_arena_id = 1;
static constexpr unsigned MAX_HEAPS = 255;

class Heap;
Heap * g_heap_map[MAX_HEAPS] = {0};

/** 
 * Manages a heap instance
 * 
 */
#define USE_AVL

class Heap
{
public:
  Heap(unsigned arena_id) : _arena_id(arena_id)
#ifdef LOG_TO_FILE
                          , _ofs("mm.log")
#endif
  {
  }

  void add_region(void * base, size_t len) {

    if(!check_aligned(base, KiB(4))) {
      PPERR("adding non 4KiB-aligned region");
      return;
    }

    bool needs_hooking = (_managed_size == 0);

#ifdef USE_AVL
    /* add memory to allocator */
    if(_managed_size == 0) {
      _avl_allocator = std::make_unique<core::AVL_range_allocator>(reinterpret_cast<addr_t>(base), len);
    }
    else {
      _avl_allocator->add_new_region(reinterpret_cast<addr_t>(base), len);
    }
#else
    _log_ptr = _log_base = reinterpret_cast<byte*>(base);
    _log_end = _log_base + len;
#endif
    /* once-only hook in extent allocators */
    if(needs_hooking) {
      hook_extents(_arena_id);
      _x_flags = MALLOCX_ARENA(_arena_id) | MALLOCX_TCACHE_NONE;
    }     
  }    

  void * allocate(const size_t size, const size_t alignment) {

    _managed_size += size;
    PPLOG("managed size: %lu MiB", REDUCE_MB(_managed_size));
#ifdef USE_AVL
    auto extent = _avl_allocator->alloc(size, alignment);
    if(!extent) return nullptr; /* ran out of memory */   
    return extent->paddr();
#else
    if(_log_ptr + size > _log_end) return nullptr;
    auto extent = _log_ptr;
    _log_ptr += size;
    return extent;
#endif
  }

  void deallocate(void * ptr, size_t size) {

#ifdef USE_AVL
    _avl_allocator->free(reinterpret_cast<addr_t>(ptr));
#endif
  }

  void set_arena(const unsigned id) { _arena_id = id; }

  unsigned x_flags() const { return _x_flags; }

  void log_allocation(const void * p, const size_t size, const size_t alignment) {
    _ofs << "alloc," << std::hex << p << "," << std::dec << size << "," << alignment << std::endl;
  }

  void log_free(const void * p) {
    _ofs << "free," << std::hex << p << "," << std::dec << 0 << "," << 0 << std::endl;
  }

  
private:
  size_t   _managed_size = 0;
  unsigned _x_flags = 0;
  unsigned _arena_id;
  std::ofstream _ofs;

#ifdef USE_AVL
  /* use an AVL tree to manage the extents */
  std::unique_ptr<core::AVL_range_allocator> _avl_allocator;
#else
  byte * _log_base;
  byte * _log_end;
  byte * _log_ptr;
#endif
};


void * custom_extent_alloc(extent_hooks_t *extent_hooks,
                           void *new_addr,
                           size_t size,
                           size_t alignment,
                           bool *zero,
                           bool *commit,
                           unsigned arena_id)
{
#ifdef DEBUG_EXTENTS
  PPLOG("%s: new_addr=%p size=%lu alignment=%lu arena=%u",
        __func__, new_addr, size, alignment, arena_id);
#endif
  
  if(g_heap_map[arena_id] == nullptr) return nullptr;
  
  if(new_addr != nullptr) {
    PPERR("%s does not know how to handle predefined newaddr",__func__);
  }
  assert(MiB(2) % alignment == 0);
  assert(arena_id < MAX_HEAPS);
  assert(g_heap_map[arena_id]);

  void * p = g_heap_map[arena_id]->allocate(size, alignment);

#ifdef DEBUG_EXTENTS  
  PPLOG("extent at %p allocated", p);
#endif
  return p;
}

bool custom_extent_dalloc(extent_hooks_t *extent_hooks, void *addr, size_t size, bool committed, unsigned arena_id)
{
  g_heap_map[arena_id]->deallocate(addr, size);
  return true;
}

void custom_extent_destroy(extent_hooks_t *extent_hooks,
                           void *addr,
                           size_t size,
                           bool committed,
                           unsigned arena_id)
{
  asm("int3");
  PPLOG("%s",__func__);
}

bool custom_extent_commit(extent_hooks_t *extent_hooks,
                          void *addr,
                          size_t size,
                          size_t offset,
                          size_t length,
                          unsigned arena_id)
{
  PPLOG("%s",__func__);
  return false;
}

bool custom_extent_decommit(extent_hooks_t *extent_hooks,
                            void *addr,
                            size_t size,
                            size_t offset,
                            size_t length,
                            unsigned arena_id)
{
  PPLOG("%s",__func__);
  return false;
}

bool custom_extent_purge(extent_hooks_t *extent_hooks,
                         void *addr,
                         size_t size,
                         size_t offset,
                         size_t length,
                         unsigned arena_id)
{
  return false;
}

bool custom_extent_split(extent_hooks_t *extent_hooks,
                         void *addr,
                         size_t size,
                         size_t size_a,
                         size_t size_b,
                         bool committed,
                         unsigned arena_id)
{
  return true; /* leave as whole */
}

bool custom_extent_merge(extent_hooks_t *extent_hooks,
                         void *addr_a,
                         size_t size_a,
                         void *addr_b,
                         size_t size_b,
                         bool committed,
                         unsigned arena_id)
{
  PPLOG("%s",__func__);
  return true;  /* leave split */
}

extent_hooks_t custom_extent_hooks =
  {
   custom_extent_alloc,
   custom_extent_dalloc,
   custom_extent_destroy,
   custom_extent_commit,
   custom_extent_decommit,
   NULL, /*custom_extent_purge_lazy*/
   NULL, /*custom_extent_purge_forced */
   custom_extent_split,
   custom_extent_merge,
  };                            

status_t mm_plugin_init()
{
  PPLOG("init");

  unsigned nbins, i, narenas;
  size_t mib[4];
  size_t len, miblen;

  len = sizeof(nbins);
  jel_mallctl("arenas.nbins", &nbins, &len, NULL, 0);
  PPLOG("n-bins: %u", nbins);
  
  len = sizeof(narenas);
  jel_mallctl("opt.narenas", &narenas, &len, NULL, 0);
  PPLOG("n-arenas: %u", narenas);

  miblen = 4;
  jel_mallctlnametomib("arenas.bin.0.size", mib, &miblen);
  for (i = 0; i < nbins; i++) {
    size_t bin_size;

    mib[2] = i;
    len = sizeof(bin_size);
    jel_mallctlbymib(mib, miblen, &bin_size, &len, NULL, 0);
    PPLOG("bin size=%lu", bin_size);
  }

  /* disable tcache */
  {
    bool off = 0;
    size_t off_size = sizeof(off);
    if(jel_mallctl("thread.tcache.enabled",(void*)&off,&off_size,NULL,0)) {
      PPERR("error: disabling tcache");
      return E_FAIL;      
    }
    PPLOG("disabled tcache.");
  }
  
  return S_OK;
}

static void hook_extents(unsigned arena_id)
{
  size_t hooks_mib[3];
  size_t hooks_miblen;
  extent_hooks_t *new_hooks, *old_hooks;
  size_t old_size, new_size;
 
  /* unsigned arena_id = 0; */
  /* size_t arena_id_size = sizeof(arena_id); */

  hooks_miblen = sizeof(hooks_mib)/sizeof(size_t);

  /* get hold of mib entry */
  if(jel_mallctlnametomib("arena.0.extent_hooks", hooks_mib, &hooks_miblen)) {
    PPERR("getting MIB entry for arena.0.extent_hooks");
    return;
  }

  hooks_mib[1] = arena_id;
  old_size = sizeof(extent_hooks_t *);
  new_hooks = &custom_extent_hooks;
  new_size = sizeof(extent_hooks_t *);
    
  if(jel_mallctlbymib(hooks_mib, hooks_miblen, (void *)&old_hooks, &old_size, (void *)&new_hooks, new_size)) {
    PERR("new hook attach failed");
    return;
  }    
}


status_t mm_plugin_create(const char * params, mm_plugin_heap_t * out_heap)
{
  PPLOG("mm_plugin_create (%s)", params);

  auto arena_id = next_arena_id;
  auto new_heap = new Heap(next_arena_id);
  g_heap_map[arena_id] = new_heap;
  *out_heap = reinterpret_cast<mm_plugin_heap_t>(new_heap);

  next_arena_id++;
  return S_OK;
}

status_t mm_plugin_destroy(mm_plugin_heap_t heap)
{
  return E_NOT_IMPL;
}

status_t mm_plugin_add_managed_region(mm_plugin_heap_t heap,
                                      void * region_base,
                                      size_t region_size)
{
  PPNOTICE("%s base=%p size=%lu",__func__, region_base, region_size);
  auto h = reinterpret_cast<Heap*>(heap);
  h->add_region(region_base, region_size);
  return S_OK;
}

status_t mm_plugin_query_managed_region(mm_plugin_heap_t heap,
                                        unsigned region_id,
                                        void** out_region_base,
                                        size_t* out_region_size)
{
  PPLOG("%s",__func__);
  return E_NOT_IMPL;
}

status_t mm_plugin_register_callback_request_memory(mm_plugin_heap_t heap,
                                                    request_memory_callback_t callback,
                                                    void * param)
{
  PPLOG("%s",__func__);
  return E_NOT_IMPL;
}

status_t mm_plugin_allocate(mm_plugin_heap_t heap, size_t n, void ** out_ptr)
{
#ifdef DEBUG_ALLOCS
  PPLOG("%s (%lu) x_flags=%x",__func__, n, CAST_HEAP(heap)->x_flags());
#endif
  void * ptr = jel_mallocx(n, CAST_HEAP(heap)->x_flags());
  if(ptr == nullptr) PPERR("out of memory");
  *out_ptr = ptr;

#ifdef LOG_TO_FILE
  CAST_HEAP(heap)->log_allocation(ptr, n, 0);
#endif
  
  return S_OK;
}

status_t mm_plugin_aligned_allocate(mm_plugin_heap_t heap, size_t n, size_t alignment, void ** out_ptr)
{
#ifdef DEBUG_ALLOCS
  PPLOG("%s (%lu,%lu) x_flags=%x",__func__, n, alignment, CAST_HEAP(heap)->x_flags());
#endif
  
  assert(is_power_of_two(alignment));
  void * ptr = jel_mallocx(n, CAST_HEAP(heap)->x_flags() | MALLOCX_ALIGN(alignment));
  if(ptr == nullptr) PPERR("out of memory");
  assert(ptr);
  assert(check_aligned(ptr, alignment));
  *out_ptr = ptr;

#ifdef LOG_TO_FILE
  CAST_HEAP(heap)->log_allocation(ptr, n, alignment);
#endif

  return S_OK;
}

status_t mm_plugin_aligned_allocate_offset(mm_plugin_heap_t heap, size_t n, size_t alignment, size_t offset, void ** out_ptr)
{
  asm("int3");
  return E_NOT_IMPL;
}

status_t mm_plugin_deallocate(mm_plugin_heap_t heap, void * ptr, size_t n)
{
#ifdef DEBUG_ALLOCS
  PPLOG("%s (%p, %lu) x_flags=%x",__func__, ptr, n, CAST_HEAP(heap)->x_flags());
#endif
  
  jel_sdallocx(ptr, n, CAST_HEAP(heap)->x_flags());

#ifdef LOG_TO_FILE
  CAST_HEAP(heap)->log_free(ptr);
#endif
  
  return S_OK;
}

status_t mm_plugin_deallocate_without_size(mm_plugin_heap_t heap, void * ptr)
{
#ifdef DEBUG_ALLOCS
  PPLOG("%s (%p) x_flags=%x",__func__, ptr, CAST_HEAP(heap)->x_flags());
#endif

  if(ptr == nullptr) return S_OK;
  //  jel_free(ptr);
  jel_dallocx(ptr, CAST_HEAP(heap)->x_flags());

#ifdef LOG_TO_FILE
  CAST_HEAP(heap)->log_free(ptr);
#endif
  
  return S_OK;
}

status_t mm_plugin_callocate(mm_plugin_heap_t heap, size_t n, void ** out_ptr)
{
#ifdef DEBUG_ALLOCS
  PPLOG("%s (%lu) x_flags=%x",__func__, n, CAST_HEAP(heap)->x_flags());
#endif
  void * ptr = jel_mallocx(n, CAST_HEAP(heap)->x_flags() | MALLOCX_ZERO);
  if(ptr == nullptr) {
    PPERR("callocate: out of memory");
    return E_NO_MEM;
  }
  assert(ptr);
  *out_ptr = ptr;
  return S_OK;
}

status_t mm_plugin_reallocate(mm_plugin_heap_t heap, void * ptr, size_t n, void ** out_ptr)
{
#ifdef DEBUG_ALLOCS
  PPLOG("%s (%p, %lu) x_flags=%x",__func__, ptr, n, CAST_HEAP(heap)->x_flags());
#endif
  
  if(ptr == nullptr) {
    /* if pointer is null, then we just do a new allocation */
    *out_ptr = jel_mallocx(n, CAST_HEAP(heap)->x_flags());
  }
  else {
    *out_ptr = jel_rallocx(ptr, n, CAST_HEAP(heap)->x_flags());
  }
  return S_OK;
}

status_t mm_plugin_usable_size(mm_plugin_heap_t heap, void * ptr, size_t * out_size)
{
  *out_size = jel_sallocx(ptr, CAST_HEAP(heap)->x_flags());
  //  *out_size = jel_malloc_usable_size(ptr);
  return S_OK;
}

void mm_plugin_debug(mm_plugin_heap_t heap)
{
  jel_malloc_stats_print(nullptr,nullptr,nullptr);
}


#pragma GCC diagnostic pop
