#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cstdlib>
#include <dlfcn.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <common/logging.h>
#include <common/utils.h>
#include <common/cycles.h>
#include <common/exceptions.h>
#include <common/stack_trace.h>
#include <sys/mman.h>

#include "mm_wrapper.h"
#include "mm_plugin_itf.h"
#include "safe_print.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"

#define LOG_TO_FILE

#define LOAD_SYMBOL(X) __mm_funcs.X = reinterpret_cast<typeof(__mm_funcs.X)>(dlsym(__mm_plugin_module, # X)); assert(__mm_funcs.X)

static void __get_os_functions(void);

static const char * build_log_filename()
{
  static char tmp[1024];
  sprintf(tmp, "%s-%d-mm.log", basename(::getenv("PLUGIN")), ::getpid());
  return tmp;
}

static size_t size_to_allocate()
{
  auto env = ::getenv("SLAB_SIZE_GB");
  if(env) {
    auto gb = std::stoul(env);
    PINF("MM: (%d) using %lu GiB for slab", ::getpid(), gb);
    return GiB(gb);
  }
  else {
    return GiB(4);
  }
}

class Logger
{
public:
  Logger() : _ofs(build_log_filename()) {
  }

  void log(const char * type, const cpu_time_t time_cycles, const void * p, const size_t size = 0, const size_t alignment = 0) {
    char tmp[1024];
    /* timestamp, type, cycles, p, size, alignment */
    sprintf(tmp, "%lu,%s,%lu,%p,%lu,%lu\n", rdtsc(), type, time_cycles, p, size, alignment);
    _ofs << tmp;
  }

private:
  std::ofstream _ofs;
};

namespace globals
{
static void * slab_memory = nullptr; /*< memory which will be used as the slab for the allocator */
static bool intercept_active = false;
static uint64_t dl_module_mask = 0;

#ifdef LOG_TO_FILE
static Logger * log = nullptr;
#endif
}

static mm_plugin_function_table_t __mm_funcs;
static void * __mm_plugin_module;
static mm_plugin_heap_t __mm_heap;

/* real function implementations */
namespace real
{
malloc_function_t        malloc = nullptr;
free_function_t          free = nullptr;
aligned_alloc_function_t aligned_alloc = nullptr;
realloc_function_t       realloc = nullptr;
calloc_function_t        calloc = nullptr;
memalign_function_t      memalign = nullptr;
vfprintf_function_t      vfprintf = nullptr;
puts_function_t          puts = nullptr;
fputs_function_t         fputs = nullptr;

malloc_usable_size_function_t malloc_usable_size = nullptr;
}

static void __init_components(void)
{
  auto path = ::getenv("PLUGIN");
  
  __mm_plugin_module = dlopen(path, RTLD_NOW | RTLD_DEEPBIND);

  globals::dl_module_mask = reinterpret_cast<uint64_t>(__mm_plugin_module) & 0xFFFFFFFFFF000000;

  if(!__mm_plugin_module) {
    printf("Error: invalid PLUGIN (%s)\n", path);
    exit(0);
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

  __mm_funcs.mm_plugin_init();  
  __mm_funcs.mm_plugin_create(nullptr, /* config */
                              nullptr, /* root pointer */
                              &__mm_heap);

  /* give some memory */
  size_t slab_size = size_to_allocate();
  globals::slab_memory = mmap(reinterpret_cast<void*>(0xAA00000000),
                              slab_size,
                              PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                              -1, 0);

  __mm_funcs.mm_plugin_add_managed_region(__mm_heap,
                                          globals::slab_memory,
                                          slab_size);

  globals::log = new Logger();
  
  globals::intercept_active = true;
}

static void __attribute__((destructor)) __mm_wrapper_dtor()
{
}

/** 
 * Collect the "original" OS implementations, so they can be used by
 * the memory manager plugin itself.
 * 
 */
static void __get_os_functions()
{
  real::malloc = reinterpret_cast<malloc_function_t>(dlsym(RTLD_NEXT, "malloc"));
  assert(real::malloc);

  real::calloc = reinterpret_cast<calloc_function_t>(dlsym(RTLD_NEXT, "calloc"));
  assert(real::calloc);
  
  real::free = reinterpret_cast<free_function_t>(dlsym(RTLD_NEXT, "free"));
  assert(real::free);
  
  real::aligned_alloc = reinterpret_cast<aligned_alloc_function_t>(dlsym(RTLD_NEXT, "aligned_alloc"));
  assert(real::aligned_alloc);

  real::realloc = reinterpret_cast<realloc_function_t>(dlsym(RTLD_NEXT, "realloc"));
  assert(real::realloc);

  real::memalign = reinterpret_cast<memalign_function_t>(dlsym(RTLD_NEXT, "memalign"));
  assert(real::memalign);

  real::vfprintf = reinterpret_cast<vfprintf_function_t>(dlsym(RTLD_NEXT, "vfprintf"));
  assert(real::vfprintf);

  real::puts = reinterpret_cast<puts_function_t>(dlsym(RTLD_NEXT, "puts"));
  assert(real::puts);

  real::fputs = reinterpret_cast<fputs_function_t>(dlsym(RTLD_NEXT, "fputs"));
  assert(real::fputs);

  real::malloc_usable_size = reinterpret_cast<malloc_usable_size_function_t>(dlsym(RTLD_NEXT, "malloc_usable_size"));
  assert(real::malloc_usable_size);
  

  /* initialize backend */
  __init_components();
}


#define EXPORT_C extern "C" __attribute__((visibility("default")))

EXPORT_C void   mm_free(void* p) noexcept;
EXPORT_C void*  mm_realloc(void* p, size_t newsize) noexcept;
EXPORT_C void*  mm_calloc(size_t count, size_t size) noexcept;
EXPORT_C void*  mm_malloc(size_t size) noexcept;
EXPORT_C size_t mm_usable_size(void* p) noexcept;
EXPORT_C void*  mm_valloc(size_t size) noexcept;
EXPORT_C void*  mm_pvalloc(size_t size) noexcept;
EXPORT_C void*  mm_reallocarray(void* p, size_t count, size_t size) noexcept;
EXPORT_C void*  mm_memalign(size_t alignment, size_t size) noexcept;
EXPORT_C int    mm_posix_memalign(void** p, size_t alignment, size_t size) noexcept;
EXPORT_C void*  mm_aligned_alloc(size_t alignment, size_t size) noexcept;


#include "alloc-override.c"

static addr_t sbrk_base = reinterpret_cast<addr_t>(sbrk(0));

EXPORT_C void mm_free(void* p) noexcept
{
  if(p == nullptr) return;

  if(!real::free) return; //__get_os_functions();

  /* hack to deal with something allocated by dl_main before we could hook
     calloc.  we have two options, one try to identify if the memory
     was part of the management region, or two, try to see if the memory
     came from the dl_main call.  this might be risky!
  */
  //  if((reinterpret_cast<uint64_t>(p) & 0xAA00000000) != 0xAA00000000) return;
  if((globals::dl_module_mask & reinterpret_cast<uint64_t>(p)) == globals::dl_module_mask) return;


  if(globals::intercept_active) {

#ifdef LOG_TO_FILE
    auto start = rdtsc();
#endif
    
    __mm_funcs.mm_plugin_deallocate_without_size(__mm_heap, p);

#ifdef LOG_TO_FILE
    globals::log->log("free", rdtsc()-start, p, 0, 0);
#endif
    
  }
  else {
    /* intercept is not yet active */
    //real::free(p); 
  }

}

EXPORT_C void* mm_realloc(void* p, size_t newsize) noexcept
{
  if(!real::realloc) __get_os_functions();

  if(globals::intercept_active) {
    void * new_ptr = nullptr;

#ifdef LOG_TO_FILE
    auto start = rdtsc();
#endif

    __mm_funcs.mm_plugin_reallocate(__mm_heap, p, newsize, &new_ptr);

#ifdef LOG_TO_FILE
    globals::log->log("realloc",rdtsc()-start, p, newsize, 0);
#endif
    
    return new_ptr;
  }
  else {
    //return real::realloc(p, newsize);
    return nullptr;
  }
}

EXPORT_C void* mm_calloc(size_t count, size_t size) noexcept
{
  /* dlopen / dlsym use calloc */
  if(!real::calloc) {
    /* use poor man's calloc, because dlopen uses calloc */
    void * p = sbrk(count*size);
    __builtin_memset(p, 0, count * size);
    return p;
  }
  void * p;
  if(globals::intercept_active) {
    p = nullptr;

#ifdef LOG_TO_FILE
    auto start = rdtsc();
#endif   

    __mm_funcs.mm_plugin_callocate(__mm_heap, count * size, &p);

#ifdef LOG_TO_FILE
    globals::log->log("calloc",rdtsc()-start, p, count * size, 0);
#endif
    
    assert(p);
  }
  else {
    p = real::calloc(count, size);
  }

  return p;
}

EXPORT_C void* mm_malloc(size_t size) noexcept
{
  if(!real::malloc) __get_os_functions();

  if(globals::intercept_active) {

    void * p = nullptr;
    
#ifdef LOG_TO_FILE
    auto start = rdtsc();
#endif
    
    __mm_funcs.mm_plugin_allocate(__mm_heap, size, &p);

#ifdef LOG_TO_FILE
    globals::log->log("malloc",rdtsc()-start,p, size, 0);
#endif

    return p;
  }
  else {
    return sbrk(size);
  }
}

EXPORT_C size_t mm_usable_size(void* p) noexcept
{
  if(p == nullptr) return 0;
  size_t us = 0;
  if(__mm_funcs.mm_plugin_usable_size(__mm_heap, p, &us) != S_OK) {
    PWRN("globals::mm->usable_size() failed");
    return 0;
  }
  return us;
}

/* valloc is Linux deprecated */
EXPORT_C void* mm_valloc(size_t size) noexcept
{
  void * p = nullptr;
  if(size == 0) return nullptr;

  __mm_funcs.mm_plugin_aligned_allocate(__mm_heap, size, sysconf(_SC_PAGESIZE), &p);
  return p;
}

EXPORT_C void* mm_pvalloc(size_t size) noexcept
{
  void * p = nullptr;
  if(size == 0) return nullptr;
  __mm_funcs.mm_plugin_aligned_allocate(__mm_heap, round_up(size, sysconf(_SC_PAGESIZE)), sysconf(_SC_PAGESIZE), &p);
  return p;
}

EXPORT_C void* mm_reallocarray(void* p, size_t count, size_t size) noexcept
{
  return mm_realloc(p, count * size);
}

EXPORT_C void* mm_memalign(size_t alignment, size_t size) noexcept
{
  if(size == 0) return nullptr;
  void * p = nullptr;
#ifdef LOG_TO_FILE
    auto start = rdtsc();
#endif

  __mm_funcs.mm_plugin_aligned_allocate(__mm_heap, size, alignment, &p);

#ifdef LOG_TO_FILE
  globals::log->log("memalign",rdtsc()-start, p, size, alignment);
#endif  
  return p;
}

EXPORT_C int mm_posix_memalign(void** p, size_t alignment, size_t size) noexcept
{
  if(p == nullptr) return EINVAL;
#ifdef LOG_TO_FILE
    auto start = rdtsc();
#endif
  
  __mm_funcs.mm_plugin_aligned_allocate(__mm_heap, size, alignment, p);

#ifdef LOG_TO_FILE
  globals::log->log("posix_memalign",rdtsc()-start, *p, size, alignment);
#endif  
  return 0;
}

EXPORT_C void* mm_aligned_alloc(size_t alignment, size_t size) noexcept
{
  void * p = nullptr;
#ifdef LOG_TO_FILE
    auto start = rdtsc();
#endif  

  __mm_funcs.mm_plugin_aligned_allocate(__mm_heap, size, alignment, &p);

#ifdef LOG_TO_FILE
  globals::log->log("aligned_alloc", rdtsc()-start, p, size, alignment);
#endif  
  
  return p;
}


#pragma GCC diagnostic pop
