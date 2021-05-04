/*
   Copyright [2017-2019] [IBM Corporation]
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



/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __COMMON_LAZY_REGION_H__
#define __COMMON_LAZY_REGION_H__

#include <signal.h>
#include <sys/mman.h>
#include <list>
#include <mutex>

#include <common/assert.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/types.h>
#include <common/utils.h>

#include "safe_print.h"

namespace core
{
namespace slab
{
// forward decls
//
class Lazily_extending_region;

/**
 * Interval map - used to map addr to instance of Lazily_extending_region class
 *
 */
struct __map_entry {
  addr_t start;
  addr_t end;
  Lazily_extending_region *inst;
};

static std::list<__map_entry> _interval_map;  // we should use a tree
static std::mutex _interval_map_lock;

/**
 * Add interval to map; used to look up class instance from SIGSEGV handler
 *
 * @param start Start address
 * @param end End address
 * @param inst Instance pointer
 */
static void __add_interval(addr_t start, addr_t end,
                           Lazily_extending_region *inst) {
  std::lock_guard<std::mutex> lock(_interval_map_lock);
  for (auto i = _interval_map.begin(); i != _interval_map.end(); i++) {
    if (start < i->start) {
      _interval_map.insert(i, {start, end, inst});
      break;
    }
  }
  _interval_map.push_back({start, end, inst});
  // PDBG("added interval (%lx-%lx)", start, end);
}

/**
 * Lookup instance for a given address
 *
 * @param addr
 *
 * @return
 */
static Lazily_extending_region *__lookup_inst(addr_t addr) {
  assert(addr > 0);
  std::lock_guard<std::mutex> lock(_interval_map_lock);
  for (auto i : _interval_map) {
    if (addr >= i.start && addr <= i.end) return i.inst;
  }
  SAFE_PRINT("address (%p) not found in interval map", reinterpret_cast<void *>(addr));
  assert(0);
  return NULL;  // not found
}

class Sigaction
{
  struct sigaction _old;
public:
  Sigaction(void (*handler)(int, siginfo_t *, void *))
    : _old()
  {
    // set up SIGSEGV handler
    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = handler;

    if (sigaction(SIGSEGV, &sa, &_old) == -1)
      throw Constructor_exception("unable to set SIGSEGV handler");
  }
  ~Sigaction()
  {
    ::sigaction(SIGSEGV, &_old, NULL);
  }
};

/**
 * Class to manage a lazily extending region of memory
 *
 */

class Lazily_extending_region {
  Lazily_extending_region(const Lazily_extending_region &) = delete;
  Lazily_extending_region& operator=(const Lazily_extending_region &) = delete;
 private:
  static constexpr bool option_DEBUG = false;
  static addr_t addr_hint;

 public:
  /**
   * Constructor
   *
   * @param size Maximum memory allocation
   */
  Lazily_extending_region(size_t size) :
    _sigstate(&Lazily_extending_region::SIGSEGV_handler)
    , _max_size((assert(size % PAGE_SIZE == 0), size))
    , _ptr(::mmap(reinterpret_cast<void *>(addr_hint), _max_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0) )
    , _mapped_size(0) {

    addr_hint += 0x10000000ULL;

    if (option_DEBUG) {
      SAFE_PRINT("mmap allocated region (%p)", _ptr);
    }

    auto ptr_addr = reinterpret_cast<addr_t>(_ptr);
    __add_interval(ptr_addr, ptr_addr + _max_size, this);
  }

  /**
   * Destructor
   *
   */
  ~Lazily_extending_region() {
    assert(_ptr);
    int rc = ::munmap(_ptr, _max_size);
    if (rc)
      SAFE_PRINT("::munmap failed in Lazilty_extending_region dtor");
    assert(rc == 0);
  }

  /**
   * Get base address of allocated region
   *
   *
   * @return Pointer to allocated region
   */
  addr_t base() { return reinterpret_cast<addr_t>(_ptr); }

  /**
   * Get pointer to allocated region
   *
   *
   * @return
   */
  void *ptr() const { return _ptr; }

  /**
   * Return amount of memory mapped
   *
   *
   * @return Size of used/memory in bytes
   */
  size_t mapped_size() {
    __sync_synchronize();
    return _mapped_size;
  }

  /**
   * Return number of mapped pages
   *
   *
   * @return Number of mapped pages
   */
  size_t mapped_pages() { return mapped_size() / PAGE_SIZE; }

  /**
   * Maximum size of the expanding region
   *
   *
   * @return
   */
  size_t max_size() {
    __sync_synchronize();
    return _max_size;
  }

 private:
  Sigaction _sigstate;
  size_t _max_size;    /**< maximum size of the expanding slab */
  void *_ptr;
  size_t _mapped_size; /**< size mapped to physical memory */

 private:
  /**
   * Increment # of map pages
   *
   */
  void increment_mapping() {
    _mapped_size += PAGE_SIZE;
    assert(_mapped_size <= _max_size);
    if (option_DEBUG) {
      PDBG("#pages mapped:%lu", _mapped_size / PAGE_SIZE);
    }
  }

  static addr_t round_down_page(addr_t a) {
    /* round up to 4K page */
    if ((a & addr_t(0xfff)) == 0)
      return a;
    else
      return a & ~(addr_t(0xfff));
  }

  /**
   * SEGV signal handler
   *
   * @param sig
   * @param si
   * @param context
   */
  static void SIGSEGV_handler(int /*sig*/, siginfo_t *si, void * /*context*/) {
    // TODO throw the stock handler on SEGV outside of
    // our memory

    //        static addr_t _last_fault_addr = 0;

    void *faulting_page = reinterpret_cast<void *>(round_down_page(reinterpret_cast<addr_t>(si->si_addr)));
    if (option_DEBUG) {
      PDBG("fault addr:%p", faulting_page);
    }
    assert(faulting_page);

    // if(_last_fault_addr) // check for sequential faults
    //   assert((_last_fault_addr + PAGE_SIZE) == ((addr_t) faulting_page));

    // _last_fault_addr = ((addr_t) faulting_page);

    Lazily_extending_region *inst = __lookup_inst(reinterpret_cast<addr_t>(faulting_page));
    assert(inst);
    inst->increment_mapping();

    ::mprotect(faulting_page, PAGE_SIZE, PROT_READ | PROT_WRITE);
  }
};

}  // namespace slab
}  // namespace core

#endif
