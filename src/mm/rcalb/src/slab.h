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

#ifndef __COMMON_SLAB_H__
#define __COMMON_SLAB_H__

#include "safe_print.h"
#include "lazy_region.h"

#include <common/memory.h>
#include <common/chksum.h>
#include <algorithm>
#include <cstring>
#include <functional>
#include <iomanip>
#include <set>
#include <sstream>
#include <vector>

#include "mm_wrapper.h"

namespace real
{
extern malloc_function_t        malloc;
extern free_function_t          free;
}

namespace core
{
namespace slab
{
/**
 * Simple slab allocator using POSIX runtime new/delete
 *
 */
template <typename T>
class CRuntime : public common::Base_slab_allocator {
 public:
  CRuntime() {}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
  CRuntime(size_t slots, bool exact = false) {}
#pragma GCC diagnostic pop
  void* alloc() { return real::malloc(sizeof(T)); }
  size_t free(void* ptr) {
    assert(ptr);
    real::free(ptr);
    return sizeof(T);
  }
  bool is_reconstructed() { return false; }
};

/**
 * In-place header for each element of the slab allocator
 *
 */
struct __BasicElementHeader {
  union u {
    struct s {
      bool used : 1;
      unsigned resv : 7;
    } use;
    uint8_t flags;
  };
} __attribute__((packed));

template <typename V>
class __BasicElement { /**< order and packing is important */
 public:
  __BasicElementHeader hdr; /**< common header */
  V val;                    /**< value itself */
} __attribute__((packed));

/**
 * Serializable slab allocator which can be re-initialized from a block
 * of memory (persistent or not)
 *
 */
template <typename T = void*,
          template <typename U> class Element = slab::__BasicElement>
class Allocator : public common::Base_slab_allocator {
 private:
  static const bool option_DEBUG = true;    /**< toggle to activate debugging */
  static const unsigned MIN_ELEMENTS = 128; /**< sanity bounds */

  // 64 byte header
  struct Header {
    char magic[8];
    size_t region_size;
    uint64_t slots;
    uint64_t max_slots;
    uint64_t slot_size;
    char label[24];
  } __attribute__((packed));

  void* _region;
  unsigned _page_shift;
  std::vector<Element<T>*> _free_slots; /**< this is not written out; its
                                           hdr.used as in-memory index */
  Element<T>* _slot_array; /**< data that can be implicitly serialized */
  Header* _header;
  bool _reconstructed;

  Allocator(const Allocator &) = delete;
  Allocator &operator=(const Allocator &) = delete;
 public:
  /**
   * Constructor.  Attempts to recreate from existing region (using the magic
   * number as a hint) or initializes new allocator.
   *
   * @param region Pointer to region (owner to clean up).
   * @param region_size Size of region in bytes
   * @param label Label to save in header
   * @param as_new Force new instantiation if set to true
   * @param page_shift Shift for IO blocks
   *
   */
  Allocator(void* region, size_t region_size, std::string label, bool as_new,
            unsigned page_shift = 12)
      : _region(region),
        _page_shift(page_shift),
        _slot_array(reinterpret_cast<Element<T>*> (reinterpret_cast<addr_t>(_region) + sizeof(Header))),
        _header(static_cast<Header*>(region))  // header is start of region
  {
    static_assert(sizeof(Header) == 64, "Header should be 64 bytes");
    assert(_header);

    _header->magic[6] = '\0';

    // recover existing state
    if ((strncmp(_header->magic, "_SLAB_", 6) == 0) && !as_new) {
      /* already initialized */
      SAFE_PRINT("Reconstructed slab: region_size=%ld, slots=%ld, max_slots=%ld, "
           "slot_size=%ld, label=%s",
           _header->region_size, _header->slots, _header->max_slots,
           _header->slot_size, _header->label);

      // rebuild free slot list
      for (size_t i = 0; i < _header->slots; i++) {
        Element<T>* e = &_slot_array[i];
        if (!e->hdr.use.used) _free_slots.push_back(e);
      }
      _reconstructed = true;
    }
    else {
      __builtin_memset(_header, 0, sizeof(Header));
      __builtin_memset(_slot_array, 0,
                       sizeof(Element<T>) * _header->max_slots);

      strncpy(_header->magic, "_SLAB_", 6);
      _header->magic[6] = '\0';
      strncpy(_header->label, label.c_str(), sizeof(_header->label));

      _header->region_size = region_size;
      _header->max_slots = (region_size - sizeof(Header)) / sizeof(Element<T>);
      assert(_header->max_slots > 0);

      if (_header->max_slots < MIN_ELEMENTS) /* sanity check */
        throw new Constructor_exception("too small a region");

      _header->slot_size = sizeof(Element<T>);
      _header->slots = 0;

      SAFE_PRINT("New slab: region_size=%ld, slots=%ld, max_slots=%ld, "
                 "slot_size=%ld, label=%s",
                 _header->region_size, _header->slots, _header->max_slots,
                 _header->slot_size, _header->label);

      _reconstructed = false;
    }
  }

  /**
   * Destructor
   *
   *
   */
  virtual ~Allocator() {}

  /**
   * Determine how much memory to allocate for N slots
   *
   * @param num_slots Number of slots
   *
   * @return Required memory size in bytes
   */
  static size_t determine_size(size_t num_slots) {
    assert(num_slots > 0);
    return (sizeof(Header) + (sizeof(Element<T>) * (num_slots + 1)));
  }

  /**
   * Determine how much memory is needed without type T
   *
   * @param num_slots
   * @param slot_size
   *
   * @return
   */
  static size_t determine_size(size_t num_slots, size_t slot_size) {
    assert(num_slots > 0);
    return (sizeof(Header) +
            ((sizeof(__BasicElementHeader) + slot_size) * (num_slots + 1)));
  }

  /**
   * Get count of number of slots
   *
   *
   * @return Number of slots of type T
   */
  size_t num_slots() const { return _header->max_slots; }

  /**
   * Get number of used slots
   *
   *
   * @return Number of used slots
   */
  size_t used_slots() const {
    size_t count = 0;
    for (size_t i = 0; i < _header->slots; i++) {
      Element<T>* slot = &_slot_array[i];
      if (slot->hdr.use.used) {
        count++;
      }
    }
    return count;
  }

  /**
   * Determine if the slab was reconstructed or not
   *
   *
   * @return True if reconstructed; false if new init
   */
  bool is_reconstructed() { return _reconstructed; }

  /**
   * Allocate an element from the slab
   *
   *
   * @return Pointer to newly allocated element. NULL when memory runs out.
   */
  void* alloc() {
    assert(_header);
    assert(_header->max_slots > 0);

    if (_free_slots.size() > 0) {
      Element<T>* slot = _free_slots.back();
      assert(slot);
      assert(slot->hdr.use.used == false);
      slot->hdr.use.used = true;
      _free_slots.pop_back();
      return &slot->val;
    }
    else {
      if (option_DEBUG)
        SAFE_PRINT("adding new slot (array_len=%ld)...", _header->slots);
      
      if (_header->slots >= _header->max_slots) {
        throw API_exception("Slab allocator (%s) run out of memory!",
                            _header->label);
      }

      Element<T>* slot = &_slot_array[_header->slots];
      assert(slot);
      std::memset(slot, 0, sizeof(Element<T>));  // could be removed.
      slot->hdr.use.used = true;
      _header->slots++;

      return &slot->val;
    }
    assert(0);
  }

  /**
   * Free a previously allocated element
   *
   * @param elem Pointer to allocated element
   */
  size_t free(void* const pval) {
    const auto pval_addr = reinterpret_cast<addr_t>(pval);
    const auto region_addr = reinterpret_cast<addr_t>(_region);
    // bounds check pointer
    if ((pval_addr < region_addr) ||
        (pval_addr > (region_addr + _header->region_size))) {
      PWRN("free on invalid pointer (%p)", pval);
      return static_cast<size_t>(-1);
    }

    if (option_DEBUG) PDBG("freeing slab element: %p", pval);

    Element<T>* slot = reinterpret_cast<Element<T>*>(pval_addr - sizeof(slot->hdr));
    slot->hdr.use.used = false;

    _free_slots.push_back(slot);
    return sizeof(Element<T>);
  }

  /**
   * Apply functor to slots (up to last used slot)
   *
   * @param functor void f(pointer to slot, size, used flag)
   */
  void apply(std::function<void(void*, size_t, bool)> functor) {
    for (size_t i = 0; i < _header->slots; i++) {
      Element<T>* slot = &_slot_array[i];
      functor(&slot->val, _header->slot_size, slot->hdr.use.used);
    }
  }

  /**
   * Return number of free slots in currently allocated memory
   *
   *
   * @return Number of free slots
   */
  size_t free_slots() const { return _free_slots.size(); }

  void* get_first_element() { return &_slot_array[0].val; }

 public:
  /**
   * Get the memory limits for used pages
   *
   * @param bottom
   * @param top
   */
  void get_memory_limits(addr_t& bottom, addr_t& top) {
    bottom = reinterpret_cast<addr_t>(_region);
    top = reinterpret_cast<addr_t>(&_slot_array[_header->slots]);
  }

  /**
   * Dump the status of the slab
   *
   */
  void dump_info() {
    addr_t base = reinterpret_cast<addr_t>(_region);
    addr_t top = base + _header->region_size;

    SAFE_PRINT("%s", "---------------------------------------------------");
    SAFE_PRINT("HEADER: magic         (%s) ", _header->magic);
    SAFE_PRINT("      : slots         (%ld)", _header->slots);
    SAFE_PRINT("      : max slots     (%ld)", _header->max_slots);
    SAFE_PRINT("      : slot size     (%ld)", _header->slot_size);
    SAFE_PRINT("      : label         (%s)", _header->label);
    SAFE_PRINT("      : memory range  (%p-%p) %ld KB", reinterpret_cast<void*>(_header), reinterpret_cast<void*>(top),
         REDUCE_KB((top - base)));
    SAFE_PRINT("      : chksum        (%x)",
         common::chksum32(_header, _header->region_size));
    SAFE_PRINT("%s", "---------------------------------------------------");

#ifdef SHOW_ENTRIES
    for (unsigned i = 0; i < _header->slots; i++) {
      if (i == 100) {
        SAFE_PRINT("...");
        break;  // short circuit
      }

      Element<T>* slot = &_slot_array[i];

      constexpr size_t val_size = sizeof(T);
      std::stringstream sstr;
      byte* v = (byte*) &slot->val;
      for (size_t i = 0; i < val_size; i++) {
        sstr << std::hex << std::setw(2) << std::setfill('0') << int(*v) << " ";
        v++;
        if (i > 10) break;
      }

      SAFE_PRINT("\t[%u]: %s %p : %s", i, slot->hdr.use.used ? "USED" : "EMPTY",
           (void*) &slot->val, sstr.str().c_str());
    }
#endif

    SAFE_PRINT("%s", "---------------------------------------------------");
    SAFE_PRINT(" Volatile status: _free_slots.size = %ld", _free_slots.size());
    unsigned i = 0;
    for (auto fs : _free_slots) {
      SAFE_PRINT("FREE slot entry: %p", fs);
      i++;
      if (i == 100) {
        SAFE_PRINT("%s", "...");
        break;
      }
    }
    SAFE_PRINT("%s", "---------------------------------------------------");
  }

  /**
   * For debugging purposes.  Sort the free slot vector.
   *
   */
  void __dbg_sort_free_slots() {
    sort(_free_slots.begin(), _free_slots.end(),
         [](Element<T>* i, Element<T>* j) -> bool {
           return (reinterpret_cast<addr_t>(i) < reinterpret_cast<addr_t>(j));
         });
  }

  /**
   * For debugging purposes. Get reference to slot vector
   *
   *
   * @return
   */
  std::vector<Element<T>*>& __dbg_slots() { return _free_slots; }
};

}  // namespace slab
}  // namespace core

#endif
