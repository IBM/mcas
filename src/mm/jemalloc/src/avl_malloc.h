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

#ifndef __NUPM_AVL_MALLOC_H__
#define __NUPM_AVL_MALLOC_H__

#if !defined(__cplusplus)
#error("C++ header only")
#endif

#include "avl_tree.h"
#include "mr_traits.h"
#include "slab.h"
#include <common/stack.h>
#include <common/types.h>
#include <common/utils.h>
#include <iostream>
#include <vector>
#include <cstdlib> // exit

#define DEBUG_AVL_ALLOCATOR

namespace core
{
/**
 * Defines an allocated (free or not free) region of memory. This is
 * basically the node of the AVL tree.
 *
 */
class Memory_region : public core::AVL_node<Memory_region> {
  friend class AVL_range_allocator;

private:
  addr_t _addr;
  size_t _size;
  bool _free = true;
  Memory_region* _next = nullptr;
  Memory_region* _prev = nullptr;
  char _padding; // packed, so 8(vft) + (8+8+2) (core::AVL_node<Memory_region>) + (8+8+1+8+8+1) = 60 bytes
  //    size_t          _max_free = 0; // TODO: subtree pruning

public:
  /**
   * Constructor
   *
   * @param addr Start of memory address
   * @param size Size of memory region in bytes
   *
   */
  Memory_region(addr_t addr, size_t size) : _addr(addr), _size(size), _padding{}
  {
    (void)_padding; // unused
  }

  Memory_region(const Memory_region &) = delete;
  Memory_region &operator=(const Memory_region &) = delete;

  /**
   * Destructor
   *
   *
   * @return
   */
  virtual ~Memory_region() {}


  /**
   * Check that region contains address
   *
   * @param addr
   *
   * @return
   */
  bool contains(addr_t addr) const
  {
    return (addr >= _addr) && (addr < (_addr + _size));
  }

  /**
   * Dump information about the region
   *
   */
  void dump() {
    PLOG("node [%p]: addr=0x%lx size=%ld free=%s", common::p_fmt(this), _addr, _size,
         _free ? "yes" : "no");
  }

  /**
   * Get start address of this region
   *
   *
   * @return Start address
   */
  addr_t addr() { return _addr; }

  /**
   * Get start pointer of the region
   *
   *
   * @return Start address
   */
  void* paddr() { return reinterpret_cast<void*>(_addr); }


  /**
   * Get size of allocation
   *
   *
   * @return
   */
  size_t size() const { return _size; }

  /**
   * Get whether region is free or not
   *
   *
   * @return True if region is free
   */
  bool is_free() const { return _free; }

private:
  /**
   * Comparator need by base class
   *
   * @param n
   *
   * @return
   */
  bool higher(AVL_node<Memory_region>* n) {
    return (_addr > (static_cast<Memory_region*>(n))->_addr);
  }

  /**
   * Helper to get left hand subtree
   *
   *
   * @return
   */
  Memory_region* left() {
    return static_cast<Memory_region*>(&*subtree[core::LEFT]);
  }

  /**
   * Helper to get right hand subtree
   *
   *
   * @return
   */
  Memory_region* right() {
    return static_cast<Memory_region*>(&*subtree[core::RIGHT]);
  }

protected:
  /**
   * Find node with corresponding address
   *
   * @param node
   * @param addr
   *
   * @return
   */
  Memory_region* find_region(Memory_region* node, addr_t addr) {
    if (node == nullptr) return nullptr;

    std::vector<Memory_region*> stack;
    stack.push_back(node);

    while (!stack.empty()) {
      node = stack.back();
      stack.pop_back();

      if (node->_addr == addr) {
        return node;
      }
      else {
        Memory_region* l = node->left();
        if (l) stack.push_back(l);

        Memory_region* r = node->right();
        if (r) stack.push_back(r);
      }
    }
    
    return nullptr;
  }

  /**
   * Find a free region that can contain a request block size. This
   * currently does a top-down, left-to-right, first-fit traversal. This
   * might not be the best traversal but it will do for now.
   *
   * @param node Memory region
   * @param size Size in bytes
   * @param alignment Alignment in bytes
   *
   * @return
   */
  Memory_region* find_free_region(Memory_region* node,
                                  size_t size,
                                  size_t alignment = 0)
  {
    if (node == nullptr) return nullptr;

    common::Fixed_stack<Memory_region*> stack;
    stack.push(node);

    while (!stack.empty()) {
      node = stack.pop();

      if (node->_size >= size && node->_free) {
        if (alignment > 0) {
          /* see if can meet the alignment needs */
          auto va = node->_addr;
          if(round_up(va, alignment) + size < (va + node->_size))
            return node;
          //if(check_aligned(node->_addr, alignment))
          return nullptr; /* does not work */
        }
        else {
          return node; /* no alignment */
        }
      }

      /* carry on search */
      Memory_region* l = node->left();
      if (l) stack.push(l);

      Memory_region* r = node->right();
      if (r) stack.push(r);

    }

    return nullptr;
  }

  /**
   * Find a free region that can contains requested address.
   *
   * @param addr Address that should be contained
   *
   * @return nullptr if not found, otherwise containing region
   */
  Memory_region* find_containing_region(Memory_region* node, addr_t addr) {
    
    if (node == nullptr) return nullptr;

    common::Fixed_stack<Memory_region*> stack;  // good for debugging
    stack.push(node);

    while (!stack.empty()) {
      node = stack.pop();

      if ((addr >= node->_addr) &&
          (addr < (node->_addr + node->_size))) {  // to check
        return node;
      }
      else {
        Memory_region* l = node->left();
        if (l) stack.push(l);

        Memory_region* r = node->right();
        if (r) stack.push(r);
      }
    }

    return nullptr; /* not found */
  }

} __attribute__((packed));


/**
 * Simple AVL-based heap memory manager.  Template provides slab
 * allocator for node elements.
 *
 */
class AVL_range_allocator {
private:
  static constexpr bool option_DEBUG = false;

  core::slab::CRuntime<Memory_region> __default_allocator;

  AVL_tree<Memory_region>* _tree;

public:
  /**
   * Constructor
   *
   * @param slab Slab allocator for metadata
   * @param addr Start of memory address
   * @param size Size of memory slab in bytes
   *
   */
  AVL_range_allocator(addr_t start, size_t size)
    : AVL_range_allocator(__default_allocator, start, size) {}

  AVL_range_allocator(const AVL_range_allocator &) = delete;
  AVL_range_allocator &operator=(const AVL_range_allocator &) = delete;

  /**
   * Constructor
   *
   * @param slab Slab allocator for metadata
   * @param addr Start of memory address
   * @param size Size of memory slab in bytes
   *
   */
  AVL_range_allocator(common::Base_slab_allocator& slab,
                      addr_t start,
                      size_t size)
    : __default_allocator{}
    , _tree(nullptr) /* filled in below */
    , _slab(slab) {
      assert(size > 0);

#ifdef DEBUG_AVL_ALLOCATOR
      PLOG("AVL: slab %lu MB", REDUCE_MB(size));
#endif

      /* establish root */
      packed_ptr<core::AVL_node<core::Memory_region>>* root = nullptr;
      bool newroot = true;

      if (slab.is_reconstructed()) {
        newroot = false;
        /* the first entry will be the root (at least it should be!) */
        root = reinterpret_cast<packed_ptr<core::AVL_node<core::Memory_region>>*>(reinterpret_cast<addr_t>(slab.get_first_element()));

        if (option_DEBUG) PLOG("reconstructed root pointer: %p", common::p_fmt(root));
      }
      else {
        /* create root pointer on slab */
        root =
          reinterpret_cast<packed_ptr<core::AVL_node<core::Memory_region>>*>(slab.alloc());

        if (option_DEBUG) PLOG("new root pointer: %p", static_cast<void*>(root));

        *root = packed_ptr<core::AVL_node<core::Memory_region>>{};
      }

      /* now we can create the tree and pass the root pointer */
      _tree = new AVL_tree<Memory_region>(root);

      if (newroot) {
        void* p = slab.alloc();

        if (!p)
          throw General_exception("AVL_range_allocator: failed to allocate from slab");

        if (option_DEBUG)
          PLOG("inserting root region (%lx-%lx)", start, start + size);

        try {
          _tree->insert_node(new (p) Memory_region(start, size));
        }
        catch (...) {
          throw Logic_exception("inserting memory region (0x%lx-0x%lx) conflict detected.",
                                start,
                                start + size);
        }
      }
    }

  addr_t base() {
    /* update base */
    Memory_region* leftmost = leftmost_region();
    assert(leftmost);
    return leftmost->addr();
  }

  /**
   * Destructor
   *
   *
   */
  virtual ~AVL_range_allocator() {
    /* delete node memory */
    _tree->apply_topdown([=](void* p, size_t) {
                           Memory_region* mr = static_cast<Memory_region*>(p);
                           if ( option_DEBUG )  {
                             PLOG("%s: region %p, 0x%zx, %s", __func__, mr->paddr(),
                                  mr->size(), mr->is_free() ? "free" : "used");
                           }
                           _slab.free(mr);
                         });
    _slab.free(_tree->root());
    delete _tree;
  }

  /**
   * Allocate a region of memory
   *
   * @param size Size in bytes of region to allocate
   * @param alignment Alignment requirement in bytes
   *
   * @return Pointer to
   */
  Memory_region* alloc(size_t size, size_t alignment = 0)
  {
    try {

#ifdef DEBUG_AVL_ALLOCATOR
      PLOG("AVL: request  %lu KiB", REDUCE_KB(size));
#endif

      /* look for fitting region that is aligned */
      Memory_region* root = static_cast<Memory_region*>(&**(_tree->root()));
      Memory_region* aligned_region = root->find_free_region(root, size, alignment);

      if (aligned_region) {
        
        assert(aligned_region->_size >= size);
        assert(check_aligned(aligned_region->_addr, alignment));

        /* we found an exact fit, easy */
        if (aligned_region->_size == size) {
          aligned_region->_free = false;
        }
        else {
          /* alignment is OK but its not an exact fit, so we'll create a remaining region on the right  */          
          size_t right_remaining_size = aligned_region->_size - size;
          Memory_region* right_remaining = new (_slab.alloc()) Memory_region(aligned_region->_addr + size, right_remaining_size);

          auto r_adjacent = aligned_region->_next;

          // aligned_region->_addr remains same
          aligned_region->_size = size;
          aligned_region->_free = false;
          aligned_region->_next = right_remaining;
          // aligned_region->_prev remains same
          
          right_remaining->_prev = aligned_region;
          right_remaining->_next = r_adjacent;
          if(r_adjacent)
            r_adjacent->_prev = right_remaining;
          right_remaining->_size = right_remaining_size;
          right_remaining->_free = true;
          _tree->insert_node(right_remaining);
        }

        return aligned_region;
      }
      else {
        
        /* OK, maybe there still is space, but alignment isn't there.  Now we need
           to three-way split a large enough block */
        assert(alignment > 0);

        Memory_region* region = root->find_free_region(root, size);
        if (region == nullptr) {
          return nullptr; /* no candidate */
        }

        assert(region->_free == true);
        if (option_DEBUG) {
          PLOG("Region to split: %lx-%lx size=%lu (requested size=%lu, requested alignment = %lu, free=%d)",
               region->_addr, region->_addr + region->_size, region->_size, size, alignment, region->_free);
          
          assert(region->_addr % alignment);
          
          PLOG("%lx rounded up %lx", region->_addr, round_up(region->_addr, alignment));
          assert(region->_addr % alignment);
        }

        /* left split */
        size_t left_split_size = round_up(region->_addr, alignment) - region->_addr;
        if (option_DEBUG) {
          PLOG("Left split:   %lx-%lx size=%lu", region->_addr, region->_addr+left_split_size, left_split_size);
        }

        /* center split */
        addr_t center_split_base = region->_addr + left_split_size;
        size_t center_split_size = size;

        if (option_DEBUG) {
          PLOG("Center split: %lx-%lx size=%lu (remaining=%lu)",
               center_split_base, center_split_base + center_split_size,
               center_split_size, center_split_base % alignment);
          assert(center_split_base % alignment == 0);
        }

        /* right split */
        addr_t right_split_base = center_split_base + center_split_size;
        size_t right_split_size = region->_size - left_split_size - center_split_size;

        if (option_DEBUG) {
          PLOG("Right split:  %lx-%lx size=%lu", right_split_base, right_split_base + right_split_size, right_split_size);
        }
        assert(right_split_size > 0);
        assert(left_split_size > 0);
        assert(center_split_size > 0);

        /* allocate and adjust nodes */
        void* p = _slab.alloc();
        if (!p) throw General_exception("AVL_range_allocator: failed to allocate %ld", size);
        aligned_region =
          new (p) Memory_region(center_split_base, center_split_size);

        void* q = _slab.alloc();
        if (!q) throw General_exception("AVL_range_allocator: failed to allocate %ld", size);
        Memory_region* right_node =
          new (q) Memory_region(right_split_base, right_split_size);

        auto adjacent_right = region->_next;

        region->_size = left_split_size;
        region->_next = aligned_region;
        aligned_region->_prev = region;
        aligned_region->_free = false;
        aligned_region->_next = right_node;
        right_node->_prev = aligned_region;
        right_node->_next = adjacent_right;
        if(adjacent_right)
          adjacent_right->_prev = right_node;

        _tree->insert_node(aligned_region);
        _tree->insert_node(right_node);

        if(alignment > 0) {
          assert(check_aligned(aligned_region->_addr, alignment));
        }

      }

#ifdef DEBUG_AVL_ALLOCATOR
      PLOG("AVL: remaining free: %lu MB", REDUCE_MB(get_free()));
#endif
      return aligned_region;
    }
    catch ( const Exception &e )
      {
        if ( option_DEBUG )
          {
            PINF("%s: size 0x%zx, alignment 0x%zx, %s", __func__, size, alignment, e.cause());
            std::string s;
            dump_info(&s);
            std::cout << s << "\n";
          }
        throw;
      }
  }

  /**
   * Get the leftmost region in the tree (gives base of region)
   *
   *
   * @return Leftmost memory region
   */
  Memory_region* leftmost_region() {
    assert(_tree);
    Memory_region* r = static_cast<Memory_region*>(&**(_tree->root()));
    if (r == nullptr)
      throw Logic_exception("AVL memory range tree root is nullptr");

    while (r->left()) r = r->left();
    return r;
  }

  /**
   * Get the rightmost region in the tree
   *
   *
   * @return Rightmost memory region (give top end of region)
   */
  Memory_region* rightmost_region() {
    Memory_region* r = static_cast<Memory_region*>(&**(_tree->root()));
    while (r->right()) r = r->right();
    return r;
  }

  /**
   * Allocate memory region at a specific location (used for rebuild). TODO
   * needs thorough testing
   *
   * @param addr Address to allocate at
   * @param size Region size
   *
   * @return
   */
  Memory_region* alloc_at(addr_t addr, size_t size) {
    // find fitting region
    Memory_region* root = static_cast<Memory_region*>(&**(_tree->root()));
    Memory_region* region = root->find_containing_region(root, addr);

    if (region == nullptr)
      throw API_exception("alloc_at: cannot find containing region (addr=%lx, size=%ld)",
                          addr, size);

    if (option_DEBUG)
      PLOG("alloc_at (addr=0x%lx,size=%ld) found fitting region %lx-%lx:", addr,
           size, region->_addr, region->_addr + region->_size);

    if(region->_size < size)
      throw Logic_exception("alloc_at failed because region (%lx,%lu) "
                            "is insufficient for requested size (%lu)",
                            std::size_t(region->_addr), std::size_t(region->_size), size);

    Memory_region* middle = nullptr;

    assert(region->_free);

    /* first split off front if needed and create middle chunk
       which will become the allocation
    */
    if (region->_addr != addr) {
      assert(addr > region->_addr);

      auto left_size = addr - region->_addr; // new size for left region
      middle = new (_slab.alloc()) Memory_region(addr, region->_size - left_size);

      region->_size = left_size;  // make the containing region left chunk
      middle->_next = region->_next;
      middle->_prev = region;
      region->_next = middle;
      // region previous stays the same
      _tree->insert_node(middle);

    }
    else {
      middle = region;  // containing node becomes middle chunk (inserted)
    }

    middle->_free = false;  // middle chunk will become our allocated region

    /* now we have middle+right chunk combined; we must split off right-chunk */
    if (middle->_size > size) {
      auto right_size = middle->_size - size;
      void* p = _slab.alloc();
      if (!p)
        throw General_exception(
                                "AVL_range_allocator: line %d failed to allocate %ld bytes",
                                __LINE__, size);

      Memory_region* right = new (p) Memory_region(middle->_addr + size, right_size);

      right->_size = right_size;
      right->_next = middle->_next;
      right->_prev = middle;
      middle->_next = right;
      middle->_size = size;  // shrink middle chunk
      _tree->insert_node(right);
    }

    return middle;
  }

  /**
   * Find a block with corresponding start address
   *
   * @param addr Start address
   *
   * @return Pointer to memory region if found, otherwise nullptr
   */
  Memory_region* find(addr_t addr) {
    Memory_region* root = static_cast<Memory_region*>(&**(_tree->root()));
    return root->find_region(root, addr);
  }

  /**
   * Add a new memory region to the allocator
   *
   * @param addr Base address of region
   * @param size Size of region in bytes
   */
  void add_new_region(addr_t addr, size_t size) {

#ifdef DEBUG_AVL_ALLOCATOR
    PLOG("AVL: add to slab %lu MB", REDUCE_MB(size));
#endif
    assert(addr > 0);
    assert(size > 0);
    void* p = _slab.alloc();
    if (!p)
      throw General_exception("AVL_range_allocator: line %d failed to allocate %ld bytes", __LINE__, size);

    auto new_region = new (p) Memory_region(addr, size);
    assert(new_region);
    _tree->insert_node(new_region);

    //    update_base_from_tree();
  }

  /**
   * Free an allocated region
   *
   * @param addr Start address of the memory region
   *
   * @return Number of bytes freed
   */
  size_t free(addr_t addr) {
    Memory_region* region = find(addr);
    if (region == nullptr) {
      PERR("invalid call to %s: bad address", __func__);
      return size_t(-1);
    }

    size_t return_count = region->_size;
    region->_free = true;

    // right coalesce
    {
      Memory_region* right = region->_next;
      if (right && right->_free) {
        region->_size += right->_size;
        region->_next = right->_next;
        if (right->_next) right->_next->_prev = region;
        _tree->remove_node(right);

        // release node memory
        _slab.free(right);
      }
    }

    // left coalesce
    {
      Memory_region* left = region->_prev;
      if (left && left->_free) {
        left->_size += region->_size;
        left->_next = region->_next;
        if (region->_next) region->_next->_prev = left;
        _tree->remove_node(region);

        // release node memory
        _slab.free(region);
      }
    }

    return return_count;
  }

  /**
   * Dump the tree for debugging purposes
   *
   */
  void dump_info(std::string * out_str = nullptr) {
    assert(_tree->root());

    if(out_str) {
      apply([&](addr_t addr, size_t s, bool state)
            {
              std::stringstream ss;
              ss << "(" << std::hex << addr << "," << s << "," << state << ")";
              out_str->append(ss.str());
            });
    }
    else {
      PINF("%s", "+ AVL_tree: ");
      AVL_tree<Memory_region>::dump(&**(_tree->root()));
    }

  }

  /**
   * Apply function to each region
   *
   * @param functor
   */
  void apply(std::function<void(addr_t, size_t, bool)> functor) {
    _tree->apply_topdown([functor](void* p, size_t) {
                           Memory_region* mr = static_cast<Memory_region*>(p);
                           functor(mr->addr(), mr->size(), mr->is_free());
                         });
  }


  /**
   * Get total number of free bytes
   */
  size_t get_free() {
    size_t total = 0;
    _tree->apply_topdown([&total](void* p, size_t) {
                           Memory_region* mr = static_cast<Memory_region*>(p);
                           if (mr->is_free())
                             total += mr->size();
                         });
    return total;
  }


private:
  /* NOTE: specifically no members that will be on the stack */

  common::Base_slab_allocator& _slab; /**< volatile slab allocator for metadata */
};

/**
 * AVL-based heap memory manager with transparency. This class ties everything
 * together. This class requires that the slab allocator and region are
 * explicitly passed to the constructor.  This should allow the use of memory
 * allocated for SPDK IO buffers etc.
 *
 */
class Arena_allocator : public common::Base_memory_allocator {
public:
  /**
   * Constructor
   *
   * @param metadata_slab Slab allocator for AVL tree nodes
   * @param region Region of memory to manage
   * @param region_size Size of memory region
   *
   */
  Arena_allocator(common::Base_slab_allocator& metadata_slab, const void* region,
                  size_t region_size)
    : _range_allocator(metadata_slab, reinterpret_cast<addr_t>(region), region_size) {
    if (!region)
      throw Constructor_exception("nullptr 'region' parameter in Arena_allocator");

    if (region_size == 0)
      throw Constructor_exception("Zero 'region_size' parameter in Arena_allocator");
  }

  /**
   * Allocate a variable sized block of memory
   *
   * @param size Size to allocate in bytes
   * @param numa_node
   * @param alignment
   *
   * @return Pointer to newly allocated memory.
   */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
  void* alloc(size_t size, int numa_node = -1, size_t alignment = 0) {
    if (size == 0) size = 1;

    try {
      Memory_region* mr = _range_allocator.alloc(size);
      assert(mr);
      return reinterpret_cast<void*>(mr->addr());
    } catch (const General_exception &e) {
      PERR("Arena_allocator: out of memory %s", e.cause());
      std::exit(0);
    }
  }
#pragma GCC diagnostic pop

  /**
   * Free a previously allocated block
   *
   * @param ptr Pointer to block
   */
  size_t free(void* ptr) {
    assert(ptr);
    return _range_allocator.free(reinterpret_cast<addr_t>(ptr));
  }

  /**
   * Get the size of an allocation
   *
   * @param ptr Pointer to allocated block
   *
   * @return Size of allocation in bytes
   */
  size_t get_size(void* ptr) {
    assert(ptr);
    const core::Memory_region* region = _range_allocator.find(reinterpret_cast<addr_t>(ptr));
    if (region == nullptr)
      return std::size_t(-1);
    else
      return region->size();
  }

  /**
   * Get free space capacity in bytes
   *
   *
   * @return Unused space in bytes
   */
  size_t free_space() {
    size_t free_bytes = 0;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
    apply([&free_bytes](addr_t a, size_t s, bool is_free) {
            if (is_free) free_bytes += s;
          });
#pragma GCC diagnostic pop
    return free_bytes;
  }

  /**
   * Used to determine how many entries the slab allocator should support
   *
   * @param entries
   *
   * @return
   */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
  static size_t recommend_metadata_slab_size(size_t entries) {
    // TODO
    return 0;
  }
#pragma GCC diagnostic pop

  /**
   * Output debugging information
   *
   */
  void dump_info(std::string * out_str = nullptr) {
    _range_allocator.dump_info(out_str);
  }

  /**
   * Apply functor to elements of the tree
   *
   * @param functor
   */
  void apply(std::function<void(addr_t, size_t, bool)> functor) {
    _range_allocator.apply(functor);
  }

  /**
   * Add a new region to the allocator
   *
   * @param vaddr Base address (virtual)
   * @param size Size of region in bytes
   */
  void add_new_region(void* vaddr, size_t size) {
    _range_allocator.add_new_region(addr_t(vaddr), size);
  }

  /**
   * Gives the upper limit on memory used by the heap. This is
   * done by looking up the rightmost node in the tree.
   *
   *
   * @return Upper limit in bytes
   */
  size_t used_zone_limit() {
    Memory_region* r = _range_allocator.rightmost_region();

    if (r->is_free())
      return r->addr() - _range_allocator.base();
    else
      return r->addr() + r->size() - _range_allocator.base();
  }

private:
  core::AVL_range_allocator _range_allocator;
};

/**
 * AVL-based heap region manager.  This uses the C-runtime heap for slab
 * purposes and therefore is only used as an in-memory structure.
 *
 */
class Region_allocator {
public:
  /**
   * Constructor
   *
   * @param region Region of memory to manage
   * @param region_size Size of memory region
   *
   */
  Region_allocator(addr_t region, size_t region_size)
    : _slab(), _range_allocator(_slab, region, region_size) {
    if (region_size == 0)
      throw Constructor_exception("invalid 'region' parameter");
  }

  /**
   * Allocate a variable sized block of memory
   *
   * @param size Size to allocate in bytes
   * @param alignment Alignment in bytes
   *
   * @return Pointer to newly allocated memory.
   */
  addr_t alloc(size_t size) {
    if (size == 0) size = 1;

    Memory_region* mr = _range_allocator.alloc(size);

    if (mr == nullptr) throw General_exception("no memory.");

    return mr->addr();
  }

  /**
   * Allocate a region at a specific location
   *
   * @param region
   * @param region_size
   *
   * @return
   */
  status_t alloc_at(addr_t region, size_t region_size) {
    auto r = _range_allocator.alloc_at(region, region_size);
    return r == nullptr ? E_FAIL : S_OK;
  }

  /**
   * Get memory region for a specific address
   *
   * @param region
   *
   * @return Pointer to memory region node
   */
  const core::Memory_region* contains(addr_t region) {
    return static_cast<const core::Memory_region*>(_range_allocator.find(region));
  }

  /**
   * Free a previously allocated block
   *
   * @param ptr Pointer to block
   */
  void free(addr_t addr) { _range_allocator.free(addr); }

  /**
   * Debugging dump
   *
   */
  void dump_info() { _range_allocator.dump_info(); }

private:
  core::slab::CRuntime<core::Memory_region> _slab;
  core::AVL_range_allocator _range_allocator;
};
}  // namespace core

template <>
struct mr_traits<core::AVL_range_allocator>
{
  static auto allocate(core::AVL_range_allocator *pmr, unsigned, std::size_t bytes, std::size_t alignment)
  {
    return pmr->alloc(bytes, alignment)->addr();
  }
  static auto deallocate(core::AVL_range_allocator *pmr, unsigned, void *p, std::size_t, std::size_t)
  {
    return pmr->free(reinterpret_cast<addr_t>(p));
  }
};

#endif  //__NUPM_AVL_MALLOC_H__
