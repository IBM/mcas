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

#ifndef __CORE_HEAP_ALLOCATOR_H__
#define __CORE_HEAP_ALLOCATOR_H__

#if !defined(__cplusplus)
#error This is a C++ header
#endif

#include "slab.h"
#include "avl_malloc.h"
#include <memory>

namespace core
{

template <class T=void>
class Heap_allocator : public std::allocator<T>
{
  static constexpr size_t MAX_SLAB_SLOTS = 2048;

 public:
  Heap_allocator(void * region,
                 size_t region_size,
                 const std::string label)
      :
      _slab_size(core::slab::Allocator<>::determine_size(MAX_SLAB_SLOTS)),
      _slab(region, _slab_size, label, false),
      _arena_start(static_cast<char *>(region) + _slab_size),
      _arena_size(region_size = _slab_size),
      _arena(_slab, _arena_start, _arena_size)
  {
    PLOG("Heap_allocator: slab size=%lu KiB", REDUCE_KiB(_slab_size));
    PLOG("Heap_allocator: arena @ %p", _arena_start);
  }

  T * allocate(size_t s) {
    return static_cast<T *>(_arena.alloc(s));
  }

 private:
  const size_t                         _slab_size;
  core::slab::Allocator<Memory_region> _slab; /*< slab allocator */
  const void *                         _arena_start; /*< managed arena */
  const size_t                         _arena_size;
  core::Arena_allocator                _arena; /*< arena manager */
};

}

#endif // __CORE_HEAP_ALLOCATOR_H__
