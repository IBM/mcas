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


#include "rc_alloc_avl.h"

#include "avl_malloc.h"
#include "slab.h"
#include <boost/numeric/conversion/cast.hpp>
#include <common/exceptions.h>
#include <common/logging.h>
#include <numa.h>
#include <stdexcept>

namespace nupm
{
namespace rca
{
int max_numa_node;
}

__attribute__((constructor)) static void init_Rca()
{
  rca::max_numa_node = numa_max_node();
}

class Rca_AVL_internal : private common::log_source {

 public:
  Rca_AVL_internal() : common::log_source(1), _slab()
    , _allocators(boost::numeric_cast<unsigned>(rca::max_numa_node + 1))
  {
  }

  ~Rca_AVL_internal()
  {
  }

  void add_managed_region(void * region_base,
                          size_t region_length,
                          const int numa_node)
  {
    const auto numa_node_u = boost::numeric_cast<unsigned>(numa_node);
    assert(region_base);
    assert(region_length > 0);

    if (_allocators[numa_node_u] == nullptr) {
      _allocators[numa_node_u] = std::make_unique<core::AVL_range_allocator>(
          _slab, reinterpret_cast<addr_t>(region_base), region_length);
    }
    else {
      _allocators[numa_node_u]->add_new_region(
          reinterpret_cast<addr_t>(region_base), region_length);
    }
  }

  void inject_allocation(void *ptr, size_t size, const int numa_node)
  {
    const auto numa_node_u = boost::numeric_cast<unsigned>(numa_node);
    assert(ptr);
    assert(_allocators[numa_node_u]);

    auto mrp = _allocators[unsigned(numa_node)]->alloc_at(reinterpret_cast<addr_t>(ptr), size);
    if (mrp == nullptr)
      throw General_exception("alloc_at on AVL range allocator failed unexpectedly");
  }

  void *alloc(size_t size, int numa_node, size_t alignment)
  {
    const auto numa_node_u = boost::numeric_cast<unsigned>(numa_node);
    try {
      auto mr = _allocators[numa_node_u]->alloc(size, alignment);
      CPLOG(1, "AVL allocated: 0x%lx size=%lu", mr->addr(), size);

      assert(mr);
      return mr->paddr();
    }
    catch(...) {
      PWRN("%s:%d region allocation out-of-space (requested %lu MiB)", __FILE__, __LINE__, REDUCE_MiB(size));
      throw std::bad_alloc();
    }
    return nullptr;
  }

  void free(void *ptr, int numa_node)
  {
    const auto numa_node_u = boost::numeric_cast<unsigned>(numa_node);
    if (ptr == nullptr)
      throw API_exception("pointer argument to free cannot be null");

    _allocators[numa_node_u]->free(reinterpret_cast<addr_t>(ptr));
  }

  void debug_dump(std::string *out_str)
  {
    if(_allocators[0])
      _allocators[0]->dump_info(out_str);

    if(_allocators[1])
    _allocators[1]->dump_info(out_str);
  }

 private:

  core::slab::CRuntime<core::Memory_region> _slab; /* use C runtime for slab? */
  std::vector<std::unique_ptr<core::AVL_range_allocator>> _allocators;
};

Rca_AVL::Rca_AVL() : _rca(new Rca_AVL_internal()) {}

Rca_AVL::~Rca_AVL() { }

void Rca_AVL::add_managed_region(void * region_base,
                                 size_t region_length,
                                 int    numa_node)
{
  if (numa_node > rca::max_numa_node)
    throw std::invalid_argument("numa node out of range");

  _rca->add_managed_region(region_base, region_length, numa_node);
}

void Rca_AVL::inject_allocation(void *ptr, size_t size, int numa_node)
{
  if (numa_node > rca::max_numa_node)
    throw std::invalid_argument("numa node out of range");

  _rca->inject_allocation(ptr, size, numa_node);
}

void *Rca_AVL::alloc(size_t size, int numa_node, size_t alignment)
{
  if (size == 0) throw std::invalid_argument("invalid size");
  if (alignment == 0) alignment = 1;

  if (numa_node > rca::max_numa_node)
    throw std::invalid_argument("numa node out of range");

  return _rca->alloc(size, numa_node, alignment);
}

void Rca_AVL::free(void *ptr, int numa_node, size_t // size unused
)
{
  _rca->free(ptr, numa_node);
}

void Rca_AVL::debug_dump(std::string *out_log)
{
  _rca->debug_dump(out_log);
}

}  // namespace nupm
