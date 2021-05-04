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

#ifndef __RC_ALLOC_LB__
#define __RC_ALLOC_LB__

#include <common/memory.h>
#include <memory>
#include <string>

class Region_map;

/**
 * Reconstituting allocator.  Metadata is held in DRAM (C runtime allocator).
 * NOTE: This class is NOT thread safe.
 *
 */
class Rca_LB : public common::Reconstituting_allocator {
 public:
  /**
   * Constructor
   *
   */
  Rca_LB(unsigned debug_level);

  /**
   * Destructor
   *
   */
  ~Rca_LB();

  /**
   * Add region of memory to be managed
   *
   * @param region_base Base of region
   * @param region_length Size of region in bytes
   * @param numa_node NUMA node
   */
  void add_managed_region(void * region_base,
                          size_t region_length,
                          int    numa_node);

  /**
   * Allocate region of memory
   *
   * @param size Size of memory in bytes
   * @param numa_node NUMA node
   * @param alignment Required alignment
   *
   * @return Pointer to newly allocated region
   */
  void *alloc(size_t size, int numa_node, size_t alignment = 0) override;

  /**
   * Free previously allocated region of memory
   *
   * @param ptr Point to region
   * @param numa_node NUMA node
   */
  void free(void *ptr, int numa_node, size_t size = 0) override;

  /**
   * Reconstitute a previous allocation.  Mark memory as allocated.
   *
   * @param p Address of region
   * @param size Size of region in bytes
   * @param numa_node NUMA node
   */
  void inject_allocation(void *p, size_t size, int numa_node) override;

  /**
   * Dump debugging information
   *
   * @param out_log Optional string to copy to, otherwise output is set to
   * console
   */
  void debug_dump(std::string *out_log = nullptr);

 private:
  std::unique_ptr<Region_map> _rmap;
};



// template <>
// 	struct mr_traits<nupm::Rca_LB>
// 	{
// 		static auto allocate(nupm::Rca_LB *pmr, unsigned numa_node, std::size_t bytes, std::size_t alignment)
// 		{
// 			return pmr->alloc(bytes, int(numa_node), alignment);
// 		}
// 		static auto deallocate(nupm::Rca_LB *pmr, unsigned numa_node, void *p, std::size_t bytes, std::size_t)
// 		{
// 			return pmr->free(p, int(numa_node), bytes);
// 		}
// 	};
#endif
