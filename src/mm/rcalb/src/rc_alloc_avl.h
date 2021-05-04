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

#ifndef __RC_ALLOC_AVL_H__
#define __RC_ALLOC_AVL_H__

#include <common/memory.h>
#include <memory>
#include <string>

class Rca_AVL_internal;

/**
 * Reconstituting allocator.  Metadata is held in DRAM (C runtime allocator).
 * NOTE: This class is NOT thread safe.
 *
 */
class Rca_AVL : public common::Reconstituting_allocator {

private:
  Rca_AVL(const Rca_AVL &) = delete;
  Rca_AVL &operator=(const Rca_AVL &) = delete;
  
public:
  Rca_AVL();
  ~Rca_AVL();

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
   * @param size Optional size (may make release faster)
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
  std::unique_ptr<Rca_AVL_internal> _rca;
};

#endif
