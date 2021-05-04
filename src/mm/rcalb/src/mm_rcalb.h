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


#ifndef __MM_RCALB_COMPONENT_H__
#define __MM_RCALB_COMPONENT_H__

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"

#define PREFIX "Rca_LB_memory_manager:"

#include <string.h>
#include <string>
#include <set>
#include <memory>
#include <common/logging.h>
#include <api/mm_itf.h>

#include "safe_print.h"
#include "rc_alloc_lb.h"

class Rca_LB_memory_manager : public component::IMemory_manager_volatile_reconstituting
{
  static constexpr int DEFAULT_NUMA_NODE = 0;
  
public:
  Rca_LB_memory_manager(const unsigned debug_level)
    : _rca_lb(debug_level)
  {
    if(debug_level > 0)
      SAFE_PRINT("Rca_LB_memory_manager: ctor");
  }

  virtual ~Rca_LB_memory_manager() {
  }
  
  DECLARE_VERSION(0.1f);
  DECLARE_COMPONENT_UUID(0xf811cb01,0x0911,0x4bb5,0x93f6,0x79,0xd6,0x11,0xba,0x9a,0xd4);
  
  void* query_interface(component::uuid_t& itf_uuid) override
  {
    if (itf_uuid == component::IMemory_manager_volatile_reconstituting::iid()) {
      return static_cast<component::IMemory_manager_volatile_reconstituting*>(this);
    }
    else return NULL;  // we don't support this interface
  }

  void unload() override { delete this; }

public:
  virtual void debug_dump() override {

    SAFE_PRINT("MM: RCA LB (mmap)");
  }

  virtual status_t allocate(std::size_t n, void **out_ptr) override {
    assert(out_ptr);
    *out_ptr = _rca_lb.alloc(n, 0);
    SAFE_PRINT("MM: RCA LB - ALLOC(%lu) --> %p", n, *out_ptr);
    return S_OK;
  }

  virtual status_t aligned_allocate(size_t n, size_t alignment, void **out_ptr) override {
    assert(n);
    assert(out_ptr);
    if(alignment == 0) return allocate(n, out_ptr);

    *out_ptr = _rca_lb.alloc(n, 0, alignment);
    SAFE_PRINT("MM: RCA LB - ALIGNED_ALLOC(%lu, %lu) --> %p", n, alignment, *out_ptr);
    return S_OK;
  }


  virtual status_t deallocate(void * ptr, std::size_t size) override {
    SAFE_PRINT("MM: RCA LB - FREE(%p, %lu)", ptr, size);
    if(ptr == nullptr || size == 0) return S_OK;
    //    _rca_lb.free(ptr, DEFAULT_NUMA_NODE, size);
    return S_OK;
  }
  
  virtual status_t deallocate_without_size(void * ptr) override {
    SAFE_PRINT("MM: RCA LB - FREE WITHOUT SIZE (%p)", ptr);
    if(ptr == nullptr) return S_OK;
    _rca_lb.free(ptr, DEFAULT_NUMA_NODE);
    return S_OK;
  }
  
  virtual status_t callocate(size_t n, void ** out_ptr) override  {
    if(out_ptr == nullptr) return E_INVAL;
    auto status = allocate(n, out_ptr);
    if(status == S_OK)
      ::memset(*out_ptr, 0, n);
    SAFE_PRINT("MM: RCA LB - CALLOC(%lu) --> %p", n, *out_ptr);        
    return status;
  }

  virtual status_t reallocate(void *ptr, size_t size, void **out_ptr) override {
    *out_ptr = nullptr; /* indicate we can't handle it */
    return S_OK;
  }

  virtual status_t add_managed_region(void * region_base,
                                      size_t region_length) override {

    SAFE_PRINT("MM: RCA LB - ADD MANAGED REGION(%p, %lu)", region_base, region_length);
    _rca_lb.add_managed_region(region_base, region_length, 0 /* numa node */);
    return S_OK;
  }

  virtual status_t inject_allocation(void * ptr, size_t size) override {
    _rca_lb.inject_allocation(ptr, size, 0);
    return S_OK;
  }
    
  
  
private:

  /* The actual Allocator! */
  Rca_LB _rca_lb;

};


/** 
 * Factory
 * 
 */
class Rca_LB_memory_manager_factory : public component::IMemory_manager_factory {
public:
  DECLARE_VERSION(0.1f);

  /* index_factory - see components.h */
  DECLARE_COMPONENT_UUID(0xfac1cb01, 0x0911, 0x4bb5, 0x93f6, 0x79, 0xd6, 0x11, 0xba, 0x9a, 0xd4);

  void* query_interface(component::uuid_t& itf_uuid) override
  {
    if (itf_uuid == component::IMemory_manager_factory::iid())
      return static_cast<component::IMemory_manager_factory*>(this);
    else return NULL;  // we don't support this interface
  }

  void unload() override { delete this; }

  virtual component::IMemory_manager_volatile * create_mm_volatile(const unsigned /*debug_level*/) override
  {
    return nullptr;
  }
  
  virtual component::IMemory_manager_volatile_reconstituting * create_mm_volatile_reconstituting(const unsigned debug_level) override
  {
    
    auto new_instance = static_cast<component::IMemory_manager_volatile_reconstituting *>
      (new Rca_LB_memory_manager(debug_level));
    
    new_instance->add_ref();
    return new_instance;
  }
  
  virtual component::IMemory_manager_persistent * create_mm_persistent(const unsigned /*debug_level*/) override {
    return nullptr;
  }
  
};


#pragma GCC diagnostic pop

#endif // __MM_RCALB_COMPONENT_H__
