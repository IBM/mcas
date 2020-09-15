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

#ifndef __MAP_STORE_COMPONENT_H__
#define __MAP_STORE_COMPONENT_H__

#include <api/kvstore_itf.h>

class Map_store : public component::IKVStore /* generic Key-Value store interface */
{
public:
  static constexpr unsigned debug_level() { return 0; }

  /**
   * Constructor
   *
   * @param block_device Block device interface
   *
   */
  Map_store(const std::string &owner, const std::string &name);

  /**
   * Destructor
   *
   */
  ~Map_store();

  /**
   * Component/interface management
   *
   */
  DECLARE_VERSION(0.1f);
  DECLARE_COMPONENT_UUID(0x8a120985, 0x1253, 0x404d, 0x94d7, 0x77, 0x92, 0x75,
                         0x21, 0xa1, 0x21);

  void *query_interface(component::uuid_t &itf_uuid) override {
    if (itf_uuid == component::IKVStore::iid()) {
      return static_cast<component::IKVStore *>(this);
    }
    else {
      return NULL;  // we don't support this interface
    }
  }

  void unload() override { delete this; }

public:
  /* IKVStore */
  virtual int thread_safety() const override { return THREAD_MODEL_RWLOCK_PER_POOL; }

  virtual int get_capability(Capability cap) const override;

  virtual pool_t create_pool(const std::string &name, const size_t size,
                             unsigned int flags = 0,
                             uint64_t expected_obj_count = 0) override;

  virtual pool_t open_pool(const std::string &name,
                           unsigned int flags = 0) override;

  virtual status_t close_pool(const pool_t pid) override;

  virtual status_t delete_pool(const std::string &name) override;

  virtual status_t put(const pool_t pool, const std::string &key,
                       const void *value, const size_t value_len,
                       unsigned int flags = FLAGS_NONE) override;

  virtual status_t get(const pool_t pool, const std::string &key,
                       void *&out_value, size_t &out_value_len) override;

  virtual status_t get_direct(const pool_t pool, const std::string &key, void *out_value,
                              size_t &out_value_len,
                              IKVStore::memory_handle_t handle) override;

  virtual status_t put_direct(const pool_t pool, const std::string &key,
                              const void *value, const size_t value_len,
                              IKVStore::memory_handle_t handle = HANDLE_NONE,
                              unsigned int flags = FLAGS_NONE) override;

  virtual status_t resize_value(const pool_t pool, const std::string &key,
                                const size_t new_size,
                                const size_t alignment) override;

  virtual status_t get_attribute(const pool_t pool, const Attribute attr,
                                 std::vector<uint64_t> &out_attr,
                                 const std::string *key = nullptr) override;

  virtual status_t swap_keys(const pool_t pool,
                             const std::string key0,
                             const std::string key1) override;

  virtual status_t lock(const pool_t pool, const std::string &key,
                        lock_type_t type, void *&out_value,
                        size_t &out_value_len,
                        IKVStore::key_t &out_key,
                        const char ** out_key_ptr) override;

  virtual status_t unlock(const pool_t pool,
                          key_t key,
                          IKVStore::unlock_flags_t flags) override;

  virtual status_t erase(const pool_t pool, const std::string &key) override;

  virtual size_t count(const pool_t pool) override;

  virtual status_t free_memory(void *p) override;

  virtual status_t map(const pool_t pool,
                       std::function<int(const void * key,
                                         const size_t key_len,
                                         const void *value,
                                         const size_t value_len)> function) override;

  virtual status_t map(const pool_t pool,
                       std::function<int(const void* key,
                                         const size_t key_len,
                                         const void* value,
                                         const size_t value_len,
                                         const common::tsc_time_t timestamp)> function,
                       const common::epoch_time_t t_begin,
                       const common::epoch_time_t t_end) override;

  virtual status_t map_keys(const pool_t pool,
                            std::function<int(const std::string &key)> function) override;

  virtual void debug(const pool_t pool, unsigned cmd, uint64_t arg) override;
  
  virtual status_t get_pool_regions(const pool_t pool,
                                    std::vector<::iovec> &out_regions) override;
  
  virtual status_t grow_pool(const pool_t pool, const size_t increment_size,
                             size_t &reconfigured_size) override;
  
  virtual status_t free_pool_memory(const pool_t pool, const void *addr,
                                    const size_t size = 0) override;
  
  virtual status_t allocate_pool_memory(const pool_t pool, const size_t size,
                                        const size_t alignment,
                                        void *&out_addr) override;

  virtual IKVStore::pool_iterator_t open_pool_iterator(const pool_t pool) override;

  virtual status_t deref_pool_iterator(const pool_t pool,
                                       pool_iterator_t iter,
                                       const common::epoch_time_t t_begin,
                                       const common::epoch_time_t t_end,
                                       pool_reference_t& ref,
                                       bool& time_match,
                                       bool increment = true) override;

  virtual status_t close_pool_iterator(const pool_t pool,
                                       pool_iterator_t iter) override;
};

class Map_store_factory : public component::IKVStore_factory {
public:
  /**
   * Component/interface management
   *
   */
  DECLARE_VERSION(0.1f);
  DECLARE_COMPONENT_UUID(0xfac20985, 0x1253, 0x404d, 0x94d7, 0x77, 0x92, 0x75,
                         0x21, 0xa1, 0x21);

  void *query_interface(component::uuid_t &itf_uuid) override {
    if (itf_uuid == component::IKVStore_factory::iid()) {
      return static_cast<component::IKVStore_factory *>(this);
    } else
      return NULL;  // we don't support this interface
  }

  void unload() override { delete this; }

  virtual component::IKVStore *create(unsigned,
    const IKVStore_factory::map_create &mc) override {
    auto owner_it = mc.find(+component::IKVStore_factory::k_owner);
    auto name_it = mc.find(+component::IKVStore_factory::k_name);
    component::IKVStore *obj =
      static_cast<component::IKVStore *>(
        new Map_store(
          owner_it == mc.end() ? "owner" : owner_it->second
          , name_it == mc.end() ? "name" : name_it->second
        )
    );
    assert(obj);
    obj->add_ref();
    return obj;
  }
};

#endif
