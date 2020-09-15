/*
   Copyright [2017-2020] [IBM Corporation]
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
#ifndef __MCAS_POOL_MANAGER_H__
#define __MCAS_POOL_MANAGER_H__

#include "fabric_connection_base.h"

#include <api/kvstore_itf.h>
#include <common/logging.h> /* log_source */

#include <cassert>
#include <cassert>
#include <map>


namespace mcas
{
/**
   Pool_manager tracks open pool handles on a per-shard basis
 */
class Pool_manager : common::log_source {
 public:
  using pool_t = component::IKVStore::pool_t;
 private:
  const void *to_ptr(component::IKVStore::pool_t p) { return reinterpret_cast<const void *>(p); }

 public:

  Pool_manager() : common::log_source(mcas::global::debug_level), _map_n2p{}, _map_p2n{}, _open_pools{}, _pool_info{} {}

  /**
   * Determine if pool is open and valid
   *
   * @param pool Pool name
   *
   * @return true of pool is open
   */
  bool check_for_open_pool(const std::string& pool_name, pool_t& out_pool) const
  {
    auto i = _map_n2p.find(pool_name);
    if (i == _map_n2p.end()) {
      CPLOG(0, "check_for_open_pool (%s) false", pool_name.c_str());
      out_pool = 0;
      return false;
    }

    auto j = _open_pools.find(i->second);
    if (j != _open_pools.end()) {
      out_pool = i->second;
      CPLOG(0, "check_for_open_pool (%s) true", pool_name.c_str());
      return true;
    }
    out_pool = 0;
    CPLOG(0, "check_for_open_pool (%s) false", pool_name.c_str());
    return false;
  }

  /**
   * Record pool as open
   *
   * @param pool Pool identifier
   */
  void register_pool(const std::string& pool_name,
                     pool_t             pool,
                     uint64_t           expected_obj_count,
                     size_t             size,
                     unsigned int       flags)
  {
    assert(pool);
    if (_open_pools.find(pool) != _open_pools.end()) throw General_exception("pool already registered");

    _open_pools[pool]   = 1;
    _map_n2p[pool_name] = pool;
    _map_p2n[pool]      = pool_name;
    _pool_info[pool]    = {expected_obj_count, size, flags};

    CPLOG(0, "(+) registered pool (%p) ref:%u pm=%p", to_ptr(pool), _open_pools[pool],
          static_cast<void*>(this));
  }

  /**
   * Get pool information
   *
   * @param pool Pool identifier
   * @param expected_obj_count Expected object count
   * @param size Size of pool
   * @param flags Creation flags
   */
  void get_pool_info(const pool_t pool, uint64_t& expected_obj_count, size_t& size, unsigned int& flags)
  {
    auto i = _pool_info.find(pool);
    if (i != _pool_info.end()) {
      expected_obj_count = i->second.expected_obj_count;
      size               = i->second.size;
      flags              = i->second.flags;
    }
  }

  /**
   * Add reference count to open pool
   *
   * @param pool Pool identifier
   */
  void add_reference(pool_t pool)
  {
    if (_open_pools.find(pool) == _open_pools.end()) throw Logic_exception("add reference to pool that is not open");

    _open_pools[pool] += 1;
    CPLOG(0, "(+) inc pool (%p) ref:%u", to_ptr(pool), _open_pools[pool]);
  }

  /**
   * Release open pool
   *
   * @param pool Pool identifier
   *
   * @return Returns true if reference becomes 0
   */
  bool release_pool_reference(pool_t pool)
  {
    auto i = _open_pools.find(pool);
    if (i == _open_pools.end()) throw std::invalid_argument("invalid pool handle");

    i->second -= 1;  // _open_pools[pool]

    CPLOG(0, "(-) release pool (%p) ref:%u", to_ptr(pool), _open_pools[pool]);

    if (i->second == 0) {
      /* zero reference count; erase entries */
      _open_pools.erase(i);
      _map_n2p.erase(_map_p2n[pool]);
      _map_p2n.erase(pool);
      return true;
    }
    return false;
  }

  /**
   * Get current reference count for a pool
   *
   * @param pool Pool identifier
   *
   * @return Reference count
   */
  auto pool_reference_count(pool_t pool)
  {
    auto i = _open_pools.find(pool);
    if (i == _open_pools.end()) throw std::invalid_argument("invalid pool handle");

    return i->second;
  }

  /**
   * Look up pool name
   *
   * @param pool Pool identifier
   *
   * @return Pool name string
   */
  auto pool_name(pool_t pool) { return _map_p2n[pool]; }

  /**
   * Determine if pool is open and valid
   *
   * @param pool Pool identifier
   */
  bool is_pool_open(pool_t pool) const
  {
    auto i = _open_pools.find(pool);

    if (i != _open_pools.end())
      return i->second > 0;
    else
      return false;
  }

  inline const std::map<pool_t, unsigned>& open_pool_set() const { return _open_pools; }

  inline size_t open_pool_count() const { return _open_pools.size(); }

 private:
  struct pool_info_t {
    uint64_t     expected_obj_count;
    size_t       size;
    unsigned int flags;
  };

  std::map<std::string, pool_t> _map_n2p;
  std::map<pool_t, std::string> _map_p2n;
  std::map<pool_t, unsigned>    _open_pools;
  std::map<pool_t, pool_info_t> _pool_info;
};
}  // namespace mcas

#endif  // __mcas_POOL_MANAGER_H__
