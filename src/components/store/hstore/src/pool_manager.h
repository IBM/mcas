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


#ifndef COMANCHE_HSTORE_POOL_MANAGER_H
#define COMANCHE_HSTORE_POOL_MANAGER_H

#include <api/kvstore_itf.h> /* status_t */

#include "alloc_key.h"

#include <common/logging.h> /* log_source */
#include <sys/uio.h>
#include <cstddef>
#include <functional>
#include <string>
#include <system_error>
#include <utility> /* tuple */

struct pool_path;

#pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Wunused-parameter"

enum class pool_ec
{
  pool_fail,
  pool_unsupported_mode,
  region_fail,
  region_fail_general_exception,
  region_fail_api_exception,
};

struct pool_category
  : public std::error_category
{
  const char* name() const noexcept override { return "pool_category"; }
  std::string message( int condition ) const noexcept override
  {
    switch ( condition )
    {
    case int(pool_ec::pool_fail):
      return "default pool failure";
    case int(pool_ec::pool_unsupported_mode):
      return "pool unsupported flags";
    case int(pool_ec::region_fail):
      return "region-backed pool failure";
    case int(pool_ec::region_fail_general_exception):
      return "region-backed pool failure (General_exception)";
    case int(pool_ec::region_fail_api_exception):
      return "region-backed pool failure (API_exception)";
    default:
      return "unknown pool failure";
    }
  }
};

namespace
{
  pool_category pool_error_category;
}

struct pool_error
  : public std::error_condition
{
  std::string _msg;
public:
  pool_error(const std::string &msg_, pool_ec val_)
    : std::error_condition(int(val_), pool_error_category)
    , _msg(msg_)
  {}

};

class Devdax_manager;

template <typename Pool>
  struct pool_manager
    : protected common::log_source
  {
    pool_manager(unsigned debug_level_) : common::log_source(debug_level_) {}
    virtual ~pool_manager() {}

    virtual void pool_create_check(const std::size_t size_) = 0;

    virtual void pool_close_check(const std::string &) = 0;

    virtual std::vector<::iovec> pool_get_regions(const Pool &) const = 0;

    /*
     * throws pool_error if create_region fails
     */
    virtual auto pool_create_1(
      const pool_path &path_
      , std::size_t size_
    ) -> std::tuple<void *, std::size_t, std::uint64_t> = 0;

    virtual auto pool_create_2(
      AK_FORMAL
      void *v_
      , std::size_t size_
      , std::uint64_t uuid_
      , component::IKVStore::flags_t flags_
      , std::size_t expected_obj_count_
    ) -> std::unique_ptr<Pool> = 0;

    virtual auto pool_open_1(
      const pool_path &path_
    ) -> void * = 0;

    virtual auto pool_open_2(
      AK_FORMAL
      void *addr_
      , component::IKVStore::flags_t flags_
    ) -> std::unique_ptr<Pool> = 0;

    virtual void pool_delete(const pool_path &path) = 0;
    virtual const std::unique_ptr<Devdax_manager> & devdax_manager() const = 0;
  };
#pragma GCC diagnostic pop

#endif
