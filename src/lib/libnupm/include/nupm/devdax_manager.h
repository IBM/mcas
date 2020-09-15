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

#ifndef __NUPM_DAX_MAP_H__
#define __NUPM_DAX_MAP_H__

#include "nd_utils.h"
#include <common/fd_open.h>
#include <common/logging.h>
#include <common/moveable_ptr.h>
#include <boost/icl/interval_set.hpp>
#include <boost/icl/right_open_interval.hpp>
#include <mutex>
#include <string>
#include <tuple>
#include <sys/uio.h>

struct iovec_owned
{
  common::moveable_ptr<void> iov_base;
  std::size_t iov_len;
  explicit iovec_owned(void *iov_base_, std::size_t iov_len_)
    : iov_base(iov_base_)
    , iov_len(iov_len_)
  {}
  iovec_owned(iovec_owned &&o_) noexcept = default;
  iovec_owned &operator=(iovec_owned &&) noexcept = default;
};

namespace nupm
{
class DM_region_header;
class Devdax_manager;

struct registered_instance
{
private:
  std::string _path;
public:
  registered_instance(const std::string &path);
  registered_instance(const registered_instance &) = delete;
  registered_instance &operator=(const registered_instance &) = delete;
  registered_instance(registered_instance &&) noexcept;
  ~registered_instance();
};

struct registered_range
{
private:
  common::moveable_ptr<Devdax_manager> _dm;
  const void *_begin;
  std::size_t _size;
public:
  registered_range(Devdax_manager *dm_, const void *begin_, std::size_t size_);
  registered_range(const registered_range &) = delete;
  registered_range &operator=(const registered_range &) = delete;
  registered_range(registered_range &&) noexcept = default;
  ~registered_range();
};

struct opened_region : private common::log_source
{
  iovec_owned iov;
  common::Fd_open _fd_open;
public:
  opened_region(unsigned debug_level_, const std::string &path, const addr_t base_addr);
  opened_region(opened_region &&) noexcept = default;
  opened_region &operator=(opened_region &&) noexcept = default;
  ~opened_region();
};

struct registered_opened_region
{
private:
  registered_instance _ri;
public:
  opened_region _or;
private:
  registered_range _rr;
public:
  registered_opened_region(unsigned debug_level_, Devdax_manager * dm_, const std::string &path_, addr_t base_addr_)
    : _ri(path_)
    , _or(debug_level_, path_, base_addr_)
    , _rr(dm_, _or.iov.iov_base, _or.iov.iov_len)
  {}
  registered_opened_region(const registered_opened_region &) = delete;
  registered_opened_region &operator=(const registered_opened_region &) = delete;
  registered_opened_region(registered_opened_region &&) noexcept = default;
  registered_opened_region &operator=(registered_opened_region &&) noexcept = default;
};

/**
 * Lowest level persisent manager for devdax devices. See dax_map.cc for static
 * configuration.
 *
 */
class Devdax_manager : protected common::log_source {
 private:
  static constexpr const char *_cname = "Devdax_manager";

 public:

  struct config_t {
    std::string path;
    addr_t addr;
    unsigned region_id;
    /* Through no fault of its own, config_t may begin life with no proper values */
    config_t() : path(), addr(0), region_id(0) {}
  };

  /**
   * Constructor e.g.
     nupm::Devdax_manager ddm({{"/dev/dax0.3", 0x9000000000, 0},
                               {"/dev/dax1.3", 0xa000000000, 1}},
                                true);
   *
   * @param dax_config Vector of dax-path, address, region_id tuples.
   * @param force_reset
   */
  Devdax_manager(const std::vector<config_t>& dax_config,
                 bool force_reset = false);

  Devdax_manager(unsigned debug_level, const std::vector<config_t>& dax_config,
                 bool force_reset = false);

  /**
   * Destructor will not unmap memory/nor invalidate pointers?
   *
   */
  ~Devdax_manager();

  /**
   * Open a region of memory
   *
   * @param uuid Unique identifier
   * @param region_id Region identifier (normally 0)
   * @param out_length Out length of region in bytes
   *
   * @return Pointer to mapped memory or nullptr on not found
   */
  void *open_region(uint64_t uuid, unsigned region_id, size_t *out_length);

  /**
   * Create a new region of memory
   *
   * @param uuid Unique identifier
   * @param region_id Region identifier (normally 0)
   * @param size Size of the region requested in bytes
   *
   * @return Pointer to mapped memory
   */
  void *create_region(uint64_t uuid, unsigned region_id, const size_t size);

  /**
   * Erase a previously allocated region
   *
   * @param uuid Unique region identifier
   * @param region_id Region identifier (normally 0)
   */
  void erase_region(uint64_t uuid, unsigned region_id);

  /**
   * Get the maximum "hole" size.
   *
   *
   * @return Size in bytes of max hole
   */
  size_t get_max_available(unsigned region_id);

  /**
   * Debugging information
   *
   * @param region_id Region identifier
   */
  void debug_dump(unsigned region_id);

  void register_range(const void *begin, std::size_t size);
  void deregister_range(const void *begin, std::size_t size);
 private:
#if 0
  void *get_devdax_region(const std::string &device_path, size_t *out_length);
#endif
  opened_region map_region(const std::string &path, addr_t base_addr);
  void  recover_metadata(const std::string &device_path,
                         void *      p,
                         size_t      p_len,
                         bool        force_rebuild = false);
  const char * lookup_dax_device(unsigned region_id);

 private:
  using guard_t = std::lock_guard<std::mutex>;
  using mapped_regions = std::map<std::string, registered_opened_region>;

  const std::vector<config_t>               _dax_configs;
  ND_control                                _nd;
  using AC = boost::icl::interval_set<const char *>;
  AC                                        _address_coverage;
  mapped_regions                            _mapped_regions;
  std::map<std::string, DM_region_header *> _region_hdrs;
  std::mutex                                _reentrant_lock;
 public:
  friend struct nupm::registered_range; /* access to _address_coverage */
};
}  // namespace nupm

#endif
