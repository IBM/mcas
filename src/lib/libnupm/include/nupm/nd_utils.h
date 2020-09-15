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

#ifndef __ND_UTILS_H__
#define __ND_UTILS_H__

#include <common/exceptions.h>  // base Exception class
#include <daxctl/libdaxctl.h>
#include <numa.h>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <sys/uio.h> /* iovec */

namespace libndctl
{
/* ndctl_namespace_get_dax declared twice */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <ndctl/libndctl.h>
#pragma GCC diagnostic pop
}

#define DIMM_HANDLE(n, s, i, c, d)                              \
  (((n & 0xfff) << 16) | ((s & 0xf) << 12) | ((i & 0xf) << 8) | \
   ((c & 0xf) << 4) | (d & 0xf))

struct stat;

namespace nupm
{
  class mapping_element : private ::iovec
  {
  public:
    mapping_element(void *vaddr, std::size_t size, int prot, int flags, int fd);
    ~mapping_element();
    using ::iovec::iov_base;
    using ::iovec::iov_len;
  };

class ND_control_exception : public Exception {
 public:
  ND_control_exception() : Exception("ND_control error"), _err_code(E_FAIL) {}

  __attribute__((__format__(__printf__, 2, 0)))
  ND_control_exception(const char *fmt, ...)
      : ND_control_exception()
  {
    va_list args;
    va_start(args, fmt);
    char msg[255] = {0};
    vsnprintf(msg, 254, fmt, args);
    set_cause(msg);
  }

  status_t error_code() { return _err_code; }

 private:
  status_t _err_code;
};

/**
 * NVDIMM control class
 *
 */
class ND_control {
 private:
  static constexpr unsigned      MAX_NUMA_ZONES = 4;
  static constexpr bool          option_DEBUG   = false;
  static constexpr unsigned long ZONE_BASE = 0x90000000000ULL;  // could use ASLR
  static constexpr unsigned long ZONE_DELTA = 0x50000000000ULL;

 public:
  /**
   * Constructor
   *
   */
  ND_control();
  ND_control(const ND_control &) = delete;
  ND_control& operator=(const ND_control &) = delete;

  /**
   * Destructor
   *
   */
  virtual ~ND_control();

  /**
   * Map regions with device DAX
   *
   * @param Hint for base address
   */
  void map_regions(unsigned long base_addr = 0) __attribute__((deprecated));
 private:
  void init_devdax();

 protected:
  bool                               _pmem_present = true;
  unsigned                           _n_sockets;
  struct libndctl::ndctl_ctx *       _ctx;
  struct libndctl::ndctl_bus *       _bus;
  std::map<std::string, int>         _ns_to_socket;
  std::map<std::string, std::string> _ns_to_dax;
  std::map<std::string, std::string> _dax_to_ns;

 private:
  using mapping_vec_t = std::vector<mapping_element>;
  using mappings_t = std::map<int, mapping_vec_t>;
  mappings_t                         _mappings;

  /**
   * Get the mapped regions for a specific NUMA zone
   *
   * @param numa_zone
   *
   * @return Copy of a vector of region base, size pairs
   */
  const mapping_vec_t &get_regions(int numa_zone)
      __attribute__((deprecated));
 protected:
  mappings_t::const_iterator get_mapping(int i) const { return _mappings.find(i); }
  mappings_t::const_iterator get_mappings_end() const { return _mappings.end(); }
};

/**
 * Get size of a dax device
 */
size_t get_dax_device_size(const struct stat &statbuf);
size_t get_dax_device_size(int fd);
size_t get_dax_device_size(const std::string& dax_path);

}  // namespace nupm

#endif
