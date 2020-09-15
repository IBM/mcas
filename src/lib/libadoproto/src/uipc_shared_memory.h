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

#ifndef __CORE_UIPC_SHARED_MEMORY_H__
#define __CORE_UIPC_SHARED_MEMORY_H__

#include <common/utils.h> /* PAGE_SIZE */
#include <common/logging.h>
#include <string>
#include <vector>

namespace core
{
namespace uipc
{
class Shared_memory : private common::log_source {

 public:

  Shared_memory(const Shared_memory &) = delete;
  Shared_memory& operator=(const Shared_memory &) = delete;
  Shared_memory(const std::string &name, size_t n_pages); /*< initiator constructor */
  Shared_memory(const std::string &name);                 /*< target constructor */

  virtual ~Shared_memory() noexcept(false);

  void* get_addr(size_t offset = 0);

  size_t get_size_in_pages() const { return _mapped_pages._size_in_pages; }
  size_t get_size() const { return get_size_in_pages() * PAGE_SIZE; }

 private:
  struct mapped_pages
  {
    void* _vaddr;
    size_t _size_in_pages;
    void* get_addr(size_t offset);
  };

  void* negotiate_addr_create(const std::string &path_name, size_t size_in_bytes);
  mapped_pages negotiate_addr_connect(const std::string &path_name);

  void open_shared_memory(const std::string &path_name, bool master);

 private:
  bool _master;
  std::vector<std::string> _fifo_names;
  std::string _name;
  mapped_pages _mapped_pages;
};

}  // namespace uipc
}  // namespace core

#endif
