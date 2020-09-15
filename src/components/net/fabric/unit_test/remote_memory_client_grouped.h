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
#ifndef _TEST_REMOTE_MEMORY_CLIENT_GROUPED_H_
#define _TEST_REMOTE_MEMORY_CLIENT_GROUPED_H_

#include "remote_memory_accessor.h"

#include <common/types.h> /* status_t */
#include <cstdint> /* uint16_t, uint64_t */
#include <cstring> /* string */
#include <memory> /* shared_ptr, unique_ptr */

struct registered_memory;

namespace component
{
  class IFabric;
  class IFabric_client_grouped;
  class IFabric_communicaator;
}

struct remote_memory_client_grouped
  : public remote_memory_accessor
{
private:
  std::shared_ptr<component::IFabric_client_grouped> _cnxn;
  std::size_t _memory_size;
  std::shared_ptr<registered_memory> _rm_out;
  std::uint64_t _vaddr;
  std::uint64_t _key;
  char _quit_flag;
  std::uint64_t _remote_key_index_for_startup_and_shutdown;
  registered_memory &rm_out() const { return *_rm_out; }
public:
  remote_memory_client_grouped(component::IFabric &fabric
    , const std::string &fabric_spec
    , const std::string ip_address
    , std::uint16_t port
    , std::size_t memory_size
    , std::uint64_t remote_key_base
  );

  remote_memory_client_grouped(remote_memory_client_grouped &&) noexcept = default;
  remote_memory_client_grouped &operator=(remote_memory_client_grouped &&) noexcept = default;

  ~remote_memory_client_grouped();

  void send_disconnect(component::IFabric_communicator &cnxn, registered_memory &rm, char quit_flag);

  std::uint64_t vaddr() const { return _vaddr; }
  std::uint64_t key() const { return _key; }
  component::IFabric_client_grouped &cnxn() { return *_cnxn; }

  std::unique_ptr<component::IFabric_communicator> allocate_group() const;
  std::size_t max_message_size() const;
};

#endif
