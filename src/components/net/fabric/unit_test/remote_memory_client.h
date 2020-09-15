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
#ifndef _TEST_REMOTE_MEMORY_CLIENT_H_
#define _TEST_REMOTE_MEMORY_CLIENT_H_

#include "remote_memory_accessor.h"

#include <common/types.h> /* status_t */
#include <cstdint> /* uint16_t, uint64_t */
#include <string>
#include <memory> /* shared_ptr */

namespace component
{
  class IFabric;
  class IFabric_client;
  class IFabric_communicator;
}

struct registered_memory;

struct remote_memory_client
  : public remote_memory_accessor
{
private:
  static void check_complete_static(void *t, void *ctxt, ::status_t stat, std::size_t len);
  void check_complete(::status_t stat, std::size_t len);

  std::shared_ptr<component::IFabric_client> _cnxn;
  std::shared_ptr<registered_memory> _rm_out;
  std::shared_ptr<registered_memory> _rm_in;
  std::uint64_t _vaddr;
  std::uint64_t _key;
  char _quit_flag;
  ::status_t _last_stat;

  registered_memory &rm_in() const { return *_rm_in; }
  registered_memory &rm_out() const { return *_rm_out; }
protected:
  void do_quit();
public:
  remote_memory_client(
    component::IFabric &fabric
    , const std::string &fabric_spec
    , const std::string ip_address
    , std::uint16_t port
    , std::size_t memory_size
    , std::uint64_t remote_key_base
  );
  remote_memory_client(remote_memory_client &&) noexcept = default;
  remote_memory_client &operator=(remote_memory_client &&) noexcept = default;

  ~remote_memory_client();

  void send_disconnect(component::IFabric_communicator &cnxn_, registered_memory &rm_, char quit_flag_);

  std::uint64_t vaddr() const { return _vaddr; }
  std::uint64_t key() const { return _key; }
  component::IFabric_client &cnxn() { return *_cnxn; }

  void write(const std::string &msg_, bool force_error = false);
  void write_badly(const std::string &msg_) { return write(msg_, true); }

  void read_verify(const std::string &msg_);
  std::size_t max_message_size() const;
};

#endif
