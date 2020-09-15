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
#ifndef _TEST_PINGPONG_CLIENT_H_
#define _TEST_PINGPONG_CLIENT_H_

#include "pingpong_stat.h"
#include <common/types.h> /* status_t */
#include <cstdint> /* int8_t, uint16_t, uint64_t */
#include <string>
#include <memory> /* shared_ptr */

namespace component
{
  class IFabric;
  class IFabric_client;
}

struct registered_memory;

struct pingpong_client
{
private:
  void check_complete(::status_t stat);

  std::shared_ptr<component::IFabric_client> _cnxn;
  pingpong_stat _stat;
  std::uint8_t _id;

protected:
  void do_quit();
public:
  pingpong_client(
    component::IFabric &fabric
    , const std::string &fabric_spec
    , const std::string ip_address
    , std::uint16_t port
    , std::uint64_t buffer_size
    , std::uint64_t remote_key_base
    , unsigned iteration_count
    , std::size_t msg_size
    , std::uint8_t id
  );
  pingpong_client(pingpong_client &&) noexcept = default;
  pingpong_client &operator=(pingpong_client &&) noexcept = default;

  ~pingpong_client();
  pingpong_stat time() const;
};

#endif
