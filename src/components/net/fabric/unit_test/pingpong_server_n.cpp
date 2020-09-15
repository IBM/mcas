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
#include "pingpong_server_n.h"

#include "eyecatcher.h"
#include "pingpong_cb_pack.h"
#include "pingpong_server_cb.h"
#include <api/fabric_itf.h> /* IFabric_server_factory */

#include <algorithm> /* transform */
#include <exception>
#include <functional> /* ref */
#include <iostream> /* cerr */
#include <list>
#include <vector>

void pingpong_server_n::listener(
  unsigned client_count_
  , component::IFabric_server_factory &factory_
  , std::uint64_t buffer_size_
  , std::uint64_t remote_key_base_
  , unsigned iteration_count_
  , std::size_t msg_size_
)
try
{
  std::list<client_state> finished_clients;
  std::list<client_state> active_clients;
  std::vector<std::shared_ptr<cb_pack>> cb_vec;
  for ( auto count = client_count_; count != 0; --count, ++remote_key_base_ )
  {
    active_clients.emplace_back(factory_, iteration_count_, msg_size_);
    cb_vec.emplace_back(
      std::make_shared<cb_pack>(
        active_clients.back().st, active_clients.back().st.comm()
        , pingpong_server_cb::send_cb
        , pingpong_server_cb::recv_cb
        , buffer_size_
        , remote_key_base_
        , msg_size_
      )
    );
  }
  _stat.do_start();

  std::uint64_t poll_count = 0U;
  while ( ! active_clients.empty() )
  {
    std::list<client_state> serviced_clients;
    for ( auto it = active_clients.begin(); it != active_clients.end(); )
    {
      auto &c = *it;
      ++poll_count;
      if ( c.st.comm().poll_completions(cb_ctxt::cb) )
      {
        auto mt = it;
        ++it;
        auto &destination_list =
          c.st.done
          ? finished_clients
          : serviced_clients
          ;
        destination_list.splice(destination_list.end(), active_clients, mt);
      }
      else
      {
        ++it;
      }
    }
    active_clients.splice(active_clients.end(), serviced_clients);
  }

  _stat.do_stop(poll_count);
}
catch ( std::exception &e )
{
  std::cerr << "pingpong_server::" << __func__ << ": " << e.what() << "\n";
  throw;
}

pingpong_server_n::pingpong_server_n(
  unsigned client_count_
  , component::IFabric_server_factory &factory_
  , std::uint64_t buffer_size_
  , std::uint64_t remote_key_base_
  , unsigned iteration_count_
  , std::uint64_t msg_size_
)
  : _stat()
  , _th(
      std::async(
        std::launch::async
        , &pingpong_server_n::listener
        , this
        , client_count_
        , std::ref(factory_)
        , buffer_size_
        , remote_key_base_
        , iteration_count_
        , msg_size_
      )
    )
{
}

pingpong_stat pingpong_server_n::time()
{
  if ( _th.valid() )
  {
    _th.get();
  }
  return _stat;
}

pingpong_server_n::~pingpong_server_n()
try
{
  if ( _th.valid() )
  {
    _th.get();
  }
}
catch ( std::exception &e )
{
  std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
}
