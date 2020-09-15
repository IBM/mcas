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
#ifndef _TEST_PINGPONG_CB_PACK_H_
#define _TEST_PINGPONG_CB_PACK_H_

#include "pingpong_cb_ctxt.h"

namespace component
{
  class IFabric_connection;
}

struct cb_pack
{
  cb_ctxt _tx_ctxt;
  cb_ctxt _rx0_ctxt;
  cb_ctxt _rx1_ctxt;
  explicit cb_pack(
    cnxn_state &state_
    , component::IFabric_connection &cnxn_
    , cb_ctxt::cb_t send_cb
    , cb_ctxt::cb_t recv_cb
    , std::uint64_t buffer_size_
    , std::uint64_t remote_key_base_
    , std::uint64_t msg_size_
  );
  cb_pack(cb_pack &&) noexcept;
  ~cb_pack();
};

#endif
