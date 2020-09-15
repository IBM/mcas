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
#ifndef _TEST_PINGPONG_CNXN_STATE_H_
#define _TEST_PINGPONG_CNXN_STATE_H_

#include "pingpong_cb_ctxt.h" /* cb_ctxt::cb_t */
#include <common/moveable_ptr.h>
#include <cstddef> /* size_t */

namespace component
{
  class IFabric_active_endpoint_comm;
}

struct buffer_state;

struct cnxn_state
{
private:
  common::moveable_ptr<component::IFabric_active_endpoint_comm> _comm;
  std::size_t _max_inject_size;
public:
  std::size_t msg_size;
  unsigned iterations_left;
  bool done;
  explicit cnxn_state(
    component::IFabric_active_endpoint_comm &comm_
    , unsigned iteration_count_
    , std::size_t msg_size_
  );
  cnxn_state(cnxn_state &&) noexcept = default;
  cnxn_state& operator=(cnxn_state &&) noexcept = default;
  component::IFabric_active_endpoint_comm &comm() const noexcept { return *_comm; }
  void send(buffer_state &bt, cb_ctxt *tx_ctxt);
};

#endif
