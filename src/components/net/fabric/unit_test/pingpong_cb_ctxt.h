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
#ifndef _TEST_PINGPONG_CB_CTXT_H_
#define _TEST_PINGPONG_CB_CTXT_H_

#include <api/fabric_itf.h>
#include <common/delete_copy.h>
#include "pingpong_buffer_state.h"
#include <cstddef> /* size_t */
#include <cstdint> /* uint64_t */
#include <vector>

struct cnxn_state;

struct cb_ctxt
{
  using cb_t = void (*)(cb_ctxt *ctxt, ::status_t stat, std::size_t len);
private:
  cnxn_state *_state;
  static void cb_simple(void *ctxt, ::status_t stat, std::uint64_t flags, std::size_t len, void *err);
  cb_t cb_call;
  /* Callback are linked to a specific buffer */
  buffer_state _buffer;
  /* Receive callbacks typically require a response. This is the callback state for the response. */
  cb_ctxt *_response;
  const char *_id;
public:
  static component::IFabric_op_completer::complete_definite cb;
  explicit cb_ctxt(
    cnxn_state &
    , cb_t cb_call
    , cb_ctxt *response
    , component::IFabric_connection &cnxn
    , std::size_t buffer_size
    , std::uint64_t remote_key
    , std::size_t msg_size
    , const char *id
  );
  DELETE_COPY(cb_ctxt);
  cnxn_state &state() { return *_state; }
  buffer_state &buffer() { return _buffer; }
  cb_ctxt *response() { return _response; }
  const char *id() { return _id; }
};

#endif
