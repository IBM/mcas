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
#include "uipc.h"
#include "uipc_channel.h"

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

#include <unistd.h>
#include <cassert>

/**
 * 'C' exported functions
 *
 */
extern "C"
{
channel_t uipc_create_channel(const char* path_name,
                              size_t message_size,
                              size_t queue_size) {
  return new core::uipc::Channel(path_name, message_size, queue_size);
}

channel_t uipc_connect_channel(const char* path_name) {
  return new core::uipc::Channel(path_name);
}

status_t uipc_close_channel(channel_t channel) {
  try {
    delete reinterpret_cast<core::uipc::Channel*>(channel);
  }
  catch (...) {
    throw General_exception("uipc_close_channel failed unexpectedly");
  }
  return S_OK;
}

void* uipc_alloc_message(channel_t channel) {
  auto ch = static_cast<core::uipc::Channel*>(channel);
  assert(ch);
  return ch->alloc_msg();
}

status_t uipc_free_message(channel_t channel, void* message) {
  auto ch = static_cast<core::uipc::Channel*>(channel);
  assert(ch);
  return ch->free_msg(message);
}

status_t uipc_send(channel_t channel, void* data) {
  auto ch = static_cast<core::uipc::Channel*>(channel);
  assert(ch);
  return ch->send(data);
}

status_t uipc_recv(channel_t channel, void** data_out) {
  auto ch = static_cast<core::uipc::Channel*>(channel);
  assert(ch);
  return ch->recv(*data_out);
}
}
