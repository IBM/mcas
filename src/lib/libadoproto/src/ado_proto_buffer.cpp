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

#include "ado_proto_buffer.h"
#include "ado_proto.h"
#include <common/errors.h> /* S_OK */
#include <common/utils.h> /* cpu_relax */
#include "uipc.h"

ado_protocol_buffer::space_shared_deleter::space_shared_deleter(channel_t ch_)
  : _channel(ch_)
{}

void ado_protocol_buffer::space_shared_deleter::operator()(void *p) const
{
  if (p)
  {
    while ( S_OK != ::uipc_free_message(_channel, p) )
    {
      cpu_relax();
    }
  }
}

ado_protocol_buffer::space_dedicated_deleter::space_dedicated_deleter(ADO_protocol_builder *builder_)
  : _builder(builder_)
{}

void ado_protocol_buffer::space_dedicated_deleter::operator()(void *p) const
{
  if (p)
  {
    _builder->free_ipc_buffer(p);
  }
}
