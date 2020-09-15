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

#ifndef __ADOPROTO_BUFFER_H__
#define __ADOPROTO_BUFFER_H__

#include <memory>

struct uipc_channel;

namespace ado_protocol_buffer
{
  /* A "space_shared_deleter" frees a buffer to the original buffer pool */
  class space_shared_deleter
  {
    uipc_channel *_channel;
  public:
    space_shared_deleter(uipc_channel *ch_);
    void operator()(void *p) const;
  };
  using space_shared_ptr_t = std::unique_ptr<void, space_shared_deleter>;
}

class ADO_protocol_builder;

namespace ado_protocol_buffer
{
  /* A "dedicated_space_deleter" frees a buffer to the _normal_buffer field
   * of an ADO_protocol_builder */
  class space_dedicated_deleter
  {
    ADO_protocol_builder *_builder;
  public:
    space_dedicated_deleter(ADO_protocol_builder *builder_);
    void operator()(void *p) const;
  };
  using space_dedicated_ptr_t = std::unique_ptr<void, space_dedicated_deleter>;
}


#endif
