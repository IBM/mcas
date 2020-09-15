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
#ifndef _TEST_PINGPONG_BUFFER_STATE_H_
#define _TEST_PINGPONG_BUFFER_STATE_H_

#include "registered_memory.h"
#include <sys/uio.h> /* iovec */
#include <cstddef> /* size_t */
#include <cstdint> /* uint64_t */
#include <vector>

namespace component
{
  class IFabric_connection;
}

struct buffer_state
{
  registered_memory _rm;
  std::vector<::iovec> v;
  std::vector<void *> d;
  explicit buffer_state(component::IFabric_connection &cnxn, std::size_t size, std::uint64_t remote_key, std::size_t msg_size);
  buffer_state(buffer_state &&) noexcept = default;
};

#endif
