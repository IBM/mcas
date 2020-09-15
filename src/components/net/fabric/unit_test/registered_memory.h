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
#ifndef _TEST_REGISTERED_MEMORY_H_
#define _TEST_REGISTERED_MEMORY_H_

#include "registration.h"
#include <array>
#include <cstddef> /* size_t */
#include <cstdint> /* uint64_t */

namespace component
{
  class IFabric_connection;
}

struct registered_memory
{
private:
  /* There may be an alignment restriction on registered memory, May be 8, or 16. */
  std::vector<char> _memory;
  registration _registration;
  static constexpr std::uint64_t offset = 35;
public:
  /*
   * NOTE: if the memory remote key is used (that is, if the mr attributes do not include FI_PROV_KEY),
   * the key must the unique among registered memories.
   */
  explicit registered_memory(component::IFabric_connection &cnxn, std::size_t size, std::uint64_t remote_key);
  registered_memory(registered_memory &&rm) noexcept = default;
  registered_memory& operator=(registered_memory &&rm) noexcept = default;

  char &operator[](std::size_t ix) { return _memory[ix+offset]; }
  volatile char &first_char() { return _memory[0+offset]; }

  std::uint64_t key() const { return _registration.key(); }
  void *desc() const { return _registration.desc(); }
};

#endif
