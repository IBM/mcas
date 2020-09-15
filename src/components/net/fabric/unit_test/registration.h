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
#ifndef _TEST_REGISTRATION_H_
#define _TEST_REGISTRATION_H_


#include <api/fabric_itf.h> /* Fabric_connection, memory_region_t */
#include <common/moveable_ptr.h>
#include <common/delete_copy.h>
#include <cstddef> /* size_t */
#include <cstdint> /* uint64_t */

struct registration
{
private:
  component::IFabric_connection &_cnxn;
  component::IFabric_connection::memory_region_t _region;
  std::uint64_t _key;
  common::moveable_ptr<void> _desc;
  DELETE_COPY(registration); /* due to _region */
public:
  explicit registration(component::IFabric_connection &cnxn_, const void *contig_addr_, std::size_t size_, std::uint64_t key_, std::uint64_t flags_);
  registration(registration &&);
  registration &operator=(registration &&);
  ~registration();

  std::uint64_t key() const { return _key; }
  void *desc() const { return _desc; }
};

#endif
