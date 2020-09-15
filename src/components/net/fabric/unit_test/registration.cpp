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
#include "registration.h"

#include "eyecatcher.h"
#include <api/fabric_itf.h> /* Fabric_connection */
#include <exception>
#include <iostream> /* cerr */

registration::registration(component::IFabric_connection &cnxn_, const void *contig_addr_, std::size_t size_, std::uint64_t key_, std::uint64_t flags_)
  : _cnxn(cnxn_)
  , _region(_cnxn.register_memory(contig_addr_, size_, key_, flags_))
  , _key(_cnxn.get_memory_remote_key(_region))
  , _desc(_cnxn.get_memory_descriptor(_region))
{
}

registration::~registration()
try
{
  if ( _desc )
  {
    _cnxn.deregister_memory(_region);
  }
}
catch ( std::exception &e )
{
  std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
}
