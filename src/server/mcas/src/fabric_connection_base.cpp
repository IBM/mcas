/*
   Copyright [2017-2020] [IBM Corporation]
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

#include "fabric_connection_base.h"

namespace mcas
{
Fabric_connection_base::Fabric_connection_base(unsigned debug_level_,
                                               gsl::not_null<component::IFabric_server_factory *> factory,
                                               gsl::not_null<component::IFabric_server *> fabric_connection)
  : common::log_source(debug_level_),
    _oc(factory, fabric_connection, open_connection_construct_key{}),
    _bm(debug_level_, fabric_connection),
    _max_message_size(transport()->max_message_size()),
    _deferred_unlock{},
    _recv_buffer_posted_count{},
    _completed_recv_buffers{},
    _send_buffer_posted_count{},
    _send_value_posted_count{}
{
}

/**
 * Dtor
 *
 */
Fabric_connection_base::~Fabric_connection_base()
{
  /* close connection */
  //  _factory->close_connection(_transport);
}

}  // namespace mcas
