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
#include "server_connection.h"

#include "eyecatcher.h"
#include <api/fabric_itf.h> /* IFabric_server_factory, IFabric_server */
#include <exception>
#include <iostream> /* cerr */

component::IFabric_server *server_connection::get_connection(component::IFabric_server_factory &ep_)
{
  component::IFabric_server *cnxn = nullptr;
  while ( ! ( cnxn = ep_.get_new_connection() ) ) {}
  return cnxn;
}

server_connection::server_connection(component::IFabric_server_factory &ep_)
  : _ep(&ep_)
  , _cnxn(get_connection(*_ep))
{
}

server_connection::~server_connection()
try
{
  if ( _cnxn )
  {
    _ep->close_connection(_cnxn);
  }
}
catch ( std::exception &e )
{
  std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
}
