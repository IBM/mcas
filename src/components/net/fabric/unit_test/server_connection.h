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
#ifndef _TEST_SERVER_CONNECTION_H_
#define _TEST_SERVER_CONNECTION_H_

#include <common/delete_copy.h>
#include <common/moveable_ptr.h>

namespace component
{
  class IFabric_server;
  class IFabric_server_factory;
}

struct server_connection
{
private:
  component::IFabric_server_factory *_ep;
  common::moveable_ptr<component::IFabric_server> _cnxn;
  DELETE_COPY(server_connection);
  static component::IFabric_server *get_connection(component::IFabric_server_factory &ep);
public:
  component::IFabric_server &cnxn() const { return *_cnxn; }
  explicit server_connection(component::IFabric_server_factory &ep);
  server_connection(server_connection &&) noexcept = default;
  server_connection& operator=(server_connection &&) noexcept = default;
  /* The presence of a destructor and a pointer member causes -Weffc++ to warn
   *
   * warning: ‘class d’ has pointer data members [-Weffc++]
   * warning:   but does not override ‘d(const d&)’ [-Weffc++]
   * warning:   or ‘operator=(const d&)’ [-Weffc++]
   *
   * g++ should not warn in this case, because the declarataion of a move constructor suppresses
   * default copy constructor and operator=.
   */
  ~server_connection();
};

#endif
