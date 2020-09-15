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
#ifndef _TEST_SERVERC_GROUPED_CONNECTION_H_
#define _TEST_SERVERC_GROUPED_CONNECTION_H_

#include <common/delete_copy.h>

namespace component
{
  class IFabric_server_grouped_factory;
  class IFabric_server_grouped;
  class IFabric_communicator;
}

struct server_grouped_connection
{
private:
  component::IFabric_server_grouped_factory &_ep;
  /* ERROR: these two ought to be shared_ptr, with appropriate destructors */
  component::IFabric_server_grouped *_cnxn;
  component::IFabric_communicator *_comm;

  server_grouped_connection(server_grouped_connection &&) noexcept;
  DELETE_COPY(server_grouped_connection);
  static component::IFabric_server_grouped *get_connection(component::IFabric_server_grouped_factory &ep);

public:
  component::IFabric_server_grouped &cnxn() const { return *_cnxn; }
  server_grouped_connection(component::IFabric_server_grouped_factory &ep);
  ~server_grouped_connection();
  component::IFabric_communicator &comm() const { return *_comm; }
  component::IFabric_communicator *allocate_group() const;
};

#endif
