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
#ifndef __FABRIC_TRANSPORT_H__
#define __FABRIC_TRANSPORT_H__

#include <api/components.h>
#include <api/fabric_itf.h>

#include <boost/optional.hpp>

#include "buffer_manager.h"
#include <memory>  // unique_ptr
#include <string>

namespace mcas
{
class Connection_handler;

class Fabric_transport {
  bool _fabric_debug;

 public:
  static constexpr unsigned INJECT_SIZE = 128;

  using memory_region_t = component::IFabric_memory_region *;
  using buffer_t        = Buffer_manager<component::IFabric_server>::buffer_internal;

  Fabric_transport(const boost::optional<std::string> &fabric,
                   const boost::optional<std::string> &fabric_provider,
                   const boost::optional<std::string> &device,
                   unsigned                            port);

  Connection_handler *get_new_connection();

  inline unsigned get_port() const { return _port; }

 private:
  std::unique_ptr<component::IFabric>                _fabric;
  std::unique_ptr<component::IFabric_server_factory> _server_factory;
  unsigned                                           _port;
};

}  // namespace mcas

#endif  // __FABRIC_TRANSPORT_H__
