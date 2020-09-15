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

#include "fabric_transport.h"

#include <common/json.h>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/optional.hpp>

#include "connection_handler.h"

namespace
{
  const std::string  none_string{"(none)"};
  const std::string &optional_string(const boost::optional<std::string> &value) { return value ? *value : none_string; }

  const char *optional_print(const boost::optional<std::string> &value) { return optional_string(value).c_str(); }

  auto make_fabric( //
    const boost::optional<std::string> &src_addr_,
    const boost::optional<std::string> &fabric_prov_name_,
    const boost::optional<std::string> &domain_name_,
    bool                                debug
  ) -> component::IFabric *
  {
    using namespace component;
    namespace c_json = common::json;
    using json = c_json::serializer<c_json::dummy_writer>;

    if (debug) PLOG("Fabric: bound to (%s,%s)", optional_print(src_addr_), optional_print(domain_name_));

    /* FABRIC */
    auto i_fabric_factory =
      make_itf_ref(static_cast<IFabric_factory *>(load_component("libcomponent-fabric.so", net_fabric_factory)));

    if (!i_fabric_factory.get()) throw General_exception("unable to load Fabric Comanche component");

    auto fabric_spec2 =
      json::object(
        json::member("ep_attr", json::object(json::member("type", "FI_EP_MSG")))
        , json::member("tx_attr", json::object(json::member("inject_size", mcas::Fabric_transport::INJECT_SIZE)))
      );
    if ( fabric_prov_name_ )
    {
      fabric_spec2
        .append(
          json::member("fabric_attr", json::object(json::member("prov_name", *fabric_prov_name_)))
        )
        ;
    }
    if ( src_addr_ )
    {
      fabric_spec2
        .append(
          json::member("addr_format", "FI_ADDR_STR")
        )
        .append(
          json::member("src_addr", "fi_sockaddr_in://" + *src_addr_ + ":0")
        )
        ;
    }
    if ( domain_name_ )
    {
      fabric_spec2
        .append(
          json::member(
            "domain_attr"
            , json::object(
                json::member("name", *domain_name_)
                , json::member("threading", "FI_THREAD_SAFE")
              )
          )
        )
        ;
    }

    return i_fabric_factory->make_fabric(fabric_spec2.str());
  }

  auto make_server_factory(component::IFabric &fabric, uint16_t port) -> component::IFabric_server_factory *
  {
    namespace c_json = common::json;
    using json = c_json::serializer<c_json::dummy_writer>;
    const std::string server_factory_spec{json::object().str()};
    return fabric.open_server_factory(server_factory_spec, port);
  }
}  // namespace

namespace mcas
{
constexpr unsigned Fabric_transport::INJECT_SIZE;

Fabric_transport::Fabric_transport(const boost::optional<std::string> &fabric,
                   const boost::optional<std::string> &fabric_provider,
                   const boost::optional<std::string> &device,
                   unsigned                            port)
  : _fabric_debug(mcas::global::debug_level > 1),
    _fabric(make_fabric(fabric, fabric_provider, device, port)),
    _server_factory(make_server_factory(*_fabric, boost::numeric_cast<uint16_t>(port))),
    _port(port)
{
  if (_fabric_debug)
    PLOG("fabric_transport: (fabric=%s, provider=%s, device=%s, port=%u)", optional_print(fabric),
         optional_print(fabric_provider), optional_print(device), port);
}

auto Fabric_transport::get_new_connection() -> Connection_handler *
{
  auto connection = _server_factory->get_new_connection();
  return
    connection
    ? new Connection_handler(mcas::global::debug_level, _server_factory.get(), connection)
    : nullptr;
}

  
}  // namespace mcas
