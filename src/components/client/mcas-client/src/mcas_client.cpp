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
#include "mcas_client.h"

#include <api/fabric_itf.h>
#include <common/json.h>
#include <city.h>

#include "connection.h"
#include "protocol.h"

#include <regex>
#include <vector>

using namespace component;

namespace mcas
{
namespace global
{
unsigned debug_level = 3;
}
}  // namespace mcas

MCAS_client::MCAS_client(const unsigned                      debug_level,
                         const boost::optional<std::string> &src_device,
                         const boost::optional<std::string> &src_addr,
                         const boost::optional<std::string> &provider,
                         const std::string &                 dest_addr,
                         std::uint16_t                       port,
                         const unsigned                      patience_)
    : _debug(debug_level, this, dest_addr, port),
      _factory(load_factory()),
      _fabric(make_fabric(*_factory, src_addr, src_device, provider)),
      _transport(_fabric->open_client(common::json::serializer<common::json::dummy_writer>::object{}.str(), dest_addr, port)),
      _connection(std::make_unique<mcas::client::Connection_handler>(debug_level, _transport.get(), patience_)),
      _open_connection(*_connection)
{
}

Mcas_client_debug::Mcas_client_debug(const unsigned     debug_level,
                                     const void *       ths,
                                     const std::string &ip_addr,
                                     std::uint16_t      port)
{
  mcas::global::debug_level = debug_level;
  if (debug_level > 1) {
    PLOG("%s: protocol session: %p (%s) (%d)", __func__, ths, ip_addr.c_str(), port);
  }
}

Mcas_client_debug::~Mcas_client_debug() { PLOG("%s: closed fabric transport.", __func__); }

Open_connection::Open_connection(mcas::client::Connection_handler &_connection)
    : _open_cnxn((_connection.bootstrap(), &_connection))
{
  PLOG("%s %p", __func__, static_cast<void *>(this));
}

Open_connection::~Open_connection()
{
  if (_open_cnxn) {
    _open_cnxn->shutdown();
  }
  PLOG("%s %p", __func__, static_cast<void *>(this));
}

auto MCAS_client::load_factory() -> IFabric_factory *
{
  IBase *comp = load_component("libcomponent-fabric.so", net_fabric_factory);

  if (!comp) throw General_exception("Fabric component not found");

  auto factory = static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid()));
  assert(factory);
  return factory;
}

auto MCAS_client::make_fabric(component::IFabric_factory &        factory_,
                              const boost::optional<std::string> &src_addr_,
                              const boost::optional<std::string> &domain_name_,
                              const boost::optional<std::string> &fabric_prov_name_) -> IFabric *
{
  namespace c_json = common::json;
  using json = c_json::serializer<c_json::dummy_writer>;
  auto fabric_spec =
    json::object(
      json::member("ep_attr", json::object(json::member("type", "FI_EP_MSG")))
    );

    if ( fabric_prov_name_ )
    {
      fabric_spec
        .append(
          json::member("fabric_attr", json::object(json::member("prov_name", *fabric_prov_name_)))
        )
        ;
    }

    if ( src_addr_ )
    {
      fabric_spec
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
      fabric_spec
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

  return factory_.make_fabric(fabric_spec.str());
}

auto MCAS_client::make_fabric(component::IFabric_factory &factory_,
                              const std::string &  // ip_addr
                              ,
                              const std::string &provider,
                              const std::string &device) -> IFabric *
{
  namespace c_json = common::json;
  using json = c_json::serializer<c_json::dummy_writer>;
  auto mr_mode =
    json::array(
      provider == "sockets"
      ? json::array(
        "FI_MR_VIRT_ADDR"
        , "FI_MR_ALLOCATED"
        , "FI_MR_PROV_KEY"
      )
      : json::array(
          "FI_MR_LOCAL"
        , "FI_MR_VIRT_ADDR"
        , "FI_MR_ALLOCATED"
        , "FI_MR_PROV_KEY"
      )
    )
    ;

  auto fabric_spec =
    json::object(
      json::member("fabric_attr", json::object(json::member("prov_name", provider)))
      , json::member(
          "domain_attr"
          , json::object(
              json::member("mr_mode", std::move(mr_mode))
              , json::member("name", device)
            )
        )
      , json::member("ep_attr", json::object(json::member("type", "FI_EP_MSG")))
    );
  return factory_.make_fabric(fabric_spec.str());
}

int MCAS_client::thread_safety() const { return IKVStore::THREAD_MODEL_SINGLE_PER_POOL; }

int MCAS_client::get_capability(Capability cap) const
{
  switch (cap) {
    case Capability::POOL_DELETE_CHECK:
      return 1;
    case Capability::POOL_THREAD_SAFE:
      return 1;
    case Capability::RWLOCK_PER_POOL:
      return 1;
    default:
      return -1;
  }
}

IKVStore::pool_t MCAS_client::create_pool(const std::string &name,
                                          const size_t       size,
                                          uint32_t           flags,
                                          uint64_t           expected_obj_count)
{
  return _connection->create_pool(name, size, flags, expected_obj_count);
}

IKVStore::pool_t MCAS_client::open_pool(const std::string &name, uint32_t flags)
{
  return _connection->open_pool(name, flags);
}

status_t MCAS_client::close_pool(const IKVStore::pool_t pool)
{
  if (!pool) return E_INVAL;
  return _connection->close_pool(pool);
}

status_t MCAS_client::delete_pool(const std::string &name) { return _connection->delete_pool(name); }

status_t MCAS_client::delete_pool(IKVStore::pool_t pool) { return _connection->delete_pool(pool); }

status_t MCAS_client::configure_pool(const IKVStore::pool_t pool, const std::string &json)
{
  return _connection->configure_pool(pool, json);
}

status_t MCAS_client::put(const IKVStore::pool_t pool,
                          const std::string &    key,
                          const void *           value,
                          const size_t           value_len,
                          uint32_t               flags)
{
  assert(flags <= IMCAS::FLAGS_MAX_VALUE);
  return _connection->put(pool, key, value, value_len, flags);
}

status_t MCAS_client::put_direct(const pool_t           pool,
                                 const std::string &    key,
                                 const void *           value,
                                 const size_t           value_len,
                                 IMCAS::memory_handle_t handle,
                                 uint32_t               flags)
{
  return _connection->put_direct(pool, key.data(), key.size(), value, value_len, this, handle, flags);
}

status_t MCAS_client::async_put(IKVStore::pool_t   pool,
                                const std::string &key,
                                const void *       value,
                                size_t             value_len,
                                async_handle_t &   out_handle,
                                unsigned int       flags)
{
  return _connection->async_put(pool, key.data(), key.size(), value, value_len, out_handle, flags);
}

status_t MCAS_client::async_put_direct(const IKVStore::pool_t          pool,
                                       const std::string &             key,
                                       const void *                    value,
                                       const size_t                    value_len,
                                       async_handle_t &                out_handle,
                                       const IKVStore::memory_handle_t handle,
                                       const unsigned int              flags)
{
  return _connection->async_put_direct(pool, key.data(), key.size(), value, value_len, out_handle, this, handle, flags);
}

status_t MCAS_client::async_get_direct(IKVStore::pool_t          pool,
                                       const std::string &       key,
                                       void *                    value,
                                       size_t &                  value_len,
                                       async_handle_t &          out_handle,
                                       IKVStore::memory_handle_t handle)
{
  return _connection->async_get_direct(pool, key.data(), key.size(), value, value_len, out_handle, this, handle);
}

status_t MCAS_client::check_async_completion(async_handle_t &handle)
{
  return _connection->check_async_completion(handle);
}

status_t MCAS_client::get(const IKVStore::pool_t pool,
                          const std::string &    key,
                          void *&                out_value, /* release with free() */
                          size_t &               out_value_len)
{
  return _connection->get(pool, key, out_value, out_value_len);
}

status_t MCAS_client::get_direct(const pool_t           pool,
                                 const std::string &    key,
                                 void *                 out_value,
                                 size_t &               out_value_len,
                                 IMCAS::memory_handle_t handle)
{
  return _connection->get_direct(pool, key.data(), key.size(), out_value, out_value_len, this, handle);
}

status_t MCAS_client::get_direct_offset(const IMCAS::pool_t          pool,
                                        const offset_t               offset,
                                        size_t &                     length,
                                        void *const                  out_buffer,
                                        const IMCAS::memory_handle_t handle)
{
  return _connection->get_direct_offset(pool, offset, length, out_buffer, this, handle);
}

status_t MCAS_client::async_get_direct_offset(const IMCAS::pool_t          pool,
                                              const offset_t               offset,
                                              size_t &                     length,
                                              void *const                  out_buffer,
                                              async_handle_t &             out_handle,
                                              const IMCAS::memory_handle_t handle)
{
  return _connection->async_get_direct_offset(pool, offset, length, out_buffer, out_handle, this, handle);
}

status_t MCAS_client::put_direct_offset(const IMCAS::pool_t          pool,
                                        const offset_t               offset,
                                        size_t &                     length,
                                        const void *const            out_buffer,
                                        const IMCAS::memory_handle_t handle)
{
  return _connection->put_direct_offset(pool, offset, length, out_buffer, this, handle);
}

status_t MCAS_client::async_put_direct_offset(const IMCAS::pool_t          pool,
                                              const offset_t               offset,
                                              size_t &                     length,
                                              const void *const            out_buffer,
                                              async_handle_t &             out_handle,
                                              const IMCAS::memory_handle_t handle)
{
  return _connection->async_put_direct_offset(pool, offset, length, out_buffer, out_handle, this, handle);
}

component::IKVStore::memory_handle_t MCAS_client::register_direct_memory(void *vaddr, const size_t len)
{
  /* register_direct_memory is exposed at the interface. A failure of madvise
   * might well be expected if the parameters passed in at the interface are
   * incompatible with madvise, e.g. vaddr is not page-aligned. That might be
   * fixed by using On Demand Paging (ODP), which should remove the need for
   * madvise.
   */
  if (madvise(vaddr, len, MADV_DONTFORK) != 0) {
    PWRN("MCAS_client::%s: madvise MADV_DONTFORK failed (%p %lu) %s", __func__, vaddr, len, strerror(errno));
    assert(false);
  }

  return _connection->register_direct_memory(vaddr, len);
}

status_t MCAS_client::unregister_direct_memory(IKVStore::memory_handle_t handle)
{
  return _connection->unregister_direct_memory(handle);
}

status_t MCAS_client::erase(const IKVStore::pool_t pool, const std::string &key)
{
  return _connection->erase(pool, key);
}

status_t MCAS_client::async_erase(const IMCAS::pool_t pool, const std::string &key, async_handle_t &out_handle)
{
  return _connection->async_erase(pool, key, out_handle);
}

size_t MCAS_client::count(const IKVStore::pool_t pool) { return _connection->count(pool); }

status_t MCAS_client::get_attribute(const IKVStore::pool_t    pool,
                                    const IKVStore::Attribute attr,
                                    std::vector<uint64_t> &   out_attr,
                                    const std::string *       key)
{
  return _connection->get_attribute(pool, attr, out_attr, key);
}

status_t MCAS_client::get_statistics(Shard_stats &out_stats) { return _connection->get_statistics(out_stats); }

status_t MCAS_client::free_memory(void *p)
{
  ::free(p);
  return S_OK;
}

void MCAS_client::debug(const IKVStore::pool_t  // pool
                        ,
                        unsigned  // cmd
                        ,
                        uint64_t  // arg
)
{
}

status_t MCAS_client::find(const IKVStore::pool_t pool,
                           const std::string &    key_expression,
                           const offset_t         offset,
                           offset_t &             out_matched_offset,
                           std::string &          out_matched_key)
{
  return _connection->find(pool, key_expression, offset, out_matched_offset, out_matched_key);
}

status_t MCAS_client::invoke_ado(const IKVStore::pool_t            pool,
                                 const std::string &               key,
                                 const void *                      request,
                                 size_t                            request_len,
                                 const uint32_t                    flags,
                                 std::vector<IMCAS::ADO_response> &out_response,
                                 const size_t                      value_size)
{
  return _connection->invoke_ado(pool, key, request, request_len, flags, out_response, value_size);
}

status_t MCAS_client::async_invoke_ado(const IMCAS::pool_t        pool,
                                       const std::string &        key,
                                       const void *               request,
                                       const size_t               request_len,
                                       const ado_flags_t          flags,
                                       std::vector<ADO_response> &out_response,
                                       async_handle_t &           out_async_handle,
                                       const size_t               value_size)
{
  return _connection->invoke_ado_async(pool, key, request, request_len, flags, out_response, out_async_handle,
                                       value_size);
}

status_t MCAS_client::invoke_put_ado(const IKVStore::pool_t            pool,
                                     const std::string &               key,
                                     const void *                      request,
                                     size_t                            request_len,
                                     const void *                      value,
                                     size_t                            value_len,
                                     size_t                            root_len,
                                     ado_flags_t                       flags,
                                     std::vector<IMCAS::ADO_response> &out_response)
{
  return _connection->invoke_put_ado(pool, key, request, request_len, value, value_len, root_len, flags, out_response);
}

/**
 * Factory entry point
 *
 */
extern "C" void *factory_createInstance(component::uuid_t component_id)
{
  if (component_id == MCAS_client_factory::component_id()) {
    auto fact = new MCAS_client_factory();
    fact->add_ref();
    return static_cast<void *>(fact);
  }
  else {
    PWRN("%s: request for bad factory type", __func__);
    return NULL;
  }
}
