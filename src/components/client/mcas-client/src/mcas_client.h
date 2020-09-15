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

/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __MCAS_CLIENT_COMPONENT_H__
#define __MCAS_CLIENT_COMPONENT_H__

#include "connection.h"
#include "mcas_client_config.h"

#include <api/components.h>
#include <api/fabric_itf.h>
#include <api/kvindex_itf.h>
#include <api/kvstore_itf.h>
#include <api/mcas_itf.h>
#include <api/itf_ref.h>
#include <common/moveable_ptr.h>

#include <boost/optional.hpp>

#include <cstdint> /* uint16_t */
#include <memory>  /* unique_ptr */
#include <string>

class Mcas_client_debug {
 public:
  Mcas_client_debug() {}
  Mcas_client_debug(unsigned debug_level, const void *ths, const std::string &ip_addr, std::uint16_t port);
  ~Mcas_client_debug();
};

class Open_connection {
  common::moveable_ptr<mcas::client::Connection_handler> _open_cnxn;

 public:
  Open_connection() : _open_cnxn(nullptr) {}
  Open_connection(mcas::client::Connection_handler &_connection);
  Open_connection(Open_connection &&) noexcept = default;
  ~Open_connection();
};

class MCAS_client
    : public virtual component::IKVStore
    , public virtual component::IMCAS {
  friend class MCAS_client_factory;

 private:
  static constexpr bool option_DEBUG = true;

 protected:
  /**
   * Constructor
   *
   * @param debug_level Debug level (e.g., 0-3)
   * @param owner Owner information (not used)
   * @param addr_port_str Address and port info (e.g. 10.0.0.22:11911)
   * @param device NIC device (e.g., mlx5_0)
   * @param provider fabric provider ("verbs" or "sockets")
   *
   */
 public:
  MCAS_client(unsigned                            debug_level,  // temp srd/dest devider
              const boost::optional<std::string> &src_device,
              const boost::optional<std::string> &src_addr,
              const boost::optional<std::string> &provider,
              const std::string &                 dest_addr,
              std::uint16_t                       port,
              unsigned                            patience);

  MCAS_client(const MCAS_client &) = delete;
  MCAS_client &operator=(const MCAS_client &) = delete;

  using pool_t = component::IKVStore::pool_t;

  /**
   * Component/interface management
   *
   */
  DECLARE_VERSION(0.1f);

  // clang-format off
  DECLARE_COMPONENT_UUID(0x2f666078, 0xcb8a, 0x4724, 0xa454, 0xd1, 0xd8, 0x8d, 0xe2, 0xdb, 0x87);
  // clang-format on

  void *query_interface(component::uuid_t &itf_uuid) override
  {
    if (itf_uuid == component::IKVStore::iid()) {
      return static_cast<component::IKVStore *>(this);
    }
    else if (itf_uuid == component::IMCAS::iid()) {
      return static_cast<component::IMCAS *>(this);
    }
    else {
      return NULL;  // we don't support this interface
    }
  }

  void unload() override { delete this; }

 public:
  /* IKVStore (as remote proxy) */
  virtual int thread_safety() const override;

  virtual int get_capability(Capability cap) const override;

  virtual pool_t create_pool(const std::string &name,
                             const size_t       size,
                             const unsigned int flags              = 0,
                             const uint64_t     expected_obj_count = 0) override;

  virtual pool_t open_pool(const std::string &name, const unsigned int flags = 0) override;

  virtual status_t close_pool(const pool_t pool) override;

  virtual status_t delete_pool(const std::string &name) override;

  virtual status_t delete_pool(const IKVStore::pool_t pool) override;

  virtual status_t configure_pool(const component::IKVStore::pool_t pool, const std::string &json) override;

  virtual status_t put(const pool_t       pool,
                       const std::string &key,
                       const void *       value,
                       const size_t       value_len,
                       const unsigned int flags = IMCAS::FLAGS_NONE) override;

  virtual status_t put_direct(const pool_t                 pool,
                              const std::string &          key,
                              const void *                 value,
                              const size_t                 value_len,
                              const IMCAS::memory_handle_t handle = IMCAS::MEMORY_HANDLE_NONE,
                              const unsigned int           flags  = IMCAS::FLAGS_NONE) override;

  virtual status_t async_put(const IKVStore::pool_t pool,
                             const std::string &    key,
                             const void *           value,
                             const size_t           value_len,
                             async_handle_t &       out_handle,
                             const unsigned int     flags = IMCAS::FLAGS_NONE) override;

  virtual status_t async_put_direct(const IKVStore::pool_t          pool,
                                    const std::string &             key,
                                    const void *                    value,
                                    const size_t                    value_len,
                                    async_handle_t &                out_handle,
                                    const IKVStore::memory_handle_t handle = IMCAS::MEMORY_HANDLE_NONE,
                                    const unsigned int              flags  = IMCAS::FLAGS_NONE) override;

  virtual status_t check_async_completion(async_handle_t &handle) override;

  virtual status_t get(const pool_t       pool,
                       const std::string &key,
                       void *&            out_value, /* release with free() */
                       size_t &           out_value_len) override;

  virtual status_t async_get_direct(const pool_t                 pool,
                              const std::string &          key,
                              void *                       out_value,
                              size_t &                     out_value_len,
                              async_handle_t &          out_handle,
                              const IMCAS::memory_handle_t handle = IMCAS::MEMORY_HANDLE_NONE) override;

  virtual status_t get_direct(const pool_t                 pool,
                              const std::string &          key,
                              void *                       out_value,
                              size_t &                     out_value_len,
                              const IMCAS::memory_handle_t handle = IMCAS::MEMORY_HANDLE_NONE) override;

  virtual status_t get_direct_offset(const IMCAS::pool_t          pool,
                                     const offset_t               offset,
                                     size_t &                     length,
                                     void *                       out_buffer,
                                     const IMCAS::memory_handle_t handle) override;

  virtual status_t async_get_direct_offset(const IMCAS::pool_t          pool,
                                           const offset_t               offset,
                                           size_t &                     length,
                                           void *                       out_buffer,
                                           async_handle_t &             out_handle,
                                           const IMCAS::memory_handle_t handle = IMCAS::MEMORY_HANDLE_NONE) override;

  virtual status_t put_direct_offset(const IMCAS::pool_t          pool,
                                     const offset_t               offset,
                                     size_t &                     length,
                                     const void *                 out_buffer,
                                     const IMCAS::memory_handle_t handle) override;

  virtual status_t async_put_direct_offset(const IMCAS::pool_t          pool,
                                           const offset_t               offset,
                                           size_t &                     length,
                                           const void *                 out_buffer,
                                           async_handle_t &             out_handle,
                                           const IMCAS::memory_handle_t handle = IMCAS::MEMORY_HANDLE_NONE) override;

  virtual status_t erase(const pool_t pool, const std::string &key) override;

  virtual status_t async_erase(const IMCAS::pool_t pool, const std::string &key, async_handle_t &out_handle) override;

  virtual size_t count(const pool_t pool) override;

  virtual status_t get_attribute(const IKVStore::pool_t    pool,
                                 const IKVStore::Attribute attr,
                                 std::vector<uint64_t> &   out_attr,
                                 const std::string *       key) override;

  virtual status_t get_statistics(Shard_stats &out_stats) override;

  virtual void debug(const pool_t pool, const unsigned cmd, const uint64_t arg) override;

  virtual IMCAS::memory_handle_t register_direct_memory(void *vaddr, const size_t len) override;

  virtual status_t unregister_direct_memory(const IMCAS::memory_handle_t handle) override;

  virtual status_t free_memory(void *p) override;

  /* IMCAS specific methods */
  virtual status_t find(const IKVStore::pool_t pool,
                        const std::string &    key_expression,
                        const offset_t         offset,
                        offset_t &             out_matched_offset,
                        std::string &          out_matched_key) override;

  virtual status_t invoke_ado(const IKVStore::pool_t            pool,
                              const std::string &               key,
                              const void *                      request,
                              const size_t                      request_len,
                              const uint32_t                    flags,
                              std::vector<IMCAS::ADO_response> &out_response,
                              const size_t                      value_size = 0) override;

  virtual status_t async_invoke_ado(const IMCAS::pool_t               pool,
                                    const std::string &               key,
                                    const void *                      request,
                                    const size_t                      request_len,
                                    const ado_flags_t                 flags,
                                    std::vector<IMCAS::ADO_response> &out_response,
                                    async_handle_t &                  out_async_handle,
                                    const size_t                      value_size = 0) override;

  virtual status_t invoke_put_ado(const IKVStore::pool_t            pool,
                                  const std::string &               key,
                                  const void *                      request,
                                  const size_t                      request_len,
                                  const void *                      value,
                                  const size_t                      value_len,
                                  const size_t                      root_len,
                                  const ado_flags_t                 flags,
                                  std::vector<IMCAS::ADO_response> &out_response) override;

 private:
  Mcas_client_debug                                 _debug;
  component::Itf_ref<component::IFabric_factory>    _factory;
  std::unique_ptr<component::IFabric>               _fabric;
  std::unique_ptr<component::IFabric_client>        _transport;
  std::unique_ptr<mcas::client::Connection_handler> _connection;
  Open_connection                                   _open_connection;

 private:
  static void set_debug(unsigned debug_level, const void *ths, const std::string &ip_addr, std::uint16_t port);
  static auto load_factory() -> component::IFabric_factory *;
  static auto make_fabric(component::IFabric_factory &,
                          const std::string &ip_addr,
                          const std::string &provider,
                          const std::string &device) -> component::IFabric *;
  static auto make_fabric(component::IFabric_factory &,
                          const boost::optional<std::string> &src_addr,
                          const boost::optional<std::string> &interface,
                          const boost::optional<std::string> &provider) -> component::IFabric *;

  void open_transport(const std::string &device,
                      const std::string &ip_addr,
                      const int          port,
                      const std::string &provider);
};

class MCAS_client_factory : public component::IMCAS_factory {
 public:
  /**
   * Component/interface management
   *
   */
  DECLARE_VERSION(0.1f);

  // clang-format off
  DECLARE_COMPONENT_UUID(0xfac66078, 0xcb8a, 0x4724, 0xa454, 0xd1, 0xd8, 0x8d, 0xe2, 0xdb, 0x87);
  // clang-format on

  void *query_interface(component::uuid_t &itf_uuid) override
  {
    if (itf_uuid == component::IMCAS_factory::iid()) {
      return static_cast<component::IMCAS_factory *>(this);
    }
    else if (itf_uuid == component::IKVStore_factory::iid()) {
      return static_cast<component::IKVStore_factory *>(this);
    }
    else
      return NULL;  // we don't support this interface
  }

  void unload() override { delete this; }

  component::IMCAS *mcas_create(unsigned                            debug_level,
      unsigned patience,
                                const std::string &                 owner,
                                const boost::optional<std::string> &src_nic_device,
                                const boost::optional<std::string> &src_ip_addr,
                                const std::string &                 dest_addr_with_port) override;

  component::IKVStore *create(unsigned           debug_level,
                              const std::string &owner,
                              const std::string &addr,
                              const std::string &device) override;

  component::IKVStore *create(unsigned debug_level, const std::map<std::string, std::string> &) override;
};

#endif
