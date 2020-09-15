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


#ifndef _FABRIC_H_
#define _FABRIC_H_

#include <api/fabric_itf.h> /* component::IFabric */
#include <common/moveable_value.h>
#include "event_producer.h"

#include "rdma-fabric.h" /* fid_t */
#include "rdma-fi_domain.h" /* fi_eq_attr */

#include <map>
#include <memory> /* shared_ptr */
#include <mutex>
#include <cstdlib> // getenv
#include <iosfwd> // ostream
#include <stdlib.h> // setenv, unsetenv

struct fi_info;
struct fid_fabric;
struct fid_eq;
struct fid;

struct env_replace
{
private:
	common::moveable_value<bool> _pre_exist;
	std::string _key;
public:
	env_replace(const char *key_, const char *value_)
		: _pre_exist(std::getenv(key_) != nullptr)
		, _key(key_)
	{
		::setenv(key_, value_, 0);
	}

	env_replace(env_replace &&) noexcept = default;

	~env_replace()
	{
		if ( ! _pre_exist )
		{
			::unsetenv(_key.c_str());
		}
	}
	const std::string key() const { return _key; }
	const char *value() const { return std::getenv(key().c_str()); }
};

std::ostream &operator<<(std::ostream &o, const env_replace &e);

/*
 * Note: Fabric is a fabric which can create servers (IFabric_server_factory) and clients (IFabric_op_completer)
 */
class Fabric
  : public component::IFabric
  , private event_producer
{
  env_replace _env_mr_cache_monitor;
  env_replace _env_use_odp;
  std::shared_ptr<::fi_info> _info;
  std::shared_ptr<::fid_fabric> _fabric;
  /* an event queue, in case the endpoint is connection-oriented */
  ::fi_eq_attr _eq_attr;
  std::shared_ptr<::fid_eq> _eq;
  int _fd;
  /* A limited number of fids use the event queue:
   *  - connection server factories created by open_server_factory will use it to
   *    - receive connection requests FI_CONNREQ)
   *  - connection clients use it to
   *    - be notified when their connection is accepted (FI_CONNECTION) or rejected (some error event)
   *    - be notified when their connection is shut down (FI_SHUTDOWN)
   *
   * These events are not expected:
   *   FI_NOTIFY (libfabric internal)
   *   FI_MR_COMPLETE (asymc MR)
   *   FI_AV_COMPLETE (async AV)
   *   FI_JOIN_COMPLETE (multicast join)
   */
  using eq_dispatch_t = std::map<::fid_t, event_consumer *>;
  /* Need to add active endpoint in passive endpoint callback, so use separate maps and separate locks */
  std::mutex _m_eq_dispatch_pep;
  eq_dispatch_t _eq_dispatch_pep;
  std::mutex _m_eq_dispatch_aep;
  eq_dispatch_t _eq_dispatch_aep;

  /* BEGIN component::IFabric */
  /**
   * @throw std::domain_error : json file parse-detected error
   * @throw fabric_runtime_error : std::runtime_error : ::fi_passive_ep fail
   * @throw fabric_runtime_error : std::runtime_error : ::fi_pep_bind fail
   * @throw fabric_runtime_error : std::runtime_error : ::fi_listen fail
   */
  component::IFabric_server_factory * open_server_factory(const std::string& json_configuration, std::uint16_t control_port) override;
  /**
   * @throw std::domain_error : json file parse-detected error
   * @throw bad_dest_addr_alloc : std::bad_alloc
   * @throw fabric_runtime_error : std::runtime_error : ::fi_domain fail
   * @throw fabric_bad_alloc : std::bad_alloc - libfabric allocation out of memory
   * @throw fabric_runtime_error : std::runtime_error : ::fi_connect fail
   *
   * @throw fabric_bad_alloc : std::bad_alloc - out of memory
   * @throw fabric_runtime_error : std::runtime_error : ::fi_ep_bind fail
   * @throw fabric_runtime_error : std::runtime_error : ::fi_enable fail
   * @throw fabric_runtime_error : std::runtime_error : ::fi_ep_bind fail (event registration)
   *
   * @throw std::logic_error : socket initialized with a negative value (from ::socket) in Fd_control
   * @throw std::logic_error : unexpected event
   * @throw std::system_error (receiving fabric server name)
   * @throw std::system_error : pselect fail (expecting event)
   * @throw std::system_error : resolving address
   *
   * @throw std::system_error : read error on event pipe
   * @throw std::system_error : pselect fail
   * @throw std::system_error : read error on event pipe
   * @throw fabric_bad_alloc : std::bad_alloc - libfabric out of memory (creating a new server)
   * @throw std::system_error - writing event pipe (normal callback)
   * @throw std::system_error - writing event pipe (readerr_eq)
   * @throw std::system_error - receiving data on socket
   */
  component::IFabric_client * open_client(const std::string& json_configuration, const std::string & remote, std::uint16_t control_port) override;
  /**
   * @throw std::domain_error : json file parse-detected error
   * @throw fabric_runtime_error : std::runtime_error : ::fi_passive_ep fail
   * @throw fabric_runtime_error : std::runtime_error : ::fi_pep_bind fail
   * @throw fabric_runtime_error : std::runtime_error : ::fi_listen fail
   */
  component::IFabric_server_grouped_factory * open_server_grouped_factory(const std::string& json_configuration, std::uint16_t control_port) override;
  /**
   * @throw std::domain_error : json file parse-detected error
   * @throw bad_dest_addr_alloc : std::bad_alloc
   * @throw fabric_runtime_error : std::runtime_error : ::fi_domain fail
   * @throw fabric_bad_alloc : std::bad_alloc - fabric allocation out of memory
   * @throw fabric_runtime_error : std::runtime_error : ::fi_connect fail
   *
   * @throw fabric_bad_alloc : std::bad_alloc - out of memory
   * @throw fabric_runtime_error : std::runtime_error : ::fi_ep_bind fail
   * @throw fabric_runtime_error : std::runtime_error : ::fi_enable fail
   * @throw fabric_runtime_error : std::runtime_error : ::fi_ep_bind fail (event registration)
   *
   * @throw std::logic_error : socket initialized with a negative value (from ::socket) in Fd_control
   * @throw std::logic_error : unexpected event
   * @throw std::system_error (receiving fabric server name)
   * @throw std::system_error : pselect fail (expecting event)
   * @throw std::system_error : resolving address
   *
   * @throw std::system_error : read error on event pipe
   * @throw std::system_error : pselect fail
   * @throw std::system_error : read error on event pipe
   * @throw fabric_bad_alloc : std::bad_alloc - libfabric out of memory (creating a new server)
   * @throw std::system_error - writing event pipe (normal callback)
   * @throw std::system_error - writing event pipe (readerr_eq)
   * @throw std::system_error - receiving data on socket
   */
  component::IFabric_client_grouped * open_client_grouped(const std::string& json_configuration, const std::string& remote_endpoint, std::uint16_t port) override;
  /* END component::IFabric */

  /* BEGIN event_producer */
  void register_pep(::fid_t ep, event_consumer &ec) override;
  void register_aep(::fid_t ep, event_consumer &ec) override;
  void deregister_endpoint(::fid_t ep) override;
  /**
   * @throw fabric_runtime_error : std::runtime_error : ::fi_ep_bind fail
   */
  void bind(::fid_ep &ep) override;
  /**
   * @throw fabric_runtime_error : std::runtime_error : ::fi_pep_bind fail
   */
  void bind(::fid_pep &ep) override;
  int fd() const override;
  /**
   * @throw fabric_runtime_error : std::runtime_error : ::fi_control fail
   * @throw std::system_error : pselect fail
   */
  void wait_eq() override;
  /**
   * @throw fabric_bad_alloc : std::bad_alloc - libfabric out of memory (creating a new server)
   * @throw std::system_error - writing event pipe (normal callback)
   * @throw std::system_error - writing event pipe (readerr_eq)
   */
  void read_eq() override;
  /* END event_producer */
  void readerr_eq();

  /* The help text does not say whether attr may be null, but the provider source expects that it is not. */
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_eq_open fail
   */
  std::shared_ptr<::fid_eq> make_fid_eq(::fi_eq_attr &attr, void *context) const;

public:
  /**
   * @throw fabric_bad_alloc : std::bad_alloc - out of memory
   * @throw std::domain_error : json file parse-detected error
   * @throw fabric_runtime_error : std::runtime_error : ::fi_getinfo fail
   * @throw fabric_runtime_error : std::runtime_error : ::fi_fabric fail
   * @throw fabric_runtime_error : std::runtime_error : ::fi_eq_open fail
   * @throw fabric_runtime_error : std::runtime_error : ::fi_control fail
   */
  explicit Fabric(const std::string& json_configuration);
  int trywait(::fid **fids, std::size_t count) const;
  /**
   * @throw fabric_runtime_error : std::runtime_error : ::fi_domain fail
   */
  std::shared_ptr<::fid_domain> make_fid_domain(::fi_info &info, void *context) const;
  /**
   * @throw fabric_runtime_error : std::runtime_error : ::fi_passive_ep fail
   */
  std::shared_ptr<::fid_pep> make_fid_pep(::fi_info &info, void *context) const;

  const char *prov_name() const noexcept override;

  std::uint16_t choose_port(std::uint16_t port);
};

#endif
