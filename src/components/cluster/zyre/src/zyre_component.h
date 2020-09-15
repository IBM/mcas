/*
   Copyright [2018] [IBM Corporation]

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

#ifndef __ZYRE_COMPONENT_H__
#define __ZYRE_COMPONENT_H__

#include <zyre.h>
#include <component/base.h>
#include <common/logging.h> /* log_source */
#include <api/cluster_itf.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"

class Zyre_component : public component::ICluster, private common::log_source
{
private:
  static constexpr unsigned HEARTBEAT_INTERVAL_MS = 500;

public:
  /**
   * Constructor
   *
   *
   */
  Zyre_component(const unsigned debug_level_,
                 const std::string& node_name,
                 const std::string& nic,
                 const unsigned int port);

  /**
   * Destructor
   *
   */
  virtual ~Zyre_component();

  /**
   * Component/interface management
   *
   */
  DECLARE_VERSION(0.1f);
  DECLARE_COMPONENT_UUID(0x5d19463b,0xa29d,0x4bc1,0x989c,0xbe,0x74,0x0a,0xc2,0x79,0x10);

  void * query_interface(component::uuid_t& itf_uuid) override {
    if(itf_uuid == component::ICluster::iid()) {
      return static_cast<component::ICluster*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

public:

  /* ICluster interface */
  virtual void start_node() override;
  virtual void stop_node() override;
  virtual void destroy_node() override;
  virtual std::string uuid() const override;
  virtual std::string node_name() const override;
  virtual void dump_info() const override;
  virtual void group_join(const std::string& group) override;
  virtual void group_leave(const std::string& group) override;
  virtual void shout(const std::string& group, const std::string& type, const std::string& message) override;
  virtual void whisper(const std::string& peer_uuid, const std::string& type, const std::string& message) override;
  virtual bool poll_recv(std::string& sender_uuid, std::string& type, std::string& message, std::vector<std::string>& values) override;
  virtual status_t set_timeout(const Timeout_type type, const int interval_ms) override;
  virtual std::string peer_address(const std::string& uuid) override;


private:
  zyre_t * _node;

};


class Zyre_component_factory : public component::ICluster_factory
{
public:

  /**
   * Component/interface management
   *
   */
  DECLARE_VERSION(0.1f);
  DECLARE_COMPONENT_UUID(0xfac9463b,0xa29d,0x4bc1,0x989c,0xbe,0x74,0x0a,0xc2,0x79,0x10);

  void * query_interface(component::uuid_t& itf_uuid) override {
    if(itf_uuid == component::ICluster_factory::iid()) {
      return static_cast<component::ICluster_factory*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

  virtual component::ICluster * create(const unsigned debug_level_,
                                       const std::string& node_name,
                                       const std::string& nic,
                                       const unsigned int port) override
  {
    component::ICluster * obj = static_cast<component::ICluster*>
      (new Zyre_component(debug_level_, node_name, nic, port));

    obj->add_ref();
    return obj;
  }

};


#pragma GCC diagnostic pop

#endif
