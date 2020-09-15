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
#include <common/logging.h>
#include <common/moveable_ptr.h>
#include <common/net.h>
#include <common/delete_copy.h>
#include <unistd.h>
#include <api/cluster_itf.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <csignal>

#include "ado_manager.h"
#include "launcher.h"
#include "mcas_config.h"
#include "config_file.h"

Program_options g_options{};

namespace
{
/* std::ostringstream initializes locale on first use. To avoid race conditions,
 * force a first use which precedes thread creation.
 */
void init_locale()
{
  std::ostringstream s;
  s << 0;
}

void global_signal_handler(int signal)
{
  /*
   * From IEEE Std 1003.1-2017 (POSIX): "the behavior is undefined if the
   * signal handler refers to any object other than errno with static storage
   * duration other than by assigning a value to an object declared as volatile
   * sig_atomic_t ..."
   */
  PLOG("signal: (%d)", signal);
  switch (signal)
  {
  case SIGINT:
    ;
#if 201703L <= __cplusplus
    [[fallthrough]]
#endif
  case SIGTERM:
    signals::sigint = 1;
    break;
  default:
    ;
  }
}

}  // namespace

struct zyre_run
{
private:
  common::moveable_ptr<component::ICluster> _zyre;

  /*< timeout in ms for a node to be considered evasive */
  static constexpr unsigned NODE_EVASIVE_TIMEOUT = 2000;

  /*< timeout in ms for a node to be considered expired */
  static constexpr unsigned NODE_EXPIRED_TIMEOUT = 3000;
public:
  zyre_run(component::ICluster *zyre_) : _zyre(zyre_)
  {
    if ( _zyre )
    {
      _zyre->start_node();
      _zyre->set_timeout(component::ICluster::Timeout_type::EVASIVE, NODE_EVASIVE_TIMEOUT);
      _zyre->set_timeout(component::ICluster::Timeout_type::EXPIRED, NODE_EXPIRED_TIMEOUT);
    }
  }
  zyre_run(zyre_run &&) noexcept = default;
  ~zyre_run() { if (_zyre) { _zyre->stop_node(); } }
};

struct zyre_group_membership
{
private:
  common::moveable_ptr<component::ICluster> _zyre;
  std::string _group_name;
public:
  zyre_group_membership(component::ICluster *zyre_, std::string group_name_)
    : _zyre(zyre_)
    , _group_name(group_name_)
  {
    if ( _zyre )
    {
      _zyre->group_join(_group_name);
    }
  }
  zyre_group_membership(zyre_group_membership &&) noexcept = default;
  ~zyre_group_membership() { if (_zyre) { _zyre->group_leave(_group_name); } }
};

int main(int argc, char *argv[])
{
  namespace po = boost::program_options;

  try {
    init_locale();
    po::options_description desc("Options");
// clang-format off
    desc.add_options()
      ("help", "Show help")
      ("config", po::value<std::string>(), "Configuration file or JSON configuration string")
      ("debug", po::value<unsigned>()->default_value(0), "Debug level 0-3")
      ("forced-exit", "Exit when the number of clients transtions from non-zero to zero")
      ("device", po::value<std::string>()->default_value("mlx5_0"), "Network device (e.g., mlx5_0)")
      ("profile", po::value<std::string>(), "profile file for main loop")
      ("triggered-profile", "Profile, if specified, is triggered by first get_attribute(COUNT) operation");
// clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help") > 0) {
      std::cout << desc;
      return -1;
    }

    if (vm.count("config") == 0) {
      std::cout << "--config option is required\n";
      return -1;
    }

    g_options.config            = vm["config"].as<std::string>();
    g_options.device            = vm["device"].as<std::string>();
    g_options.forced_exit       = vm.count("forced-exit");
    g_options.triggered_profile = vm.count("triggered-profile");
    g_options.profile_file_main = vm.count("profile") ? vm["profile"].as<std::string>() : "";

    PLOG("forced-exit:%s", g_options.forced_exit ? "yes" : "no");

    mcas::global::debug_level = g_options.debug_level = vm["debug"].as<unsigned>();

    std::unique_ptr<ADO_manager> mgr;

    try {
      mgr = std::make_unique<ADO_manager>(g_options);
    }
    catch (const Config_exception &e) {
      std::cout << "error: bad configuration file: " << e.cause() << "\n";
      return -1;
    }

    mcas::Config_file config(g_options.debug_level, g_options.config);

    component::Itf_ref<component::ICluster> zyre;

    auto group = config.cluster_group();

    if (group.empty() == false) {
      PMAJOR("MCAS: clustering-enabled: %s %s %d", group.c_str(), config.cluster_ip_addr().c_str(),
             config.cluster_net_port());

      using namespace component;

      IBase *comp = component::load_component("libcomponent-zyre.so", cluster_zyre_factory);

      auto factory = make_itf_ref(static_cast<ICluster_factory *>(comp->query_interface(ICluster_factory::iid())));

      if (!factory) throw Logic_exception("unable to create Zyre factory");

      auto local_node_name = config.cluster_local_name();

      std::string net_interface = common::get_eth_device_from_ip(config.cluster_ip_addr());
      if(net_interface.empty())
        throw General_exception("cannot find interface for %s", config.cluster_ip_addr().c_str());

      /* if the node name is not specified in the cluster section,
         then derive a node name from hostname, net, port
      */
      if (local_node_name.empty()) {
        std::array<char,255> host_name;
        ::gethostname(&host_name[0], host_name.size());

        std::ostringstream node_name;
        node_name << "mcas-" << &host_name[0] << "-" << net_interface << "-" << config.cluster_net_port();
        local_node_name = node_name.str();
      }

      PLOG("Starting Zyre node: %s", local_node_name.c_str());

      zyre = make_itf_ref(factory->create(g_options.debug_level,
                                            local_node_name,
                                            net_interface,
                                            config.cluster_net_port()));

      if (!zyre) throw Logic_exception("unable to create Zyre component instance");
    }

    zyre_run zr(zyre.get());
    zyre_group_membership zg(zyre.get(), group);

    /* launch shards */
    try
    {
      auto launcher = std::make_unique<mcas::Shard_launcher>(g_options);

      for ( auto sig : { SIGINT, SIGTERM } )
      {
        if (signal(sig, global_signal_handler) == SIG_ERR)
          throw General_exception("signal call failed");
      }

      std::string msg_sender_uuid;
      std::string msg_type;
      std::string msg_content;
      std::vector<std::string> values;

      while (launcher->threads_running()) {
        if (zyre) {
          while (zyre->poll_recv(msg_sender_uuid, msg_type, msg_content, values)) {

            std::stringstream ss;
            ss << msg_content;
            int added = 0;
            for(auto& v : values) {
              ss << "," << v;
              if(++added == 2) break; /* only take two fields  */
            }

            PMAJOR("MCAS server got message (uuid=%s type=%s content=%s)",
                   msg_sender_uuid.c_str(), msg_type.c_str(), ss.str().c_str());

            if(ss.str() != "MCAS-AVAIL") { /* don't pass up heartbeats */
              launcher->send_cluster_event(msg_sender_uuid, /* translate? */
                                             msg_type,
                                             ss.str());
            }

          }

          if (g_options.debug_level > 3) zyre->dump_info();

          /* sent heartbeat */
         zyre->shout(group,"X-MSG", "MCAS-AVAIL");
        }

        /* should be coordinated with zyre heartbeat */
        sleep(1);
      }

      launcher->wait_for_all();
    }
    catch ( ... )
    {
    }

    PMAJOR("MCAS: all shards shut down gracefully.");
    sleep(3);
  }
  catch (const po::error &) {
    printf("bad command line option\n");
    return -1;
  }

  return 0;
}
