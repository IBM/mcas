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
#ifndef __MCAS_CONFIG_FILE_H__
#define __MCAS_CONFIG_FILE_H__

#include <boost/optional.hpp>

#include <common/exceptions.h>
#include <common/logging.h> /* log_source */
#include <rapidjson/document.h>
#include <stdio.h>

#include <map>
#include <stdexcept>  // std::runtime_error
#include <string>
#include <vector>

class Config_exception : public Exception {
 public:
  Config_exception(int err) : Exception("Config exception"), _err_code(err) {}

  Config_exception() : _err_code(E_FAIL) {}

  __attribute__((__format__(__printf__, 2, 0))) Config_exception(const char *fmt, ...)
      : Exception("Config exception"),
        _err_code(E_FAIL)
  {
    va_list args;
    va_start(args, fmt);
    char msg[255] = {0};
    vsnprintf(msg, 254, fmt, args);
    set_cause(msg);
  }
  status_t error_code() const { return _err_code; }

 private:
  status_t _err_code;
};

namespace config
{
  static constexpr const char *default_backend = "default_backend";
  static constexpr const char *index = "index";
  static constexpr const char *addr = "addr";
  static constexpr const char *net = "net";
}

namespace mcas
{
class Config_file : private common::log_source {
 public:
  Config_file(unsigned debug_level_, const std::string &config_spec);

  Config_file(unsigned debug_level_, rapidjson::Document &&doc);

  rapidjson::SizeType shard_count() const { return _shards.Size(); }

  std::string get_ado_cores() const;

  int get_ado_manager_core() const;

  auto get_shard(rapidjson::SizeType i) const;

  std::string get_shard_ado_cores(rapidjson::SizeType i) const;

  float get_shard_ado_core_number(rapidjson::SizeType i) const;

  unsigned int get_shard_core(rapidjson::SizeType i) const;

  unsigned int get_shard_port(rapidjson::SizeType i) const;

  boost::optional<std::string> get_shard_optional(std::string field, rapidjson::SizeType i) const;

  std::string get_shard_required(std::string field, rapidjson::SizeType i) const;

  boost::optional<std::string> get_net_providers() const { return _net_providers; };

  const boost::optional<std::string> &get_ado_path() const { return _ado_path; };

  std::vector<std::string> get_shard_ado_plugins(rapidjson::SizeType i) const;

  std::map<std::string, std::string> get_shard_ado_params(rapidjson::SizeType i) const;

  auto get_shard_object(std::string name, rapidjson::SizeType i) const;

  boost::optional<rapidjson::Document> get_shard_dax_config_raw(rapidjson::SizeType i);

  std::string get_cert_path() const;

  std::string cluster_group() const;

  std::string cluster_local_name() const;

  std::string cluster_ip_addr() const;

  unsigned int cluster_net_port() const;

  unsigned int debug_level() const;

 private:
  rapidjson::Document          _doc;
  rapidjson::Value             _shards;
  boost::optional<std::string> _net_providers;
  boost::optional<std::string> _ado_path;
  rapidjson::Value             _resources;
  rapidjson::Value             _security;
  rapidjson::Value             _cluster;
};

}  // namespace mcas
#endif  // __mcas_CONFIG_FILE_H__
