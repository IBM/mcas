/*
  Copyright [2017-2021] [IBM Corporation]
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

#ifndef __API_KVSTORE_ITF__
#define __API_KVSTORE_ITF__

#include <api/components.h>
#include <api/kvstore.h>
#include <common/string_view.h>

namespace component
{

#define DECLARE_OPAQUE_TYPE(NAME) \
  struct Opaque_##NAME {          \
    virtual ~Opaque_##NAME() {}   \
  }

/**
 * Key-value interface for pluggable backend (e.g. mapstore, hstore, hstore-cc)
 */
class IKVStore : public IBase,
                 public KVStore
{
 public:
  // clang-format off
  DECLARE_INTERFACE_UUID(0x62f4829f,0x0405,0x4c19,0x9898,0xa3,0xae,0x21,0x5a,0x3e,0xe8);
  // clang-format on
};


class IKVStore_factory : public component::IBase {
  using string_view = common::string_view;
 public:
  // clang-format off
  DECLARE_INTERFACE_UUID(0xface829f,0x0405,0x4c19,0x9898,0xa3,0xae,0x21,0x5a,0x3e,0xe8);
  // clang-format on

  virtual IKVStore* create(const std::string& owner, const std::string&)
  {
    (void)owner;
    throw API_exception("IKVstore_factory::create(owner,param) not implemented");
  };

  virtual IKVStore* create(const std::string& owner, const std::string&, const std::string&)
  {
    (void)owner;
    throw API_exception("IKVstore_factory::create(owner,param,param2) not implemented");
  }

  virtual IKVStore* create(unsigned debug_level,
                           string_view owner,
                           string_view,
                           string_view)
  {
    (void)debug_level;
    (void)owner;
    throw API_exception("IKVstore_factory::create(debug_level,owner,param,param2) not implemented");
  }

  using map_create = std::map<std::string, std::string>;

  static constexpr const char *k_src_addr = "src_addr";
  static constexpr const char *k_dest_addr = "dest_addr";
  static constexpr const char *k_dest_port = "dest_port";
  static constexpr const char *k_interface = "interface";
  static constexpr const char *k_provider = "provider";
  static constexpr const char *k_patience = "patience";

  static constexpr const char *k_debug = "debug";
  static constexpr const char *k_owner = "owner";
  static constexpr const char *k_name = "name";
  static constexpr const char *k_dax_config = "dax_config";
  static constexpr const char *k_mm_plugin_path = "mm_plugin_path";

  /* this is the preferred create method - the others will be deprecated */
  virtual IKVStore* create(unsigned debug_level, const map_create& params)
  {
    (void)debug_level;
    (void)params;
    throw API_exception("IKVstore_factory::create(debug_level,param-map) not implemented");
  }
};

}  // namespace component
#endif
