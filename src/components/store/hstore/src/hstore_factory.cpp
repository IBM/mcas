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


#include "hstore.h"

#include "devdax_manager.h"

#include <common/json.h>
#include <common/utils.h>

#include <cstdlib> /* getenv */
#include <string>

using IKVStore = component::IKVStore;

/**
 * Factory entry point
 *
 */
extern "C" void * factory_createInstance(component::uuid_t component_id)
{
  return
    component_id == hstore_factory::component_id()
    ? new ::hstore_factory()
    : nullptr
    ;
}

void * hstore_factory::query_interface(component::uuid_t& itf_uuid)
{
  return itf_uuid == component::IKVStore_factory::iid()
     ? static_cast<component::IKVStore_factory *>(this)
     : nullptr
     ;
}

void hstore_factory::unload()
{
  delete this;
}

/*
 * See devdax_manager.cpp for the schema for the JSON "dax_map" parameter.
 */
auto hstore_factory::create(
  unsigned
  , const IKVStore_factory::map_create & mc
) -> component::IKVStore *
{
  auto debug_it = mc.find(+k_debug);
  auto owner_it = mc.find(+k_owner);
  auto name_it = mc.find(+k_name);
  auto dax_config_it = mc.find(+k_dax_config);

  namespace c_json = common::json;
  using json = c_json::serializer<c_json::dummy_writer>;
  unsigned debug_level = unsigned(debug_it == mc.end() ? 0 : std::stoul(debug_it->second));
  component::IKVStore *obj =
    new hstore(
      owner_it == mc.end() ? "owner" : owner_it->second
      , name_it == mc.end() ? "name" : name_it->second
      , std::make_unique<Devdax_manager>(debug_level, dax_config_it == mc.end() ? json::array().str() : dax_config_it->second, bool(std::getenv("DAX_RESET")))
    );
  obj->add_ref();

  return obj;
}
