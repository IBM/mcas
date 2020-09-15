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

#include "config_file.h"

#include "rapidjson/error/error.h"  // rapidjson::ParseErrorCode
namespace mcas
{
  [[noreturn]] void throw_parse_exception(rapidjson::ParseErrorCode code, const char *msg, size_t offset);
}

#include <common/json.h>
#include <boost/optional.hpp>

#include <rapidjson/error/en.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/prettywriter.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#include <rapidjson/schema.h>
#pragma GCC diagnostic pop
#include <rapidjson/stringbuffer.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cassert>
#include <cstring>
#include <limits>
#include <map>
#include <memory>     // std::make_unique
#include <stdexcept>  // std::runtime_error
#include <string>
#include <vector>

#include <iostream>

namespace config
{
  static constexpr const char *debug_level = "debug_level";

  static constexpr const char *shards = "shards";
  static constexpr const char *port = "port";
  static constexpr const char *net_providers = "net_providers";
  static constexpr const char *dax_config = "dax_config";
  static constexpr const char *resources = "resources";
  /* There are ado_cores in two places: under shard and under resources. Are both used? */
  static constexpr const char *ado_cores = "ado_cores";
  static constexpr const char *ado_manager_core = "ado_manager_core";
  static constexpr const char *ado_core_count = "ado_core_count";
  static constexpr const char *ado_plugins = "ado_plugins";
  static constexpr const char *ado_params = "ado_params";
  static constexpr const char *ado_path = "ado_path";
  static constexpr const char *security = "security";
  static constexpr const char *cert = "cert";
  static constexpr const char *cluster = "cluster";
  static constexpr const char *core = "core";
  static constexpr const char *group = "group";
  static constexpr const char *name = "name";
}

namespace
{
  struct ParseException
      : std::runtime_error
      , private rapidjson::ParseResult {
    ParseException(rapidjson::ParseErrorCode code, const char *msg, size_t offset)
        : std::runtime_error(msg),
          ParseResult(code, offset)
    {
    }
  };

  const char *k_typenames[] = {"Null", "False", "True", "Object", "Array", "String", "Number"};
  constexpr unsigned int DEFAULT_CLUSTER_PORT = 11800U;

  boost::optional<std::string> init_net_providers(rapidjson::Document &doc_)
  {
    auto nit = doc_.FindMember(config::net_providers);
    if (nit != doc_.MemberEnd()) {
      /* Note rapidjson.org says that a missing operator[] will assert.
       * If true, the catch may not do much good.
       */
      auto &net_provider = nit->value;

      if (!net_provider.IsString()) {
        const auto &np_type = k_typenames[net_provider.GetType()];
        PLOG("%s type:%s", config::net_providers, np_type);
        throw Config_exception("bad JSON: %s is %s but should be string", config::net_providers, np_type);
      }
      return std::string(net_provider.GetString());
    }
    else {
      return boost::optional<std::string>();
    }
  }

  rapidjson::Value init_shards(rapidjson::Document &doc_)
  {
    rapidjson::Value shards;
    if (!doc_.HasMember(config::shards)) throw Config_exception("invalid configuration (missing %s)", config::shards);
    try {
      shards = doc_[config::shards];
      return shards;
    }
    catch (...) {
      throw Config_exception("bad JSON in configuration (%s))", config::shards);
    }
  }

  rapidjson::Value init_resources(rapidjson::Document &doc_)
  {
    rapidjson::Value resources;
    if (doc_.HasMember(config::resources)) { /* optional */
      try {
        resources = std::move(doc_[config::resources]);
      }
      catch (...) {
        throw Config_exception("bad JSON in configuration file (%s))", config::resources);
      }

      if (!resources.IsObject()) {
        const auto &res_type = k_typenames[resources.GetType()];
        PLOG("resource type:%s", res_type);
        throw Config_exception("bad JSON: %s is a %s, should be {..}", config::resources, res_type);
      }
      if (resources.HasMember(config::ado_cores) && !resources[config::ado_cores].IsString())
        throw Config_exception("bad JSON: %s::%s members not string", config::resources, config::ado_cores);
      if (resources.HasMember(config::ado_manager_core) & !resources[config::ado_manager_core].IsInt())
        throw Config_exception("bad JSON: optional resources::ado manager core member not int");
    }
    return resources;
  }

  rapidjson::Value init_cluster(rapidjson::Document &doc_)
  {
    rapidjson::Value cluster;
    if (doc_.HasMember(config::cluster)) {
      try {
          cluster = doc_[config::cluster];
      }
      catch (...) {
        throw Config_exception("bad JSON in configuration file (%s))", config::cluster);
      }
      if (!cluster.IsObject()) {
        const auto &res_type = k_typenames[cluster.GetType()];
        PLOG("resource type:%s", res_type);
        throw Config_exception("bad JSON: %s should be {..}", config::cluster);
      }
    }
    return cluster;
  }

  rapidjson::Value init_security(rapidjson::Document &doc_)
  {
    rapidjson::Value security;
    if (doc_.HasMember(config::security)) {
      try {
        security = doc_[config::security];
      }
      catch (...) {
        throw Config_exception("bad JSON in configuration file (security))");
      }
      if (!security.IsObject()) {
        const auto &res_type = k_typenames[security.GetType()];
        PLOG("resource type:%s", res_type);
        throw Config_exception("bad JSON: security should be {..}");
      }
      if (!security.HasMember(config::cert) || !security[config::cert].IsString())
        /* And odd exception: An option element is missing! Is it really optional? */
        throw Config_exception("bad JSON: optional %s::%s missing or wrong type", config::security, config::cert);
    }
    return security;
  }

  boost::optional<std::string> init_ado_path(rapidjson::Document &doc_)
  {
    auto ado_path = doc_.FindMember(config::ado_path);
    if (ado_path != doc_.MemberEnd()) {
      rapidjson::Value &ap = ado_path->value;
      if (!ap.IsString()) {
        throw Config_exception("bad JSON: ado_path should be string");
      }
      return std::string(ap.GetString());
    }

    return boost::optional<std::string>();
  }

  std::string error_report(const std::string &prefix, const std::string &text, const rapidjson::Document &doc)
  {
    return prefix + " '" + text + "': " + rapidjson::GetParseError_En(doc.GetParseError()) + " at " + std::to_string(doc.GetErrorOffset());
  }

  using PrettyWriter = rapidjson::PrettyWriter<rapidjson::StringBuffer>;

  auto make_schema_cores_list(const std::string &desc_)
  {
    namespace c_json = common::json;
    namespace schema = c_json::schema;
    using json = c_json::serializer<PrettyWriter>;
    return
      json::object
      ( json::member(schema::description, desc_ + " A comma-separated list of CPU core numbers or ranges, or both")
      , json::member(schema::examples, json::array("0:3,5,7-10", "0-2,5,7:4") )
      , json::member
        ( schema::type
        , schema::string
        )
      , json::member
        ( schema::pattern
        , "([0-9]+([-:][0-9]+)?)(,[0-9]+([-:][0-9]+)?)*"
        )
      );
  }

  auto make_schema_port(const std::string &desc_, std::uint16_t dflt_)
  {
    namespace c_json = common::json;
    namespace schema = c_json::schema;
    using json = c_json::serializer<PrettyWriter>;
    return
      json::object
      ( json::member(schema::description, desc_ + " An IP port number")
      , json::member(schema::examples, json::array(11911, 19000))
      , json::member
        (  schema::type
        , schema::integer
        )
      , json::member
        ( schema::minimum
        , json::number(0)
        )
      , json::member
        ( schema::maximum
        , std::numeric_limits<std::uint16_t>::max()
        )
      , json::member
        ( schema::k_default /* informational only */
        , json::number(dflt_)
        )
      );
  }

  /* The schema for a single shard */
  auto make_schema_shard()
  {
    namespace c_json = common::json;
    namespace schema = c_json::schema;
    using json = c_json::serializer<PrettyWriter>;
    return
      json::object
      ( json::member(schema::type, schema::object)
      , json::member(schema::additionalProperties, json::boolean(false))
      , json::member
        ( schema::properties
        , json::object
          ( json::member( config::port, make_schema_port("When the value of netproviders is 'verbs', the default is 11911. When 'sockets', 11921", 11911) )
          , json::member
            ( config::core
            , json::object
              ( json::member(schema::description, "CPU core to which the shard thread should be assigned")
              , json::member(schema::examples, json::array(json::number(0), json::number(5)))
              , json::member
                ( schema::type
                , schema::integer
                )
              , json::member
                ( schema::minimum
                , json::number(0)
                )
              )
            )
          , json::member
            ( config::index
              , json::object
                ( json::member(schema::description, "Unused.")
                , json::member(schema::type, schema::string)
                )
            )
          , json::member
            ( config::net
              , json::object
                ( json::member(schema::description, "Device on which to listen for clients (alternative to addr).")
                , json::member(schema::examples, json::array("mlx5_0", "mlx4_1"))
                , json::member(schema::type, schema::string)
                )
            )
          , json::member
            ( config::addr
              , json::object
                ( json::member(schema::description, "IPv4 address on which to listen for clients (alternative to net).")
                , json::member(schema::examples, json::array("10.0.0.1", "14.4.0.25"))
                , json::member(schema::type, schema::string)
                )
            )
          , json::member
            ( config::default_backend
              , json::object
                ( json::member(schema::description, "Key/value store implementation to use.")
                , json::member(schema::examples, json::array("hstore", "hstore-cc", "mapstore"))
                , json::member(schema::type, schema::string)
                , json::member(schema::type, schema::string)
                )
            )
          , json::member
            ( config::dax_config
            , json::object
              ( json::member(schema::description, "An array. The schema for items is up to dax_map. See the DAX Configuration Schema.")
              , json::member(schema::type, schema::array)
              )
            )
          , json::member
            ( config::ado_plugins
            , json::object
              ( json::member(schema::description, "An array of ADO shared libraries to load.")
              , json::member(schema::examples, json::array( json::array("libcomponent-adoplugin-testing.so")))
              , json::member(schema::type, schema::array)
              , json::member
                ( schema::items
                , schema::string
                )
              )
            )
          , json::member
            ( config::ado_params
            , json::object
              ( json::member(schema::description, "Key/value pairs passed to ADO. The values must be strings")
              , json::member(schema::examples, json::array(json::array("Country", "Italy", "City", "Turin")))
              , json::member(schema::type, schema::object)
              , json::member
                ( schema::additionalProperties
                , json::object(json::member(schema::type, schema::string))
                )
              , json::member
                ( schema::description
                , "Key/value pairs. The values must be strings"
                )
              )
            )
          , json::member
            ( config::ado_cores
            , make_schema_cores_list("Cores to use for ADO processes.")
            )
          , json::member
            ( config::ado_core_count
            , json::object
              ( json::member(schema::description, "A scheduling parameter for ADO. Perhaps it indicates the expected CPU load, relative to other AOO prcoesses.")
              , json::member(schema::examples, json::array(json::number("0", "2"), json::number("1", "4")))
              , json::member(schema::type, schema::number)
              )
            )
          )
        )
      , json::member
        ( schema::required
          , json::array(config::core)
        )
      );
  }

  auto make_schema_shard_array()
  {
    namespace c_json = common::json;
    namespace schema = c_json::schema;
    using json = c_json::serializer<PrettyWriter>;
    return
      json::object
      ( json::member(schema::description, "Zero or more 'shards', addressable by client through their IP/port addresses")
      , json::member(schema::type, schema::array)
      , json::member
        ( schema::items
        , make_schema_shard()
        )
      );
  }

  std::string make_schema_string()
  {
    namespace c_json = common::json;
    namespace schema = c_json::schema;
    using json = c_json::serializer<PrettyWriter>;

    auto schema_object =
      json::object
      ( json::member(schema::type, schema::object)
      , json::member(schema::additionalProperties, json::boolean(false))
      , json::member
        ( schema::properties
        , json::object
          ( json::member
            ( config::debug_level
              , json::object
                ( json::member(schema::description, "Amount of debugging information to be emitted. Higher numbers may produce more debug messages")
                , json::member(schema::examples, json::array(json::number(0),json::number(1)))
                , json::member(schema::type, schema::integer)
                )
            )
          , json::member
            ( config::shards
            , make_schema_shard_array()
            )
          , json::member
            ( config::ado_path
            , json::object
              ( json::member(schema::description, "Full patch to the server ADO plugin manager executable.")
              , json::member(schema::examples, json::array("/opt/mcas/bin/ado"))
              , json::member(schema::type, schema::string))
            )
          , json::member
            ( config::net_providers
            , json::object
              ( json::member(schema::description, "ilibfabric net provider ('verbs' or 'sockets')")
              , json::member(schema::examples, json::array("verbs", "sockets"))
              , json::member(schema::type, schema::string))
            )
          , json::member
            ( config::resources
            , json::object
              ( json::member(schema::description, "ADO(?) resources")
              , json::member(schema::type, schema::object)
              , json::member(schema::additionalProperties, json::boolean(false))
              , json::member
                ( schema::properties
                , json::object
                  ( json::member
                    ( config::ado_cores
                    , make_schema_cores_list("Cores which may be used for non shard-specific ADO threads?")
                    )
                  , json::member
                    ( config::ado_manager_core
                    , json::object
                      ( json::member(schema::description, "ADO manager core")
                      , json::member(schema::examples, json::array(json::number(0),json::number(3)))
                      , json::member(schema::type, schema::integer)
                      , json::member(schema::minimum, json::number(0))
                      )
                    )
                  )
                )
              )
            )
          , json::member
            ( config::security
            , json::object
              ( json::member(schema::description, "Security parameters")
              , json::member(schema::type, schema::object)
              , json::member
                ( schema::properties
                , json::object
                  ( json::member
                    ( config::cert
                    , json::object
                      ( json::member(schema::description, "Certificate file path")
                      , json::member(schema::examples, json::array("/opt/mcas/certs/mcas-cert.pem"))
                      , json::member(schema::type, schema::string)
                      )
                    )
                  )
                )
              , json::member
                ( schema::required
                , json::array(config::cert)
                )
              , json::member(schema::additionalProperties, json::boolean(false))
              )
            )
          , json::member
            ( config::cluster
            , json::object
              ( json::member(schema::description, "Some sort of cluster identificaton")
              , json::member(schema::type, schema::object)
              , json::member
                ( schema::properties
                , json::object
                  ( json::member
                    ( config::group
                      , json::object
                        ( json::member(schema::description, "Zyre cluster group to which the server will belong")
                        , json::member(schema::type, schema::string)
                        )
                    )
                  , json::member
                    ( config::name
                    , json::object
                      ( json::member(schema::description, "local node name in Zyre cluster")
                      , json::member(schema::type, schema::string)
                      )
                    )
                  , json::member
                    ( config::addr
                    , json::object
                      ( json::member(schema::description, "local IPv4 address for Zyre cluster communication")
                      , json::member(schema::type, schema::string)
                      )
                    )
                  , json::member
                    ( config::port
                    , make_schema_port("local port for Zyre cluster communication", DEFAULT_CLUSTER_PORT)
                    )
                  )
                )
              , json::member(schema::required, json::array(config::group, config::addr))
              , json::member(schema::additionalProperties, json::boolean(false))
              )
            )
          )
        )

      , json::member
        ( schema::required
        , json::array(config::shards)
        )
      );

    rapidjson::StringBuffer buffer;
    PrettyWriter writer(buffer);
    schema_object.serialize(writer);
    return buffer.GetString();
  }

  rapidjson::SchemaDocument make_schema_doc(unsigned debug_level_)
  {
    auto config_schema = make_schema_string();

    rapidjson::Document doc;
	if ( 4 < debug_level_ )
	{
		std::cerr << config_schema << "\n";
	}
    doc.Parse(config_schema.c_str());
    if ( doc.HasParseError() )
    {
      throw std::logic_error(error_report("Bad JSONi config schema", config_schema.c_str(), doc));
    }
    return rapidjson::SchemaDocument(doc);
  }

  rapidjson::Document validate(unsigned debug_level_, rapidjson::Document &&doc)
  {
    auto schema_doc(make_schema_doc(debug_level_));
    rapidjson::SchemaValidator validator(schema_doc);

    if ( ! doc.Accept(validator) )
    {
      std::string why;
      {
        rapidjson::StringBuffer sb;
        validator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
        why += std::string("Invalid schema: ") + sb.GetString() + "\n";
        why += std::string("Invalid keyword: ") + validator.GetInvalidSchemaKeyword() + "\n";
      }

      {
        rapidjson::StringBuffer sb;
        validator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
        why += std::string("Invalid document: ") + sb.GetString() + "\n";
      }
      throw std::domain_error(error_report("JSON config failed validation", why, doc));
    }
    return std::move(doc);
  }

  rapidjson::Document string_to_doc(unsigned debug_level_, const std::string &config)
  {
    if (0 < debug_level_) PLOG("config: (%s)", config.c_str());

    try {
      rapidjson::Document doc;
      doc.Parse(config.c_str());
      return validate(debug_level_, std::move(doc));
    }
    catch (...) {
      throw Config_exception("configuration string parse failed");
    }
  }

  rapidjson::Document filename_to_doc(unsigned debug_level_, const std::string &filename)
  {
    if (0 < debug_level_) PLOG("config_file: (%s)", filename.c_str());

    using namespace rapidjson;

    /* use file stream instead of istreamwrapper because of older Ubuntu 16.04
     */
    FILE *fp = ::fopen(filename.c_str(), "rb");
    if (fp == nullptr) throw Config_exception("configuration file open/parse failed");

    struct stat st;
    stat(filename.c_str(), &st);
    auto file_size = std::size_t(st.st_size);
    auto buffer    = std::make_unique<char[]>(file_size);

    try {
      FileReadStream      is(fp, buffer.get(), file_size);
      rapidjson::Document doc;
      doc.ParseStream(is);
      ::fclose(fp);

      return validate(debug_level_, std::move(doc));
    }
    catch (...) {
      throw Config_exception("configuration file open/parse failed");
    }
  }
}

void mcas::throw_parse_exception(rapidjson::ParseErrorCode code, const char *msg, size_t offset)
{
  throw ParseException(code, msg, offset);
}

mcas::Config_file::Config_file(unsigned debug_level_, const std::string &config_spec)
    : Config_file(debug_level_, (config_spec[0] == '{' ? string_to_doc : filename_to_doc)(debug_level_, config_spec))
{
}

mcas::Config_file::Config_file(unsigned debug_level_, rapidjson::Document &&doc)
    : common::log_source(debug_level_),
      _doc(std::move(doc)),
      _shards(init_shards(_doc)),
      _net_providers(init_net_providers(_doc)),
      _ado_path(init_ado_path(_doc)),
      _resources(init_resources(_doc)),
      _security(init_security(_doc)),
      _cluster(init_cluster(_doc))
{
  if (!_shards.IsArray()) {
    const auto &shard_type = k_typenames[_shards.GetType()];
    PLOG("shards type:%s", shard_type);
    throw Config_exception("bad JSON: shards is %s, should be array", shard_type);
  }

  for (auto &m : _shards.GetArray()) {
    {
      auto it = m.FindMember(config::core);
      if (it == m.MemberEnd()) throw Config_exception("bad JSON: %s::%s member missing", config::shards, config::core);
      if (!it->value.IsInt()) throw Config_exception("bad JSON: %s::%s member not an integer", config::shards, config::core);
    }

    {
      auto it = m.FindMember(config::port);
      if (it != m.MemberEnd() && !it->value.IsInt())
        throw Config_exception("bad JSON: %s::%s member present but not an integer", config::shards, config::port);
    }

    if (m.HasMember(config::net) && !m[config::net].IsString())
      throw Config_exception("bad JSON:i %s::%s member present but not string", config::shards, config::net);

    if (m.HasMember(config::addr) && !m[config::addr].IsString())
      throw Config_exception("bad JSON: %s::%s member present but not string", config::shards, config::addr);
  }
  if (0 < debug_level()) {
    for (unsigned i = 0; i < shard_count(); i++) {
      auto net  = get_shard_optional(config::net, i);
      auto addr = get_shard_optional(config::addr, i);
      PLOG("shard: %s(%d) %s(%d) %s(%s) %s(%s)", config::core, get_shard_core(i), config::core, get_shard_port(i),
           config::addr, addr ? addr->c_str() : "<none>", config::net, net ? net->c_str() : "<none>");
    }

    PLOG("%s: %s", config::net_providers, _net_providers ? _net_providers->c_str() : "<none>");
  }
}

std::string mcas::Config_file::get_ado_cores() const
{
  if (_resources.IsNull() || !_resources.HasMember(config::ado_cores)) return "";
  return std::string(_resources[config::ado_cores].GetString());
}

int mcas::Config_file::get_ado_manager_core() const
{
  if (_resources.IsNull()) return -1;
  if (!_resources.HasMember(config::ado_manager_core)) return -1;
  return _resources[config::ado_manager_core].GetInt();
}

auto mcas::Config_file::get_shard(rapidjson::SizeType i) const
{
  if (i > shard_count()) throw Config_exception("%s out of bounds", __func__);
  assert(_shards[i].IsObject());
  return _shards[i].GetObject();
}

std::string mcas::Config_file::get_shard_ado_cores(rapidjson::SizeType i) const
{
  if (i > shard_count()) throw Config_exception("%s: out of bounds", __func__);
  assert(_shards[i].IsObject());
  auto shard = _shards[i].GetObject();
  if (!shard.HasMember(config::ado_cores)) return "";
  return std::string(shard[config::ado_cores].GetString());
}

float mcas::Config_file::get_shard_ado_core_number(rapidjson::SizeType i) const
{
  if (i > shard_count()) throw Config_exception("%s: out of bounds", __func__);
  assert(_shards[i].IsObject());
  auto shard = _shards[i].GetObject();
  if (shard.HasMember(config::ado_cores)) return -1;
  if (!shard.HasMember(config::ado_core_count)) return 1;
  return shard[config::ado_core_count].GetFloat();
}

unsigned int mcas::Config_file::get_shard_core(rapidjson::SizeType i) const
{
  if (i > shard_count()) throw Config_exception("%s out of bounds", __func__);
  assert(_shards[i].IsObject());
  auto shard = _shards[i].GetObject();
  return shard[config::core].GetUint();
}

unsigned int mcas::Config_file::get_shard_port(rapidjson::SizeType i) const
{
  if (i > shard_count()) throw Config_exception("%s out of bounds", __func__);
  assert(_shards[i].IsObject());
  auto shard = _shards[i].GetObject();
  auto m     = shard.FindMember(config::port);
  return m == shard.MemberEnd() ? 0 : m->value.GetUint();
}

boost::optional<std::string> mcas::Config_file::get_shard_optional(std::string field, rapidjson::SizeType i) const
{
  if (field.empty()) throw Config_exception("%s invalid field", __func__);
  auto shard = get_shard(i);
  auto m     = shard.FindMember(field.c_str());
  if (m != shard.MemberEnd() && !m->value.IsString()) {
    throw Config_exception("%s: \"%s\" value not a string", __func__, field.c_str());
  }
  return m == shard.MemberEnd() ? boost::optional<std::string>() : std::string(m->value.GetString());
}

std::string mcas::Config_file::get_shard_required(std::string field, rapidjson::SizeType i) const
{
  if (field.empty()) throw Config_exception("%s invalid field", __func__);
  auto shard = get_shard(i);
  return shard.HasMember(field.c_str()) ? std::string(shard[field.c_str()].GetString()) : std::string();
}

std::vector<std::string> mcas::Config_file::get_shard_ado_plugins(rapidjson::SizeType i) const
{
  auto result = std::vector<std::string>();
  if (i > shard_count()) throw Config_exception("%s shard out of bounds", __func__);

  auto shard = get_shard(i);
  if (shard.HasMember(config::ado_plugins)) {
    if (!shard[config::ado_plugins].IsArray()) throw Config_exception("%s should be an array", config::ado_plugins);
    auto array = shard[config::ado_plugins].GetArray();
    for (auto itr = array.Begin(); itr != array.End(); ++itr) {
      result.push_back(itr->GetString());
    }
  }
  return result;
}

std::map<std::string, std::string> mcas::Config_file::get_shard_ado_params(rapidjson::SizeType i) const
{
  auto result = std::map<std::string, std::string>();
  if (i > shard_count()) throw Config_exception("%s shard out of bounds", __func__);

  auto shard = get_shard(i);
  if (shard.HasMember(config::ado_params)) {
    auto obj = shard[config::ado_params].GetObject();  //.GetObject();
    for (auto itr = obj.MemberBegin(); itr != obj.MemberEnd(); ++itr) {
      result[itr->name.GetString()] = itr->value.GetString();
    }
#if 0
    if (!shard[config::ado_params].IsArray()) throw Config_exception("%s should be an array", config::ado_params);
    auto array = shard[config::ado_params].GetArray();
     for (auto itr = array.Begin(); itr != array.End(); ++itr) {
      result.push_back(itr->GetString());
    }
#endif
  }
  return result;
}

auto mcas::Config_file::get_shard_object(std::string name, rapidjson::SizeType i) const
{
  if (i > shard_count()) throw Config_exception("%s out of bounds", __func__);
  if (name.empty()) throw Config_exception("%s invalid name", __func__);
  auto shard = get_shard(i);
  if (!shard.HasMember(name.c_str()))
    throw Config_exception("%s: object (%s) does not exist", __func__, name.c_str());
  return shard[name.c_str()].GetObject();
}

boost::optional<rapidjson::Document> mcas::Config_file::get_shard_dax_config_raw(rapidjson::SizeType i)
{
  if (i > shard_count()) throw Config_exception("%s out of bounds", __func__);

  auto shard = get_shard(i);

  if (!shard.HasMember("dax_config")) return boost::optional<rapidjson::Document>();

  if (0 != ::strcmp(k_typenames[shard["dax_config"].GetType()], "Array"))
    throw Config_exception("dax_config attribute should be an array");

  rapidjson::Document d(rapidjson::kArrayType);
  d.CopyFrom(shard["dax_config"], _doc.GetAllocator());
  return d;
}

std::string mcas::Config_file::get_cert_path() const
{
  return (!_doc.HasMember(config::security)) ? std::string() : std::string(_security[config::cert].GetString());
}

std::string mcas::Config_file::cluster_group() const
{
  if (!_cluster.IsObject()) return std::string();
  return (_cluster.HasMember(config::group)) ? std::string(_cluster[config::group].GetString()) : std::string();
}

std::string mcas::Config_file::cluster_local_name() const
{
  if (!_cluster.IsObject()) return std::string();
  return (_cluster.HasMember(config::name)) ? std::string(_cluster[config::name].GetString()) : std::string();
}

std::string mcas::Config_file::cluster_ip_addr() const
{
  if (!_cluster.IsObject()) return std::string();
  return (_cluster.HasMember(config::addr)) ? std::string(_cluster[config::addr].GetString()) : std::string();
}

unsigned int mcas::Config_file::cluster_net_port() const
{
  if (!_cluster.IsObject()) return -1U;
  return (_cluster.HasMember(config::port)) ? _cluster[config::port].GetUint() : DEFAULT_CLUSTER_PORT;
}

unsigned int mcas::Config_file::debug_level() const
{
  return std::max(
    common::log_source::debug_level()
    , _doc.HasMember(config::debug_level) ? _doc[config::debug_level].GetUint() : 0U
  );
}
