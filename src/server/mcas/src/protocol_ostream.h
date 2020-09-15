
/*
  Copyright [2020] [IBM Corporation]
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

#ifndef MCAS_PROTOCOL_OSTREAM_H
#define MCAS_PROTOCOL_OSTREAM_H

#include "protocol.h"
#include <boost/io/ios_state.hpp>
#include <iostream>
#include <map>

namespace
{
struct msg_attrs {
  const char *desc;
  enum category { req, rsp, other } c;
};
static const std::map<mcas::protocol::MSG_TYPE, msg_attrs> type_map{
    {mcas::protocol::MSG_TYPE_HANDSHAKE, {"HANDSHAKE", msg_attrs::category::other}},
    {mcas::protocol::MSG_TYPE_HANDSHAKE_REPLY, {"HANDSHAKE_REPLY", msg_attrs::category::other}},
    {mcas::protocol::MSG_TYPE_CLOSE_SESSION, {"CLOSE_SESSION", msg_attrs::category::other}},
    {mcas::protocol::MSG_TYPE_STATS, {"STATS", msg_attrs::category::other}},
    {mcas::protocol::MSG_TYPE_POOL_REQUEST, {"POOL", msg_attrs::category::req}},
    {mcas::protocol::MSG_TYPE_POOL_RESPONSE, {"POOL", msg_attrs::category::rsp}},
    {mcas::protocol::MSG_TYPE_IO_REQUEST, {"IO", msg_attrs::category::req}},
    {mcas::protocol::MSG_TYPE_IO_RESPONSE, {"IO", msg_attrs::category::rsp}},
    {mcas::protocol::MSG_TYPE_INFO_REQUEST, {"INFO", msg_attrs::category::req}},
    {mcas::protocol::MSG_TYPE_INFO_RESPONSE, {"INFO", msg_attrs::category::rsp}},
    {mcas::protocol::MSG_TYPE_ADO_REQUEST, {"ADO", msg_attrs::category::req}},
    {mcas::protocol::MSG_TYPE_ADO_RESPONSE, {"ADO", msg_attrs::category::rsp}},
    {mcas::protocol::MSG_TYPE_PUT_ADO_REQUEST, {"PUT_ADO", msg_attrs::category::req}},
};

static const std::map<mcas::protocol::OP_TYPE, const char *> op_map{
    {mcas::protocol::OP_NONE, "NONE"},
    {mcas::protocol::OP_CREATE, "CREATE"},
    {mcas::protocol::OP_OPEN, "OPEN"},
    {mcas::protocol::OP_CLOSE, "CLOSE"},
    {mcas::protocol::OP_PUT, "PUT"},
    {mcas::protocol::OP_SET, "SET"},
    {mcas::protocol::OP_GET, "GET"},
    {mcas::protocol::OP_PUT_ADVANCE, "PUT_ADVANCE"},
    {mcas::protocol::OP_PUT_SEGMENT, "PUT_SEGMENT"},
    {mcas::protocol::OP_DELETE, "DELETE"},
    {mcas::protocol::OP_ERASE, "ERASE"},
    {mcas::protocol::OP_PREPARE, "PREPARE"},
    {mcas::protocol::OP_COUNT, "COUNT"},
    {mcas::protocol::OP_CONFIGURE, "CONFIGURE"},
    {mcas::protocol::OP_STATS, "STATS"},
    {mcas::protocol::OP_SYNC, "SYNC"},
    {mcas::protocol::OP_ASYNC, "ASYNC"},
    {mcas::protocol::OP_PUT_LOCATE, "PUT_LOCATE"},
    {mcas::protocol::OP_PUT_RELEASE, "PUT_RELEASE"},
    {mcas::protocol::OP_GET_LOCATE, "GET_LOCATE"},
    {mcas::protocol::OP_GET_RELEASE, "GET_RELEASE"},
    {mcas::protocol::OP_LOCATE, "LOCATE"},
    {mcas::protocol::OP_RELEASE, "RELEASE"},
    {mcas::protocol::OP_INVALID, "N/A"},
};

std::ostream &operator<<(std::ostream &o_, const mcas::protocol::OP_TYPE op_)
{
  auto op_it = op_map.find(op_);
  o_ << (op_it == op_map.end() ? "OP??" : op_it->second);
  return o_;
}

std::ostream &operator<<(std::ostream &o_, const mcas::protocol::Message &msg)
{
  o_ << "Message ";
  auto type_it = type_map.find(msg.type_id());
  if (type_it == type_map.end()) {
    o_ << "TYPE?";
  }
  else {
    o_ << type_it->second.desc << " ";
    switch (type_it->second.c) {
      case msg_attrs::category::req:
        o_ << "Q OP " << msg.op();
        break;
      case msg_attrs::category::rsp:
        o_ << "R status " << msg.get_status();
        break;
      case msg_attrs::category::other:
        break;
      default:
        o_ << "??";
        break;
    }
  }
  return o_;
}

std::ostream &operator<<(std::ostream &o_, const mcas::protocol::Message_numbered_request &msg)
{
  return o_ << static_cast<const mcas::protocol::Message &>(msg) << " id " << msg.request_id();
}

std::ostream &operator<<(std::ostream &o_, const mcas::protocol::Message_numbered_response &msg)
{
  boost::io::ios_flags_saver s(o_);
  return o_ << static_cast<const mcas::protocol::Message &>(msg) << " id " << msg.request_id();
}

inline std::ostream &operator<<(std::ostream &o_, const mcas::protocol::Message_IO_request &msg)
{
  return o_ << static_cast<const mcas::protocol::Message_numbered_request &>(msg) << " " << msg.op();
}

inline std::ostream &operator<<(std::ostream &o_, const mcas::protocol::Message_IO_response &msg)
{
  boost::io::ios_flags_saver s(o_);
  return o_ << static_cast<const mcas::protocol::Message_numbered_response &>(msg) << " addr " << std::hex
            << std::showbase << msg.addr << " data_len " << msg.data_length();
}
}  // namespace

#endif
