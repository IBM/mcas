#include "zyre_component.h"
#include <iostream>
#include <mutex>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wunused-value"

Zyre_component::Zyre_component(const unsigned debug_level_,
                               const std::string& node_name,
                               const std::string& nic,
                               const unsigned int port) : common::log_source(debug_level_)
{
  auto version = zyre_version();
  assert((version / 10000) % 100 == ZYRE_VERSION_MAJOR);
  assert((version / 100) % 100 == ZYRE_VERSION_MINOR);
  assert(version % 100 == ZYRE_VERSION_PATCH);
  assert(!nic.empty());

  _node = zyre_new(node_name.c_str());
  if(_node == nullptr)
    throw General_exception("zyre_new failed unexpectedly.");

  /* configure endpoint - this is using UDP beaconing, but
   * it can be configured to gossip protocol
   */
  zyre_set_interface(_node, nic.c_str());
  zyre_set_port(_node, port & INT_MAX);

  /* configure protocol */
  zyre_set_interval(_node, HEARTBEAT_INTERVAL_MS);
  zyre_set_header(_node, "X-ZYRE-MCAS", "1");

  if(debug_level() > 2) {
    PLOG("Zyre: version %f (name=%s)", static_cast<float>(version)/10000.0f, zyre_name(_node));
    zyre_set_verbose(_node);
  }

}

Zyre_component::~Zyre_component()
{
  zyre_destroy(&_node);
}


void Zyre_component::start_node()
{
  zyre_start(_node);
}

void Zyre_component::stop_node()
{
  zyre_stop(_node);
}

void Zyre_component::destroy_node()
{
  zyre_destroy(&_node);
}

void Zyre_component::group_join(const std::string& group)
{
  if(zyre_join(_node, group.c_str()))
    throw General_exception("zyre_join failed");
}

void Zyre_component::group_leave(const std::string& group)
{
  if(zyre_leave(_node, group.c_str()))
    throw General_exception("zyre_join failed");
}

void Zyre_component::shout(const std::string& group,
                           const std::string& type,
                           const std::string& message)
{
  zmsg_t * msg = zmsg_new();
  type;
  zmsg_addstr(msg, message.c_str());
  zmsg_addstr(msg, type.c_str());

  if(zyre_shout(_node, group.c_str(), &msg))
    throw General_exception("zyre_shout failed");
}

void Zyre_component::whisper(const std::string& peer_uuid,
                             const std::string& type,
                             const std::string& message)
{
  zmsg_t * msg = zmsg_new();

  zmsg_addstr(msg, message.c_str());
  zmsg_addstr(msg, type.c_str());

  if(zyre_whisper(_node, peer_uuid.c_str(), &msg)) /* destroy message after sending */
    throw General_exception("zyre_whisper failed");
}

bool Zyre_component::poll_recv(std::string& sender_uuid,
			       std::string& type,
			       std::string& message,
			       std::vector<std::string>& values)
{
  zmsg_t * msg = nullptr;
  auto socket = zyre_socket(_node);

  values.clear();

  if((msg = zmsg_recv_nowait(socket))) {
    assert(zmsg_is(msg));

    char * msgtype = zmsg_popstr(msg);
    assert(msgtype);
    type = msgtype;
    zstr_free(&msgtype);

    char * uuid = zmsg_popstr(msg);
    assert(uuid);
    sender_uuid = uuid;
    zstr_free(&uuid);

    char * body = zmsg_popstr(msg);
    assert(body);
    message = body;
    zstr_free(&body);

    char * other = zmsg_popstr(msg);
    while(other) {
      values.push_back(std::string(other));
      zstr_free(&other);
      other = zmsg_popstr(msg);
    }

    zmsg_destroy(&msg);

    return true;
  }

  return false;
}

// void Zyre_component::poll_recv(std::function<void(const std::string& sender_uuid,
//                                                   const std::string& type,
//                                                   const std::string& message)> callback)
// {
//   zmsg_t * msg = nullptr;
//   auto socket = zyre_socket(_node);
//   while((msg = zmsg_recv_nowait(socket))) {
//     assert(zmsg_is(msg));

//     char * type = zmsg_popstr(msg);
//     assert(type);
//     std::string msgtype = type;
//     zstr_free(&type);

//     char * uuid = zmsg_popstr(msg);
//     assert(uuid);
//     std::string sender = uuid;
//     zstr_free(&uuid);

//     char * body = zmsg_popstr(msg);
//     assert(body);
//     std::string msgstr = body;
//     zstr_free(&body);

//     callback(sender, msgtype, msgstr); /* invoke callback */

//     zmsg_destroy(&msg);
//   }
// }

status_t Zyre_component::set_timeout(const Timeout_type type, const int interval_ms)
{
  switch(type) {
  case Timeout_type::EVASIVE:
    zyre_set_evasive_timeout (_node, interval_ms);
    return S_OK;
  case Timeout_type::EXPIRED:
    zyre_set_expired_timeout (_node, interval_ms);
    return S_OK;
  default:
    return E_NOT_SUPPORTED;
  }
}

std::string Zyre_component::uuid() const
{
  return std::string(zyre_uuid(_node));
}

std::string Zyre_component::node_name() const
{
  return std::string(zyre_name(_node));
}

std::string Zyre_component::peer_address(const std::string& uuid)
{
  char * paddr = zyre_peer_address(_node, uuid.c_str());

  std::string result;
  if(paddr) {
    result = paddr;
    zstr_free(&paddr);
  }
  return result;
}

void Zyre_component::dump_info() const
{
  zyre_dump(_node);
}

/**
 * Factory entry point
 *
 */
extern "C" void * factory_createInstance(component::uuid_t component_id)
{
  if(component_id == Zyre_component_factory::component_id()) {
    return static_cast<void*>(new Zyre_component_factory());
  }
  else {
    PWRN("component id requested from zyre component factory does not match");
    return NULL;
  }
}


#pragma GCC diagnostic pop
