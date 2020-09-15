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
#ifndef __SHARD_CONNECTION_H__
#define __SHARD_CONNECTION_H__

#ifdef __cplusplus

#include "buffer_manager.h"
#include "fabric_connection_base.h"  // default to fabric transport
#include "mcas_config.h"
#include "pool_manager.h"
#include "protocol.h"
#include "protocol_ostream.h"
#include "region_manager.h"

#include <api/components.h>
#include <api/fabric_itf.h>
#include <api/kvstore_itf.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <gsl/pointers>

#include <cassert>
#include <map>
#include <queue>
#include <utility> /* swap */
#include <vector>

namespace mcas
{
using Connection_base = Fabric_connection_base;

/**
 * Connection handler is instantiated for each "connected" client
 */
class Connection_handler
    : public Connection_base
    , public Region_manager {

 protected:
  enum State {
    INITIAL,
    WAIT_HANDSHAKE,
    WAIT_NEW_MSG_RECV,
    CLIENT_DISCONNECTED,
  };

 public:
  enum {
    TICK_RESPONSE_CONTINUE        = 0,
    TICK_RESPONSE_BOOTSTRAP_SPAWN = 1,
    TICK_RESPONSE_CLOSE           = 0xFF,
  };

  enum {
    ACTION_NONE = 0,
    ACTION_RELEASE_VALUE_LOCK_EXCLUSIVE,
    ACTION_POOL_DELETE,
  };

 private:
  State    _state       = State::INITIAL;
  unsigned option_DEBUG = mcas::global::debug_level;

  /* list of pre-registered memory regions; normally one region */
  std::vector<component::IKVStore::memory_handle_t> _mr_vector;

  uint64_t               _tick_count alignas(8);
  uint64_t               _auth_id;
  std::queue<buffer_t *> _pending_msgs;
  std::queue<action_t>   _pending_actions;
#if 0
  double                 _freq_mhz;
#endif
  Pool_manager _pool_manager; /* instance shared across connections */
  /* Adaptor point for different transports */
  using Connection = component::IFabric_server;
  using Factory    = component::IFabric_server_factory;

  struct stats {
    uint64_t response_count;
    uint64_t recv_msg_count;
    uint64_t send_msg_count;
    uint64_t wait_send_value_misses;
    uint64_t wait_msg_recv_misses;
    uint64_t wait_respond_complete_misses;
    uint64_t last_count;
    uint64_t next_stamp;
    stats()
        : response_count(0),
          recv_msg_count(0),
          send_msg_count(0),
          wait_send_value_misses(0),
          wait_msg_recv_misses(0),
          wait_respond_complete_misses(0),
          last_count(0),
          next_stamp(0)
    {
    }
  } _stats alignas(8);

  void dump_stats()
  {
    PINF("-----------------------------------------");
    PINF("| Connection Handler Statistics         |");
    PINF("-----------------------------------------");
    PINF("Ticks                       : %lu", _tick_count);
    PINF("Open pools                  : %lu", _pool_manager.open_pool_count());
    PINF("NEW_MSG_RECV misses         : %lu", _stats.wait_msg_recv_misses);
    PINF("Recv message count          : %lu", _stats.recv_msg_count);
    PINF("Send message count          : %lu", _stats.send_msg_count);
    PINF("Response count              : %lu", _stats.response_count);
    PINF("WAIT_SEND_VALUE misses      : %lu", _stats.wait_send_value_misses);
    PINF("WAIT_RESPOND_COMPLETE misses: %lu", _stats.wait_respond_complete_misses);
    PINF("-----------------------------------------");
  }

  /**
   * Change state in FSM
   *
   * @param s
   */
  inline void set_state(State s)
  {
    if (2 < option_DEBUG) {
      const std::map<int, const char *> m{
          {INITIAL, "INITIAL"}, {WAIT_HANDSHAKE, "WAIT_HANDSHAKE"}, {WAIT_NEW_MSG_RECV, "WAIT_NEW_MSG_RECV"}};
      PLOG("state %s -> %s", m.find(_state)->second, m.find(s)->second);
    }
    _state = s; /* we could add transition checking later */
  }

  static void static_send_callback2(void *cnxn, buffer_t *iob) noexcept
  {
    auto base = static_cast<Fabric_connection_base *>(cnxn);
    static_cast<Connection_handler *>(base)->send_callback2(iob);
  }

  void send_callback2(buffer_t *iob) noexcept
  {
    assert(iob->value_adjunct);
    if (0 < option_DEBUG) {
      PLOG("Completed send2 (value_adjunct %p)", static_cast<const void *>(iob->value_adjunct));
    }
    _deferred_unlock.push(iob->value_adjunct);
    send_callback(iob);
  }

 public:
  explicit Connection_handler(unsigned debug_level_,
    gsl::not_null<Factory *> factory,
    gsl::not_null<Connection *> connection);

  ~Connection_handler();

  inline bool client_connected() { return _state != State::CLIENT_DISCONNECTED; }

  auto allocate_send() { return allocate(static_send_callback); }
  auto allocate_recv() { return allocate(static_recv_callback); }

  /**
   * State machine transition tick.  It is really important that this tick
   * execution duration is small, so that other connections are not impacted by
   * the thread not returning to them.
   *
   */
  int tick();
  /**
   * Check for network completions
   *
   */
  Fabric_connection_base::Completion_state check_network_completions()
  {
    auto state = poll_completions();

    while (!_deferred_unlock.empty()) {
      //      void *deferred_unlock = nullptr;
      //???     std::swap(deferred_unlock, _deferred_unlock.front());
      void *deferred_unlock = _deferred_unlock.front();
      if (option_DEBUG > 2)
        PLOG("adding action for deferred unlocking value @ %p", deferred_unlock);
      add_pending_action(action_t{ACTION_RELEASE_VALUE_LOCK_EXCLUSIVE, deferred_unlock});
      _deferred_unlock.pop();
    }

    return state;
  }

  /**
   * Peek at pending message from the connection. Used when the resources
   * required to process a pending nessage may depend on the content of the
   * message.
   *
   * @param msg [out] Pointer to base protocol message
   *
   * @return Pointer to buffer holding the message or null if there are none
   */
  inline mcas::protocol::Message *peek_pending_msg() const
  {
    return _pending_msgs.empty() ? nullptr
                                 : static_cast<mcas::protocol::Message *>(_pending_msgs.front()->base().get());
  }

  /**
   * Discard a pending message from the connection. Used along with tick
   * (which adds pending messages) and peek (which peeks at them).
   *
   * @param msg [out] Pointer to base protocol message
   *
   * @return Pointer to buffer holding the message or null if there are none
   */
  inline buffer_t *pop_pending_msg()
  {
    assert(!_pending_msgs.empty());
    auto iob = _pending_msgs.front();
    _pending_msgs.pop();
    return iob;
  }

  /**
   * Get deferred actions
   *
   * @param action [out] Action
   *
   * @return True if action
   */
  inline bool get_pending_action(action_t &action)
  {
    if (_pending_actions.empty()) return false;

    action = _pending_actions.front();

    if (option_DEBUG > 2) PLOG("Connection_handler: popped pending action (%u, %p)", action.op, action.parm);

    _pending_actions.pop();
    return true;
  }

  template <typename MT>
  void msg_log(const unsigned level_, const MT *msg_, const char *desc_, const char *direction_)
  {
    if (level_ < Connection_base::debug_level()) {
      std::ostringstream m;
      m << *msg_;
      PLOG("%s (%s) %s", direction_, desc_, m.str().c_str());
    }
  }

  template <typename MT>
  void msg_recv_log(const MT *m, const char *desc)
  {
    msg_recv_log(1, m, desc);
  }

  template <typename MT>
  void msg_recv_log(const unsigned level, const MT *msg, const char *desc)
  {
    msg_log(level, msg, desc, "RECV");
  }

  template <typename MT>
  void msg_send_log(const MT *m, const char *desc)
  {
    msg_send_log(1, m, desc);
  }

  template <typename MT>
  void msg_send_log(const unsigned level, const MT *msg, const char *desc)
  {
    msg_log(level, msg, desc, "SEND");
  }

  /**
   * Add an action to the pending queue
   *
   * @param action Action to add
   */
  inline void add_pending_action(const action_t& action)
  {
    _pending_actions.push(action);
  }

  template <typename Msg>
  void post_send_buffer(gsl::not_null<buffer_t *> buffer, Msg *msg, const char *desc)
  {
    msg_send_log(msg, desc);
    Connection_base::post_send_buffer(buffer);
  }

  template <typename Msg>
  void post_send_buffer2(gsl::not_null<buffer_t *> buffer,
                         const ::iovec &           val_iov,
                         void *                    val_desc,
                         Msg *                     msg,
                         const char *              func_name)
  {
    msg_send_log(msg, func_name);
    Connection_base::post_send_buffer2(buffer, val_iov, val_desc);
  }

  /**
   * Post a response
   *
   * @param iob IO buffer to post
   */
  template <typename Msg>
  inline void post_response2(gsl::not_null<buffer_t *> iob,
                             const ::iovec &           val_iov,
                             void *                    val_desc,
                             Msg *                     msg,
                             const char *              func_name)
  {
    iob->set_completion(static_send_callback2, val_iov.iov_base);
    post_send_buffer2(iob, val_iov, val_desc, msg, func_name);
    /* don't wait for this, let it be picked up in the check_completions cycle
     */

    _stats.response_count++;
  }

  /**
   * Post a response
   *
   * @param iob IO buffer to post
   */
  template <typename Msg>
  inline void post_response(gsl::not_null<buffer_t *> iob, Msg *msg, const char *desc)
  {
    post_send_buffer(iob, msg, desc);
    /* don't wait for this, let it be picked up in the check_completions cycle
     */

    _stats.response_count++;
  }

  /**
   * Set up for pending value send/recv
   *
   * @param target
   * @param target_len
   * @param region
   */
  void add_memory_handle(component::IKVStore::memory_handle_t handle) { _mr_vector.push_back(handle); }

  component::IKVStore::memory_handle_t pop_memory_handle()
  {
    if (_mr_vector.empty()) return nullptr;
    auto mr = _mr_vector.back();
    _mr_vector.pop_back();
    return mr;
  }

  inline uint64_t      auth_id() const { return _auth_id; }
  inline void          set_auth_id(uint64_t id) { _auth_id = id; }
  inline size_t        max_message_size() const { return _max_message_size; }
  inline Pool_manager &pool_manager() { return _pool_manager; }
};

}  // namespace mcas
#endif

#endif  // __CONNECTION_HANDLER_H__
