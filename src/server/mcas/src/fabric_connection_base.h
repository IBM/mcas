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
#ifndef __FABRIC_CONNECTION_BASE_H__
#define __FABRIC_CONNECTION_BASE_H__

#include "mcas_config.h"

#include "buffer_manager.h" /* Buffer_manager */

#include <api/fabric_itf.h> /* IFabric_memory_region, IFabric_server */
#include <gsl/pointers>
#include <algorithm>
#include <list>
#include <queue>

namespace mcas
{

struct open_connection_construct_key;

struct open_connection
{
private:
  gsl::not_null<component::IFabric_server_factory *> _factory;
  gsl::not_null<component::IFabric_server *>         _transport;
public:
  open_connection(
    gsl::not_null<component::IFabric_server_factory *> factory_
    , gsl::not_null<component::IFabric_server *> transport_
    , const open_connection_construct_key &
    )
    : _factory(factory_)
    , _transport(transport_)
  {}

  open_connection(const open_connection &) = delete;
  open_connection &operator=(const open_connection &) = delete;

  auto transport() const { return _transport; }

  ~open_connection()
  {
    /* close connection */
    _factory->close_connection(_transport);
  }
};

class Fabric_connection_base : protected common::log_source {

 public:
  friend class Connection;
  friend class Shard;
  using memory_region_t = component::IFabric_memory_region *;

 protected:
  using buffer_t = Buffer_manager<component::IFabric_server>::buffer_internal;
  using pool_t   = component::IKVStore::pool_t;

  /* deferred actions */
  struct action_t {
    int   op;
    void *parm;
  };

  enum class Completion_state {
    COMPLETIONS,
    CLIENT_DISCONNECT,
    NONE,
  };

 private:
  open_connection _oc;
  Buffer_manager<component::IFabric_server> _bm;
 protected:
  size_t                             _max_message_size;

  /* Filled by base class     check_for_posted_value_complete
   * Drained by derived class check_network_completions
   */
  std::queue<void *> _deferred_unlock;

  /* xx_buffer_outstanding is the signal for completion,
     xx_buffer is the buffer pointer that needs to be freed (and set to null)
  */
  unsigned              _recv_buffer_posted_count;
  std::list<buffer_t *> _completed_recv_buffers;
  unsigned              _send_buffer_posted_count;

  /* values for two-phase get
   */
  unsigned _send_value_posted_count;

  /**
   * Ctor
   *
   * @param factory
   * @param fabric_connection
   */
  explicit Fabric_connection_base( //
    unsigned                           debug_level_,
    gsl::not_null<component::IFabric_server_factory *>factory,
    gsl::not_null<component::IFabric_server *>        fabric_connection);

  Fabric_connection_base(const Fabric_connection_base &) = delete;
  Fabric_connection_base &operator=(const Fabric_connection_base &) = delete;

  ~Fabric_connection_base();

  static void static_send_callback(void *cnxn, buffer_t *iob) noexcept
  {
    static_cast<Fabric_connection_base *>(cnxn)->send_callback(iob);
  }

  void send_callback(buffer_t *iob) noexcept
  {
    --_send_buffer_posted_count;
    posted_count_log();
    CPLOG(2, "Completed send (%p)", static_cast<const void *>(iob));
    free_buffer(iob);
  }

  static void static_recv_callback(void *cnxn, buffer_t *iob) noexcept
  {
    static_cast<Fabric_connection_base *>(cnxn)->recv_callback(iob);
  }

  void recv_callback(buffer_t *iob) noexcept
  {
    --_recv_buffer_posted_count;
    posted_count_log();
    _completed_recv_buffers.push_front(iob);
    CPLOG(2, "Completed recv (%p) (complete %zu)", static_cast<const void *>(iob), _completed_recv_buffers.size());
  }

  static void completion_callback(void *   context,
                                  status_t st,
                                  std::uint64_t,  // completion_flags,
                                  std::size_t len,
                                  void *      error_data,
                                  void *      cnxn) noexcept
  {
    if (LIKELY(st == S_OK)) {
      auto iob = static_cast<buffer_t *>(context);
      iob->completion_cb(cnxn, iob);
    }
    else {
      PERR("Fabric_connection_base: fabric operation failed st != S_OK (st=%d, "
           "context=%p, len=%lu)",
           st, context, len);
      PERR("Error: %s", static_cast<char *>(error_data));
    }
  }

  bool check_for_posted_recv_complete()
  {
    /* don't free buffer (such as above); it will be used for response */
    //    return _recv_buffer_posted_outstanding == false;
    return _completed_recv_buffers.size() > 0;
  }

  void free_recv_buffer()
  {
    PLOG("freeing recv buffers");
    for (auto b : _completed_recv_buffers) {
      free_buffer(b);
    }
  }

  void post_recv_buffer(buffer_t *buffer)
  {
    transport()->post_recv(buffer->iov, buffer->iov + 1, buffer->desc, buffer);
    ++_recv_buffer_posted_count;
    posted_count_log();
    CPLOG(2, "Posted recv (%p) (complete %zu)", static_cast<const void *>(buffer), _completed_recv_buffers.size());
  }

  void post_send_buffer(gsl::not_null<buffer_t *> buffer)
  {
    const auto iov = buffer->iov;

    /* if packet is small enough use inject */
    if (iov->iov_len <= transport()->max_inject_size()) {
      CPLOG(2, "Fabric_connection_base: posting send with inject (iob %p %p,len=%lu)",
             static_cast<const void *>(buffer), iov->iov_base, iov->iov_len);

      transport()->inject_send(iov->iov_base, iov->iov_len);
      CPLOG(2, "%s: buffer %p", __func__, static_cast<const void *>(buffer));
      free_buffer(buffer); /* buffer is immediately complete fi_inject */
    }
    else {
      CPLOG(2, "Fabric_connection_base: posting send (%p, %p)", static_cast<const void *>(buffer), iov->iov_base);

      transport()->post_send(iov, iov + 1, buffer->desc, buffer);
      ++_send_buffer_posted_count;
      posted_count_log();
      CPLOG(0, "%s buffer (%p)", __func__, static_cast<const void *>(buffer));
    }
  }

  void post_send_buffer2(gsl::not_null<buffer_t *> buffer, const ::iovec &val_iov, void *val_desc)
  {
    buffer->iov[1] = val_iov;
    buffer->desc[1] = val_desc;

    ++_send_buffer_posted_count;
    posted_count_log();
    CPLOG(0, "Posted send (%p) ... value (%.*s) (len=%lu,ptr=%p)", static_cast<const void *>(buffer),
         int(val_iov.iov_len), static_cast<char *>(val_iov.iov_base), val_iov.iov_len, val_iov.iov_base);

    transport()->post_send(buffer->iov, buffer->iov + 2, buffer->desc, buffer);
  }

  buffer_t *posted_recv()
  {
    if (_completed_recv_buffers.size() == 0) return nullptr;

    auto iob = _completed_recv_buffers.back();
    _completed_recv_buffers.pop_back();
    CPLOG(2, "Presented recv (%p) (complete %zu)", static_cast<const void *>(iob), _completed_recv_buffers.size());
    return iob;
  }

  void posted_count_log() const
  {
    CPLOG(1, "POSTs recv %u send %u value %u", _recv_buffer_posted_count, _send_buffer_posted_count,
           _send_value_posted_count);
  }

  Completion_state poll_completions()
  {
    if (_recv_buffer_posted_count != 0 || _send_buffer_posted_count != 0 || _send_value_posted_count != 0) {
#if 0
      PLOG("%s posted_recv %u posted_send %u posted_value %u", __func__, _recv_buffer_posted_count, _send_buffer_posted_count, _send_value_posted_count);
#endif
      try {
        transport()->poll_completions(&Fabric_connection_base::completion_callback, this);
        return Completion_state::COMPLETIONS;
      }
      catch (const std::logic_error &e) {
        return Completion_state::CLIENT_DISCONNECT;
      }
    }
    return Completion_state::NONE;
  }

  /**
   * Forwarders that allow us to avoid exposing transport() and _bm
   *
   */
 public:
  inline auto register_memory(const void *base, size_t len, std::uint64_t key, std::uint64_t flags)
  {
    return transport()->register_memory(base, len, key, flags); /* flags not supported for verbs */
  }

  inline void deregister_memory(memory_region_t region) { return transport()->deregister_memory(region); }

  inline void *get_memory_descriptor(memory_region_t region) { return transport()->get_memory_descriptor(region); }

  inline uint64_t get_memory_remote_key(memory_region_t region) { return transport()->get_memory_remote_key(region); }

 protected:
  inline auto allocate(buffer_t::completion_t c) { return _bm.allocate(c); }

  inline void free_buffer(buffer_t *buffer) { _bm.free(buffer); }

  inline size_t IO_buffer_size() const { return Buffer_manager<component::IFabric_server>::BUFFER_LEN; }

  gsl::not_null<component::IFabric_server *> transport() const { return _oc.transport(); }

  inline std::string get_local_addr() { return transport()->get_local_addr(); }
};

struct open_connection_construct_key
{
private:
  open_connection_construct_key() {};
public:
  friend class Fabric_connection_base;
};

}  // namespace mcas

#endif  // __FABRIC_CONNECTION_BASE_H__
