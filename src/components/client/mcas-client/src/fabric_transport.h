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
#ifndef __CLIENT_FABRIC_TRANSPORT_H__
#define __CLIENT_FABRIC_TRANSPORT_H__

#include "mcas_client_config.h"

#include "buffer_manager.h" /* Buffer_manager */

#include <api/fabric_itf.h> /* IFabric_client, IFabric_memory_region, IFabric_op_completer. IKVStore */
#include <common/exceptions.h>
#include <iterator> /* begin, end */

namespace mcas
{
using memory_registered_fabric = memory_registered<component::IFabric_client>;
namespace client
{

  /* Wrapper around a type T, enable destruction when T's desstructor is protected */
  template <typename T>
    struct destructible
      : public T
    {
      template <typename ... Args>
        explicit destructible(Args&& ... args)
          : T(std::forward<Args>(args)...)
        {}
    };

static constexpr size_t NUM_BUFFERS = 64; /* defines max outstanding */

class Fabric_transport : protected common::log_source {
  friend struct mcas_client;

  inline void u_post_recv(const ::iovec *first, const ::iovec *last, void **descriptors, void *context)
  {
    _transport->post_recv(first, last, descriptors, context);
  }

 public:
  using Transport = component::IFabric_client;
  /* Buffer manager defined in server/mcas/src/ */
  using buffer_base     = Buffer_manager<Transport>::buffer_base;
  using buffer_t        = Buffer_manager<Transport>::buffer_internal;
  using buffer_external = destructible<buffer_base>;
  using memory_region_t = component::IFabric_memory_region *;

  double cycles_per_second;

  explicit Fabric_transport(unsigned debug_level_, component::IFabric_client *fabric_connection, unsigned patience_);

  Fabric_transport(const Fabric_transport &) = delete;
  Fabric_transport &operator=(const Fabric_transport &) = delete;

  ~Fabric_transport() {}

  static component::IFabric_op_completer::cb_acceptance completion_callback(void *        context,
                                                                            status_t      st,
                                                                            std::uint64_t completion_flags,
                                                                            std::size_t,  // len
                                                                            void *,       // error_data
                                                                            void *param);

  /**
   * Wait for completion of a IO buffer posting
   *
   * @param iob IO buffer to wait for completion of
   */
  void wait_for_completion(void *wr);

  /**
   * Test completion of work request
   *
   * @param Work request
   *
   * @return True iff complete
   */
  inline bool test_completion(void *wr)
  {
    _transport->poll_completions_tentative(completion_callback, &wr);
    return (wr == nullptr);
  }

  /**
   * Forwarders that allow us to avoid exposing _transport and _bm
   *
   */
  auto inline make_memory_registered(void *base, size_t len)
  {
    return memory_registered_fabric(debug_level(), _transport, base, len, 0, 0);
  }

 private:
  inline memory_region_t register_memory(void *base, size_t len)
  {
    return _transport->register_memory(base, len, 0, 0);
  }

 public:
  inline void *get_memory_descriptor(memory_region_t region) { return _transport->get_memory_descriptor(region); }

  inline void deregister_memory(memory_region_t region) { _transport->deregister_memory(region); }

  inline void post_send(const ::iovec *first, const ::iovec *last, void **descriptors, void *context)
  {
    _transport->post_send(first, last, descriptors, context);
  }

  inline void post_recv(const ::iovec *first, const ::iovec *last, void **descriptors, void *context)
  {
    CPLOG(2, "%s (%p): IOV count %zu first %p:%zx", __func__, context, std::size_t(last - first), first->iov_base, first->iov_len);
    u_post_recv(first, last, descriptors, context);
  }

  inline size_t max_message_size() const { return _transport->max_message_size(); }

  /**
   * Post send one buffer and wait for completion.
   *
   * @param iob IO buffer
   */
  void sync_send(buffer_t *iob)
  {
    post_send(iob->iov, iob->iov + 1, iob->desc, iob);
    wait_for_completion(iob);
  }

  /**
   * Post send two buffers and wait for completion.
   *
   * @param iob First IO buffer
   * @param iob_extra Second IO buffer
   */
  void sync_send(buffer_t *iob, buffer_external *iob_extra)
  {
    iob->iov[1] = ::iovec{iob_extra->base(), iob_extra->original_length()};
    iob->desc[1] = iob_extra->get_desc();
    post_send(iob->iov, iob->iov + 2, iob->desc, iob);
    wait_for_completion(iob);
  }

  /**
   * Perform inject send (fast for small packets)
   *
   * @param iob Buffer to send
   */
  void sync_inject_send(buffer_t *iob, std::size_t len)
  {
    iob->set_length(len);
    if (len <= _max_inject_size) {
      /* when this returns, iob is ready for immediate reuse */
      _transport->inject_send(iob->base(), iob->length());
    }
    else {
      /* too big for inject, do plain send */
      post_send(iob->iov, iob->iov + 1, iob->desc, iob);
      wait_for_completion(iob);
    }
  }

  /**
   * Post send one buffer
   *
   * @param iob First IO buffer
   */
  void post_send(gsl::not_null<buffer_t *> iob)
  {
    post_send(iob->iov, iob->iov + 1, iob->desc, iob);
  }

#if 0
  void post_send(buffer_t *iob, buffer_external *iob_extra)
  {
    iovec v[2]   = {iob->iov[0], iob_extra->iov[0]};
    void *desc[] = {iob->desc[0], iob_extra->desc[0]};
    post_send(&v[0], &v[2], desc, iob);
  }
#endif

  /**
   * Post receive then wait for completion before returning.
   * Use after post_send may lead to poor performance if the response
   * arrives before the receive buffer is posted.
   *
   * @param iob IO buffer
   */
  void sync_recv(buffer_t *iob)
  {
    post_recv(iob);
    wait_for_completion(iob);
  }

  void post_recv(buffer_t *iob)
  {
    post_recv(iob->iov, iob->iov + 1, iob->desc, iob);
  }

  /**
   * Post write (DMA)
   *
   * @param iob First IO buffer
   */
  void post_write(const ::iovec *first,
                  const ::iovec *last,
                  void **        desc,
                  uint64_t       remote_addr,
                  std::uint64_t  remote_key,
                  void *         context)
  {
    _transport->post_write(first, last, desc, remote_addr, remote_key, context);
  }

  /**
   * Post read (DMA)
   *
   * @param iob First IO buffer
   */
  void post_read(const ::iovec *first,
                 const ::iovec *last,
                 void **        desc,
                 uint64_t       remote_addr,
                 std::uint64_t  remote_key,
                 void *         context)
  {
    _transport->post_read(first, last, desc, remote_addr, remote_key, context);
  }

  component::IKVStore::memory_handle_t register_direct_memory(void *region, size_t region_len)
  {
    // if (!check_aligned(region, 64))
    //   throw API_exception("register_direct_memory: region should be
    //   aligned");

    auto buffer = new buffer_external(debug_level(), _transport, region, region_len);
    CPLOG(0, "register_direct_memory (%p, %lu, mr=%p)", region, region_len, static_cast<const void *>(buffer->region()));
    return buffer;
  }

  status_t unregister_direct_memory(component::IKVStore::memory_handle_t handle)
  {
    delete static_cast<buffer_external *>(handle);
    return S_OK;
  }

  inline auto allocate(buffer_t::completion_t c) { return _bm.allocate(c); }
  inline void free_buffer(buffer_t *buffer) { _bm.free(buffer); }

 protected:
  Transport *               _transport;
  size_t                    _max_inject_size;
  Buffer_manager<Transport> _bm;          /*< IO buffer manager */
  unsigned                  _patience;  // in seconds
};

}  // namespace client
}  // namespace mcas

#endif  //__CLIENT_FABRIC_TRANSPORT_H__
