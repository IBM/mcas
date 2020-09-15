/*
  Copyright [2017-2019] [IBM Corporation]
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

#ifndef __ADOPROTO_H__
#define __ADOPROTO_H__

#include "ado_proto_buffer.h"
#include "channel_wrap.h"
#include "uipc.h"
#include <common/errors.h>
#include <common/exceptions.h>
#include <common/types.h>
#include <common/utils.h>
#include <api/kvindex_itf.h>
#include <api/ado_itf.h>
#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <string>

#define DEBUG

class Buffer_header;

using buffer_space_shared_ptr_t = ado_protocol_buffer::space_shared_ptr_t;
using buffer_space_dedicated_ptr_t = ado_protocol_buffer::space_dedicated_ptr_t;

/**
 * Class to help UIPC message construction (both sides)
 *
 */

class ADO_protocol_builder
{
public:
  static constexpr size_t MAX_MESSAGE_SIZE  = MB(2); //4096;
  static constexpr size_t QUEUE_SIZE        = 32;
  static constexpr size_t POLL_SLEEP_USEC   = 1000; /* 1 ms */
  static constexpr size_t POLL_RETRY_LIMIT  = 1000000;

  static_assert(MAX_MESSAGE_SIZE > 64, "MAX_MESSAGE_SIZE too small");

private:
  const unsigned option_DEBUG = 0;

private:
  ADO_protocol_builder(const ADO_protocol_builder &) = delete;
  ADO_protocol_builder& operator=(const ADO_protocol_builder &) = delete;
  Buffer_header *msg_build_chirp_hello(buffer_space_dedicated_ptr_t && buffer);

  /* get a buffer from the dedicated pool */
  buffer_space_dedicated_ptr_t get_buffer()
  {
    std::lock_guard<std::mutex> g{_b_mutex};
    assert( ! _buffer.empty() );
    auto tmp = _buffer.back().release();
    auto a = buffer_space_dedicated_ptr_t(tmp, this);
    _buffer.pop_back();
    return a;
  }

  /* UIPC helpers */
  /* get a buffer from the shared pool */
  inline buffer_space_shared_ptr_t buffer_allocate()
  {
    auto p = ::uipc_alloc_message(_channel);
    return buffer_space_shared_ptr_t(p, channel_t(_channel));
  }

  inline void send(void * buffer)
  {
    while ( ::uipc_send(_channel, buffer) != S_OK )
    {
      cpu_relax();
    }
  }

  inline status_t recv(channel_t ch, Buffer_header *& out_buffer) __attribute__((warn_unused_result))
  {
    void *v;
    auto rc = ::uipc_recv(ch, &v);
    if ( rc == S_OK )
    {
      out_buffer = static_cast<Buffer_header *>(v);
    }
    return rc;
  }

public:
  enum class Role {
    CONNECT,
    ACCEPT,
  };

  ADO_protocol_builder(unsigned option_debug,
		       const std::string& channel_prefix,
                       Role role);

  ~ADO_protocol_builder();

  void create_uipc_channels();

  status_t recv_bootstrap_response();

  void send_bootstrap(const uint64_t auth_id,
                      const std::string& pool_name,
                      const size_t pool_size,
                      const unsigned int pool_flags,
                      const unsigned int memory_type,
                      const uint64_t expected_obj_count,
                      const bool open_existing);

  void send_bootstrap_response();

  void send_op_event(component::ADO_op event);

  void send_op_event_response(component::ADO_op event);

  void send_cluster_event(const std::string& sender,
			  const std::string& type,
			  const std::string& content);

  void send_shutdown();

  void send_shutdown_to_shard();

  /* shard-side, must not block */
  void send_memory_map(uint64_t token,
                       size_t size,
                       void * value_vaddr);

  /* shard-side, must not block */
  void send_work_request(const uint64_t work_request_key,
                         const char * key,
                         const size_t key_len,
                         const void * value,
                         const size_t value_len,
                         const void * detached_value,
                         const size_t detached_value_len,
                         const void * invocation_data,
                         const size_t invocation_data_len,
                         const bool new_root);

  void send_work_response(status_t status,
                          uint64_t work_key,
                          const component::IADO_plugin::response_buffer_vector_t& response_buffers);

  ssize_t recv_from_proxy(void * target, const size_t target_len);

  /* shard-side, must not block */
  bool recv_from_ado_work_completion(uint64_t& work_key,
                                     status_t& status,
                                     component::IADO_plugin::response_buffer_vector_t& response_buffers);

  /* table operations */
  void send_table_op_create(const uint64_t work_request_id,
                            const std::string& key,
                            const size_t value_len,
                            const std::uint64_t flags);

  void send_table_op_open(const uint64_t work_request_id,
                          const std::string& key,
                          const size_t value_len,
                          const std::uint64_t flags);

  void send_table_op_resize(const uint64_t work_request_id,
                            const std::string& key,
                            const size_t new_value_len);

  void send_table_op_erase(const std::string& key);

  void send_table_op_allocate_pool_memory(const size_t size,
                                          const size_t alignment);

  void send_table_op_free_pool_memory(const void * ptr,
                                      const size_t size);

  void send_find_index_request(const std::string& key_expression,
                               offset_t begin_position,
                               component::IKVIndex::find_t find_type);

  void send_vector_response(const status_t status,
                            const component::IADO_plugin::Reference_vector& rv);

  void send_vector_request(const common::epoch_time_t t_begin,
                           const common::epoch_time_t t_end);

  void recv_vector_response(status_t& status,
                            component::IADO_plugin::Reference_vector& out_vector);

  void send_pool_info_request();

  void send_pool_info_response(const status_t status,
                               const std::string& info);


  void recv_pool_info_response(status_t& status, std::string& out_response);

  /* shard-side, must not block */

  bool recv_vector_request(const Buffer_header * buffer,
                           common::epoch_time_t& t_begin,
                           common::epoch_time_t& t_end);

  bool recv_pool_info_request(const Buffer_header * buffer);

  bool recv_table_op_request(const Buffer_header * buffer,
                             uint64_t& work_request_id,
                             component::ADO_op& op,
                             std::string& key,
                             size_t& value_len,
                             size_t& value_alignment,
                             void*& addr);

  bool recv_op_event_response(const Buffer_header * buffer,
                              component::ADO_op& op);

  bool recv_index_op_request(const Buffer_header * buffer,
                             std::string& key_expression,
                             offset_t& begin_pos,
                             int& find_type);

  void send_table_op_response(const status_t status,
                              const void * value_addr = nullptr,
                              size_t value_len = 0,
                              const char * key_addr = nullptr,
                              component::IKVStore::key_t key_handle = nullptr);

  void recv_table_op_response(status_t& status,
                              void *& out_value_addr,
                              size_t * p_out_value_len = nullptr,
                              const char ** out_key_ptr = nullptr,
                              component::IKVStore::key_t * out_key_handle = nullptr);

  void recv_find_index_response(status_t& status,
                                offset_t& out_matched_position,
                                std::string& out_matched_key);

  void send_find_index_response(const status_t status,
                                const offset_t matched_position,
                                const std::string& matched_key);

  void send_iterate_request(const common::epoch_time_t t_begin,
                            const common::epoch_time_t t_end,
                            component::IKVStore::pool_iterator_t iterator);

  void recv_iterate_response(status_t& status,
                             component::IKVStore::pool_iterator_t& iterator,
                             component::IKVStore::pool_reference_t& reference);

  bool recv_iterate_request(const Buffer_header * buffer,
                            common::epoch_time_t& t_begin,
                            common::epoch_time_t& t_end,
                            component::IKVStore::pool_iterator_t& iterator);

  void send_iterate_response(const status_t rc,
                             component::IKVStore::pool_iterator_t iterator,
                             component::IKVStore::pool_reference_t reference);

  void send_unlock_request(const uint64_t work_id,
                           const component::IKVStore::key_t key_handle);

  bool recv_unlock_response(status_t& status);

  bool recv_unlock_request(const Buffer_header * buffer,
                           uint64_t& work_id,
                           component::IKVStore::key_t& key_handle);

  void send_unlock_response(const status_t status);

  void send_configure_request(const uint64_t options);

  bool recv_configure_request(const Buffer_header * buffer,
                              uint64_t& options);

  void send_configure_response(const status_t status);

  bool recv_configure_response(status_t& status);

  /* free the singleton buffer */
  void free_ipc_buffer(void * p)
  {
    assert(p);
    std::lock_guard<std::mutex> g{_b_mutex};
#ifdef DEBUG
    assert(std::find_if(_buffer.begin(), _buffer.end(),
                         [&] (auto &e) { return e.get() == p; }) == _buffer.end());
#endif
    _buffer.emplace_back(p, channel_t(_channel));
  }

  /* UIPC helpers */
  inline void send_callback(void * buffer)
  {
    while ( ::uipc_send(_channel_callback, buffer) != S_OK )
    {
      cpu_relax();
    }
  }

  inline status_t recv(Buffer_header *& out_buffer) __attribute__((warn_unused_result)) { return recv(_channel, out_buffer); }

  inline status_t recv_callback(Buffer_header *& out_buffer) __attribute__((warn_unused_result))
  {
    return recv(_channel_callback, out_buffer);
  }

  inline size_t max_message_size() const { return MAX_MESSAGE_SIZE; }
  /* out of line, to avoid exposing Buffer_header layout */
  static const uint8_t * buffer_header_to_message(Buffer_header *buffer);

  status_t poll_recv(Buffer_header *& out_buffer) __attribute__((warn_unused_result)) {
    status_t s;
    while((s = recv(out_buffer)) == E_EMPTY) {
      cpu_relax();
    }
    return s;
  }

  status_t poll_recv_sleep(Buffer_header *& out_buffer) __attribute__((warn_unused_result)) {
    status_t s;
    unsigned retries = 0;
    while((s = recv(out_buffer)) == E_EMPTY) {
      if(retries++ < POLL_RETRY_LIMIT)
        cpu_relax();
      else
        usleep(POLL_SLEEP_USEC);
    }
    return s;
  }

  status_t poll_recv_callback(Buffer_header *& out_buffer) __attribute__((warn_unused_result)) {
    status_t s;
    while((s = recv_callback(out_buffer)) == E_EMPTY) {
      cpu_relax();
    }

    return s;
  }

private:
  std::string _channel_prefix;
  Channel_wrap _channel;
  Channel_wrap _channel_callback;
  /* A non-shared buffer "pool," for messages.
   * Required because the ADO interface does not otherwise ensure that a buffer
   * is available. We hope that the ADO protocol will not exhaust the pool.
   */
  std::mutex _b_mutex; // buffer guard
  std::vector<buffer_space_shared_ptr_t> _buffer;
};


#endif
