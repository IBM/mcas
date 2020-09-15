/*
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
#ifndef __ADO_IPC_PROTOCOL_H__
#define __ADO_IPC_PROTOCOL_H__

#include <boost/numeric/conversion/cast.hpp>
#include <api/ado_itf.h>
#include <common/exceptions.h>
#include <common/dump_utils.h>
#include <vector>
#include <string.h>
#include <stdexcept>
#include <cstring>

// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Weffc++"
// #pragma GCC diagnostic ignored "-Wpedantic"

//#define DEBUG_ADO_IPC

namespace mcas
{

namespace ipc
{
enum {
  MSG_TYPE_BOOTSTRAP_REQUEST = 1,
  MSG_TYPE_WORK_REQUEST  = 2,
  MSG_TYPE_WORK_RESPONSE = 3,
  MSG_TYPE_TABLE_OP_REQUEST = 4,
  MSG_TYPE_TABLE_OP_RESPONSE = 5,
  MSG_TYPE_INDEX_REQUEST = 6,
  MSG_TYPE_INDEX_RESPONSE = 7,
  MSG_TYPE_CHIRP = 8,
  MSG_TYPE_MAP_MEMORY = 9,
  MSG_TYPE_VECTOR_REQUEST = 10,
  MSG_TYPE_VECTOR_RESPONSE = 11,
  MSG_TYPE_POOL_INFO_RESPONSE = 12,
  MSG_TYPE_OP_EVENT = 13,
  MSG_TYPE_OP_EVENT_RESPONSE = 14,
  MSG_TYPE_ITERATE_REQUEST = 15,
  MSG_TYPE_ITERATE_RESPONSE = 16,
  MSG_TYPE_UNLOCK_REQUEST = 17,
  MSG_TYPE_CONFIGURE_REQUEST = 18,
  MSG_TYPE_CLUSTER_EVENT = 19,
};

typedef enum {
  HELLO = 1,
  HEARTBEAT,
  OK,
  SHUTDOWN,
  PANIC,
  POOL_INFO_REQUEST,
  UNLOCK_RESPONSE,
  CONFIGURE_RESPONSE,
} chirp_t;

static constexpr uint16_t MAGIC = 0xDEAF;

struct Message {

  Message(uint16_t type_id_p) : magic(MAGIC), type_id(type_id_p)
  {
  }

  template <typename Type>
  const Type *ptr_cast() const
  {
    if (this->type_id != Type::id)
      throw Protocol_exception("expected %s (0x%x) message - got 0x%x",
                               Type::description, Type::id,
                               this->type_id);
    return static_cast<const Type *>(this);
  }

  inline static bool is_valid(const void * buffer) {
    if(buffer == nullptr) throw std::invalid_argument(__FILE__ " Message::is_valid - pointer is null");
    return *(reinterpret_cast<const uint16_t*>(buffer)) == MAGIC;
  }

  inline static uint16_t type(const void * buffer) {
    if(buffer == nullptr) throw std::invalid_argument(__FILE__ " Message::type - pointer is null");
    return (reinterpret_cast<const Message*>(buffer))->type_id;
  }

  uint16_t  magic;
  uint16_t  type_id;  // message type id
};

//-------------

struct Chirp : public Message {
  static constexpr uint8_t id = MSG_TYPE_CHIRP;
  static constexpr const char *description = "mcas::ipc::Chirp";

  Chirp(chirp_t _type, status_t _status = S_OK)
    : Message(id), type(_type), status(_status)
  {
  }

  chirp_t type;
  union {
    status_t status;
  };
};

//-------------

struct Op_event : public Message {
  static constexpr uint8_t id = MSG_TYPE_OP_EVENT;
  static constexpr const char *description = "mcas::ipc::Op_event";

  Op_event(component::ADO_op _op) : Message(id), op(_op)
  {
  }

  component::ADO_op op;
};

//-------------

struct Op_event_response : public Message {
  static constexpr uint8_t id = MSG_TYPE_OP_EVENT_RESPONSE;
  static constexpr const char *description = "mcas::ipc::Op_event_response";

  Op_event_response(component::ADO_op _op) : Message(id), op(_op)
  {
  }

  component::ADO_op op;
};

//-------------

struct Cluster_event : public Message {
  static constexpr uint8_t id = MSG_TYPE_CLUSTER_EVENT;
  static constexpr const char *description = "mcas::ipc::Cluster_event";

  Cluster_event(size_t buffer_size,
		const std::string& sender,
		const std::string& type,
		const std::string& message)
    : Message(id),
      sender_strlen(boost::numeric_cast<uint16_t>(sender.size())),
      type_strlen(boost::numeric_cast<uint16_t>(type.size()))
  {
    if((sizeof(Cluster_event) + sender.size() + type.size() + message.size() + 3)
       > buffer_size)
      throw std::length_error(description);
    char * ptr = data;
    strncpy(ptr, sender.c_str(), sender.size() + 1);
    ptr += sender.size() + 1;
    strncpy(ptr, type.c_str(), type.size() + 1);
    ptr += type.size() + 1;
    strncpy(ptr, message.c_str(), message.size() + 1);
  }

  inline const char * sender() const { return data; }
  inline const char * type() const { return &data[sender_strlen + 1]; }
  inline const char * message() const { return &data[sender_strlen + type_strlen + 2]; }

  uint16_t sender_strlen;
  uint16_t type_strlen;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
  char     data[]; /* strings are null-terminated */
#pragma GCC diagnostic pop
};

//-------------

struct Bootstrap_request : public Message {
  static constexpr uint8_t id = MSG_TYPE_BOOTSTRAP_REQUEST;
  static constexpr const char *description = "mcas::ipc::Bootstrap_request";

  Bootstrap_request(size_t buffer_size,
                    const uint64_t _auth_id,
                    const std::string& _pool_name,
                    const size_t _pool_size,
                    const unsigned int _pool_flags,
                    const unsigned int _memory_type,
                    const uint64_t _expected_obj_count,
                    const bool _open_existing)
    : Message(id),
      auth_id(_auth_id),
      pool_size(_pool_size),
      expected_obj_count(_expected_obj_count),
      pool_name_len(_pool_name.size()),
      pool_flags(_pool_flags),
      memory_type(_memory_type),
      open_existing(_open_existing)
  {
    if((sizeof(Bootstrap_request) + _pool_name.size() + 1) > buffer_size)
      throw std::length_error(description);

    ::memcpy(pool_name, _pool_name.data(), _pool_name.size());
    pool_name[_pool_name.size()] = '\0';
  }

  uint64_t     auth_id;
  size_t       pool_size;
  uint64_t     expected_obj_count;
  size_t       pool_name_len;
  unsigned int pool_flags;
  unsigned int memory_type;
  bool         open_existing;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic" // zero-size array (replace with variable-length region following the class)
  char         pool_name[];
#pragma GCC diagnostic pop

};

//-------------

struct Map_memory : public Message {
  static constexpr uint8_t id = MSG_TYPE_MAP_MEMORY;
  static constexpr const char *description = "mcas::ipc::Map_memory";

  Map_memory(size_t buffer_size,
             uint64_t _token,
             size_t _size,
             void * _shard_addr)
    : Message(id), token(_token), size(_size), shard_addr(_shard_addr)
  {
    if(sizeof(Map_memory) > buffer_size)
      throw std::length_error(description);
  }

  uint64_t token;
  size_t   size;
  void *   shard_addr;

};

//-------------

struct Work_request : public Message {
  static constexpr uint8_t id = MSG_TYPE_WORK_REQUEST;
  static constexpr const char *description = "mcas::ipc::Work_request";

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++" // key_ptr uninitialized - unclear whether it has a purpose
  Work_request(size_t buffer_size,
               const uint64_t _work_key,
               const char * _key,
               const uint64_t _key_len,
               const uint64_t _value_addr,
               const uint64_t _value_len,
               const uint64_t _detached_value_addr,
               const uint64_t _detached_value_len,
               const void * _invocation_data,
               const size_t _invocation_data_len,
               const bool _new_root)
    : Message(id),
      work_key(_work_key),
      key_addr(reinterpret_cast<uint64_t>(_key)),
      key_len(_key_len),
      value_addr(_value_addr),
      value_len(_value_len),
      detached_value_addr(_detached_value_addr),
      detached_value_len(_detached_value_len),
      invocation_data_len(_invocation_data_len),
      new_root(_new_root)
  {
    assert(detached_value_addr ? detached_value_len > 0 : true);
    // bounds check
    if((_invocation_data_len + _key_len + sizeof(Work_request)) > buffer_size)
      throw std::length_error(description);
    ::memcpy(data, _invocation_data, invocation_data_len);
  }
#pragma GCC diagnostic pop

  inline size_t get_key_len() const { return key_len; }
  inline const char * get_key() const { return reinterpret_cast<const char*>(key_addr); }
  inline size_t get_invocation_data_len() const { return invocation_data_len; }
  inline const char * get_invocation_data() const { return static_cast<const char*>(static_cast<const void *>(&data[0])); }
  inline void * get_value_addr() const { return reinterpret_cast<void *>(value_addr); }
  inline void * get_detached_value_addr() const { return reinterpret_cast<void *>(detached_value_addr); }

  uint64_t work_key;
  uint64_t key_addr;
  uint64_t key_len;
  uint64_t value_addr;
  uint64_t value_len;
  uint64_t key_ptr;
  uint64_t detached_value_addr;
  uint64_t detached_value_len;
  uint64_t invocation_data_len;
  bool     new_root;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic" // zero-size array (replace with variable-length region following the class)
  char     data[];
#pragma GCC diagnostic pop

};



//-------------

struct Work_response : public Message {
  static constexpr uint8_t id = MSG_TYPE_WORK_RESPONSE;
  static constexpr const char *description = "mcas::ipc::Work_response";

  Work_response(size_t buffer_size,
                const uint64_t _work_key,
                const status_t _status,
                const component::IADO_plugin::response_buffer_vector_t& response_buffers)
    : Message(id), work_key(_work_key), status(_status),
      count(0),
      response_len(0)
  {
    using namespace component;
    assert(buffer_size > 0);

    assert(data);
    char * data_ptr = data;
    char *const data_ptr_begin = data_ptr;
    /* first copy over response buffers */

    for(auto i=response_buffers.begin(); i!=response_buffers.end(); i++) {
      const IADO_plugin::response_buffer_t& b = *i;

      /* copy over response buffer record */
      assert(b.ptr);

      /* clone response buffer record */
      void *const r_ptr = data_ptr;
      new (r_ptr) IADO_plugin::response_buffer_t(
        b
        , [&b, &data_ptr] (const IADO_plugin::response_buffer_t &b2) -> void *
          {
            ::memcpy(data_ptr + sizeof(IADO_plugin::response_buffer_t), b2.ptr, b2.len);
            data_ptr += b2.len;
            return nullptr;
          }
      );

      data_ptr += sizeof(IADO_plugin::response_buffer_t);

#if defined DEBUG_ADO_IPC
      PLOG("Work_response: adding to IPC response (pool=%s) %p : %lu type=%d",
           b.is_pool() ? "y":"n", b.ptr, b.len, b.alloc_type);
      //if(b.len > 0) hexdump(b.ptr, b.len);
#endif

      count++;
    }

    response_len += (data_ptr - data_ptr_begin);

    /* if this happens, corruption already occurred */
    if((response_len + sizeof(Work_response)) > buffer_size)
      throw Logic_exception("IPC response too large");
  }

  size_t get_message_size() const { return sizeof(Work_response) + response_len; }

  void copy_responses(component::IADO_plugin::response_buffer_vector_t& out_vector) const {

    auto data_ptr = data;

    for(uint32_t i=0; i<count; i++) {
      auto record = reinterpret_cast<const component::IADO_plugin::response_buffer_t*>(data_ptr);
      out_vector.emplace_back(
        *record
        , [&record, &data_ptr] (const component::IADO_plugin::response_buffer_t &src) -> void *
          {
            assert(src.ptr == nullptr);
            auto dst_ptr = ::malloc(src.len);
            if ( dst_ptr == nullptr )
            {
              throw std::bad_alloc();
            }
            ::memcpy(dst_ptr, &record[1], src.len);
            data_ptr += src.len;
            return dst_ptr;
          }
      );

      data_ptr += sizeof(component::IADO_plugin::response_buffer_t);
    }
  }

  uint64_t work_key;
  int32_t  status;
  uint32_t count;
  uint64_t response_len;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic" // zero-size array
  char     data[]; /* pool buffer vector, followed by response data */
#pragma GCC diagnostic pop
};


//-------------

struct Table_request : public Message {
  static constexpr uint8_t id = MSG_TYPE_TABLE_OP_REQUEST;
  static constexpr const char *description = "mcas::ipc::Table_request";

  Table_request(size_t buffer_size,
                uint64_t _work_key,
                const std::string& _key,
                uint64_t _value_len,
                uint64_t _addr,
                uint64_t _align_or_flags,
                component::ADO_op _op)
    : Message(id), work_key(_work_key),value_len(_value_len),
      key_len(_key.length()), addr(_addr),
      align_or_flags(_align_or_flags), op(_op)
  {
    if((_key.size() + sizeof(Table_request)) > buffer_size)
      throw std::length_error(description);
    ::memcpy(key, _key.data(), _key.size());
  }

  Table_request(size_t buffer_size,
                uint64_t _work_key,
                uint64_t _value_len,
                uint64_t _addr,
                uint64_t _align_or_flags,
                component::ADO_op _op)
    : Message(id), work_key(_work_key),value_len(_value_len),
      key_len(0), addr(_addr),
      align_or_flags(_align_or_flags), op(_op)
  {
    if(sizeof(Table_request) > buffer_size)
      throw std::length_error(description);
  }

  uint64_t          work_key;
  uint64_t          value_len;
  uint64_t          key_len;
  uint64_t          addr;
  uint64_t          align_or_flags;
  component::ADO_op op;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic" // zero-size array
  char              key[];
#pragma GCC diagnostic pop

};


//-------------

struct Table_response : public Message {
  static constexpr uint8_t id = MSG_TYPE_TABLE_OP_RESPONSE;
  static constexpr const char *description = "mcas::ipc::Table_response";

  Table_response(size_t buffer_size,
                 status_t _status,
                 uint64_t _value_addr,
                 uint64_t _value_len,
                 uint64_t _key_addr,
                 uint64_t _key_handle)
    : Message(id),
      value_addr(_value_addr),
      value_len(_value_len),
      key_addr(_key_addr),
      key_handle(_key_handle),
      status(_status)
  {
    if(sizeof(Table_response) > buffer_size)
      throw std::length_error(description);
  }

  uint64_t value_addr;
  uint64_t value_len;
  uint64_t key_addr;
  uint64_t key_handle;
  status_t status;

};


//-------------

struct Index_request : public Message {
  static constexpr uint8_t id = MSG_TYPE_INDEX_REQUEST;
  static constexpr const char *description = "mcas::ipc::Index_request";

private:
  static auto checked_expr_len(const std::string& expr_, size_t buffer_size_)
  {
    if ((sizeof(Index_request) + expr_.size()) > buffer_size_)
      throw std::length_error(description);
    return expr_.size();
  }

public:
  Index_request(size_t buffer_size,
                uint64_t _begin_pos,
                component::IKVIndex::find_t _find_type,
                const std::string& _expr)
    : Message(id), begin_pos(_begin_pos),
      expr_len(checked_expr_len(_expr, buffer_size)),
      find_type(_find_type)
  {
    ::memcpy(expr, _expr.data(), expr_len);
  }

  uint64_t begin_pos;
  uint64_t expr_len;
  component::IKVIndex::find_t find_type;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic" // zero-size array
  char expr[];
#pragma GCC diagnostic pop

};


//-------------

struct Index_response : public Message {
  static constexpr uint8_t id = MSG_TYPE_INDEX_RESPONSE;
  static constexpr const char *description = "mcas::ipc::Index_response";

  Index_response(size_t buffer_size,
                 uint64_t _matched_pos,
                 status_t _status,
                 const std::string _matched_key)
    : Message(id), matched_pos(_matched_pos),
      matched_key_len(_matched_key.size()), status(_status)
  {
    if((sizeof(Index_response) + _matched_key.size()) > buffer_size)
      throw std::length_error(description);

    ::memcpy(matched_key, _matched_key.data(), matched_key_len);
  }

  uint64_t matched_pos;
  uint64_t matched_key_len;
  status_t status;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic" // zero-size array
  char matched_key[];
#pragma GCC diagnostic pop

};


//-------------

struct Vector_request : public Message {
  static constexpr uint8_t id = MSG_TYPE_VECTOR_REQUEST;
  static constexpr const char *description = "mcas::ipc::Vector_request";

  Vector_request(const common::epoch_time_t _t_begin,
                 const common::epoch_time_t _t_end)
    : Message(id), t_begin(_t_begin), t_end(_t_end)
  {
  }

  common::epoch_time_t t_begin;
  common::epoch_time_t t_end;

};

//-------------

struct Vector_response : public Message {
  static constexpr uint8_t id = MSG_TYPE_VECTOR_RESPONSE;
  static constexpr const char *description = "mcas::ipc::Vector_request";

  Vector_response(status_t _status,
                  const component::IADO_plugin::Reference_vector& _refs)
    : Message(id), status(_status), refs(_refs)
  {
  }

  status_t                                 status;
  component::IADO_plugin::Reference_vector refs;

};


//-------------

struct Pool_info_response : public Message {
  static constexpr uint8_t id = MSG_TYPE_POOL_INFO_RESPONSE;
  static constexpr const char *description = "mcas::ipc::Pool_info_request";

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++" // result_len uninitialized - unclear whether it has a purpose
  Pool_info_response(size_t buffer_size,
                     status_t _status,
                     const std::string& _result)
    : Message(id), status(_status)
  {
    if((sizeof(Pool_info_response) + _result.size() + 1) > buffer_size)
      throw std::length_error(description);

    ::memcpy(result, _result.data(), _result.size());
    result[_result.size()] = '\0';
  }
#pragma GCC diagnostic pop

  status_t status;
  size_t   result_len;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic" // zero-size array
  char     result[];
#pragma GCC diagnostic pop

};


//-------------

struct Iterate_request : public Message {
  static constexpr uint8_t id = MSG_TYPE_ITERATE_REQUEST;
  static constexpr const char *description = "mcas::ipc::Iterate_request";

  Iterate_request(const common::epoch_time_t _t_begin,
                  const common::epoch_time_t _t_end,
                  component::IKVStore::pool_iterator_t _iterator)
    : Message(id), t_begin(_t_begin), t_end(_t_end), iterator(_iterator)
  {
  }

  const common::epoch_time_t t_begin;
  const common::epoch_time_t t_end;
  component::IKVStore::pool_iterator_t iterator;

};


struct Iterate_response : public Message {
  static constexpr uint8_t id = MSG_TYPE_ITERATE_RESPONSE;
  static constexpr const char *description = "mcas::ipc::Iterate_response";

  Iterate_response(status_t _status,
                   const component::IKVStore::pool_iterator_t _iterator,
                   const component::IKVStore::pool_reference_t& _reference)
    : Message(id), status(_status), iterator(_iterator), reference(_reference)
  {
  }

  status_t                              status;
  component::IKVStore::pool_iterator_t  iterator;
  component::IKVStore::pool_reference_t reference;

};


//-------------

struct Unlock_request : public Message {
  static constexpr uint8_t id = MSG_TYPE_UNLOCK_REQUEST;
  static constexpr const char *description = "mcas::ipc::Unlock_request";

  Unlock_request(const uint64_t _work_id,
                 const component::IKVStore::key_t _key_handle)
    : Message(id), work_id(_work_id), key_handle(_key_handle)
  {
    assert(work_id);
    assert(key_handle);
  }

  const uint64_t                   work_id;
  const component::IKVStore::key_t key_handle;

};

//---------------
struct Configure_request : public Message {
  static constexpr uint8_t id = MSG_TYPE_CONFIGURE_REQUEST;
  static constexpr const char *description = "mcas::ipc::Configure_request";

  Configure_request(const uint64_t _options)
    : Message(id), options(_options)
  {
    assert(options);
  }

  const uint64_t                   options;

};


} // ipc
} // mcas

#endif
