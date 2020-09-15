/*
   Copyright [2019] [IBM Corporation]
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

/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 * Luna Xu (xuluna@ibm.com)
 *
 */

#ifndef __API_ADO_ITF_H__
#define __API_ADO_ITF_H__

#include <api/kvindex_itf.h>
#include <api/kvstore_itf.h>
#include <common/errors.h>
#include <common/types.h>
#include <common/time.h>
#include <component/base.h>

#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <vector>

class Buffer_header;

namespace component
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

class SLA; /* to be defined - placeholder only */

enum class ADO_op {
  UNDEFINED = 0,
  CREATE = 1,
  OPEN = 2,
  CLOSE = 3,
  ERASE = 4,
  VALUE_RESIZE = 5,
  ALLOCATE_POOL_MEMORY = 6,
  FREE_POOL_MEMORY = 7,
  SHUTDOWN = 8,
  POOL_DELETE = 9,
};

#define MDECL(X)  \
  case ADO_op::X: \
    return #X;

inline std::string to_str(ADO_op op)
{
  switch (op) {
    MDECL(UNDEFINED);
    MDECL(CREATE);
    MDECL(OPEN);
    MDECL(CLOSE);
    MDECL(ERASE);
    MDECL(VALUE_RESIZE);
    MDECL(ALLOCATE_POOL_MEMORY);
    MDECL(FREE_POOL_MEMORY);
    MDECL(SHUTDOWN);
    MDECL(POOL_DELETE);
    default:
      return "UNKNOWN";
  }
}

#undef MDECL

/**
 * Component for ADO process plugins
 *
 */
class IADO_plugin : public component::IBase {
 public:
  // clang-format off
  DECLARE_INTERFACE_UUID(0xacb20ef2,0xe796,0x4619,0x845d,0x4e,0x8e,0x6b,0x5c,0x35,0xaa);
  // clang-format on

  /* return S_ERASE_TARGET from do_work to force target pair to be
     deleted after do_work calls
  */
  static constexpr int S_ERASE_TARGET = S_USER0;

  /* Used to define the set of buffers to return to client. The first
     buffer in the vector will be copied into the return message and
     freed. Subsequent are passed as references to the pool
  */
  struct response_buffer_t {

    /* Compile time ALLOC_TYPE differentiators. */
    struct alloc_type_malloc {}; /* use in place of response_buffer_t(..., false) */
    struct alloc_type_pool {}; /* use in place of response_buffer_t(..., true */
  private:
    enum class alloc_type_t : std::uint32_t {
      POOL         = 0,
      MALLOC       = 1,
      INLINE       = 3,
      POOL_TO_FREE = 4, /* buffer is pool memory which should be freed */
    };

    response_buffer_t(void* ptr_p,
                      size_t len_p,
                      uint32_t layer_id_,
                      alloc_type_t type)
      : ptr(ptr_p), len(len_p), layer_id(layer_id_), _alloc_type(type) {
    }

    /* An obfuscated (possibly outdated) version of
     * response_buffer_t(p, len_p, alloc_type_t::MALLOC (or not alloc_type_t::POOL).
     * Preserved as an undefined function to catch accidental match with the
     * uint32_t parameter constructor.
     */
    response_buffer_t(void* ptr_p,
                      size_t len_p,
                      bool pool_ref); /* false => alloc_type_malloc, true => alloc_type_pool */

  public:
    response_buffer_t(const void * addr_)
      : addr(addr_), len(0), layer_id(0), _alloc_type(alloc_type_t::INLINE) {
    }

    response_buffer_t(void* ptr_p,
                      size_t len_p,
                      alloc_type_malloc)
      : response_buffer_t(ptr_p, len_p, 0, alloc_type_t::MALLOC)
    {}

    response_buffer_t(void* ptr_p,
                      size_t len_p,
                      alloc_type_pool)
      : response_buffer_t(ptr_p, len_p, 0, alloc_type_t::POOL)
    {}

    response_buffer_t(const response_buffer_t &) = delete;
    response_buffer_t &operator=(const response_buffer_t &) = delete;

    response_buffer_t(const response_buffer_t &other, std::function<void *(const response_buffer_t &src)> copy_data_)
      : response_buffer_t( (other.is_malloc() ? copy_data_(other) : other.ptr), other.len, other.layer_id, other._alloc_type)
    {
    }

    response_buffer_t(response_buffer_t &&other) noexcept
      : len(other.len)
      , layer_id(other.layer_id)
      , _alloc_type(other._alloc_type)
    {
      switch (_alloc_type)
      {
      case alloc_type_t::INLINE:
        addr = other.addr;
        break;
      default:
        ptr = other.ptr;
        break;
      }
      other.ptr = nullptr; /* safety */
    }

    response_buffer_t &operator=(response_buffer_t &&other) noexcept
    {
      len = other.len;
      layer_id = other.layer_id;
      _alloc_type = other._alloc_type;
      switch (_alloc_type)
      {
      case alloc_type_t::INLINE:
        addr = other.addr;
        break;
      default:
        ptr = other.ptr;
        break;
      }
      other.ptr = nullptr;
      return *this;
    }

    inline bool is_pool() const { return _alloc_type == alloc_type_t::POOL || _alloc_type == alloc_type_t::POOL_TO_FREE; }
    inline bool is_malloc() const { return _alloc_type == alloc_type_t::MALLOC; }
    inline bool is_inline() const { return _alloc_type == alloc_type_t::INLINE; }
    inline bool is_pool_to_free() const { return _alloc_type == alloc_type_t::POOL_TO_FREE; }
    inline void set_pool_to_free() { _alloc_type = alloc_type_t::POOL_TO_FREE; }
    inline uint64_t get_len() const { return len; }
    uint32_t alloc_type() const { return uint32_t(_alloc_type); }

    union {
      uint64_t    offset;
      void*       ptr;
      const void* addr; /* responding with an address */
    };

    uint64_t len = 0;
    uint32_t layer_id;
  private:
    alloc_type_t _alloc_type;
  public:
    ~response_buffer_t()
    {
      if ( is_malloc() ) { ::free(ptr); }
    }
  };

  struct response_buffer_vector_t : private std::vector<response_buffer_t> {
  private:
    using base = std::vector<response_buffer_t>;
    using base::push_back; /* If code uses this, change it to use emplace_back */
   public:
    using base::begin;
    using base::end;
    using base::emplace_back;
    using base::operator[];
    using base::clear;
    using base::size;
  };

  struct value_t {
    void*  ptr;
    size_t len;
  };

  struct value_space_t : public std::vector<value_t> {
    inline void append(void* ptr, size_t len) { push_back({ptr, len}); }
  };

 public:
  IADO_plugin() : _cb{} {}

  /**
   * Inform the plugin of memory mapped into the ADO process (normally pool
   * memory)
   *
   * @param shard_vaddr Virtual address as mapped in shard
   * @param local_vaddr Virtual address as mapped in local ADO process
   * @param len Size of mapping in bytes
   *
   * @return S_OK on success
   */
  virtual status_t register_mapped_memory(void* shard_vaddr, void* local_vaddr, size_t len) = 0;

  /**
   * Upcall to perform ADO operation on a specific key-value pair
   *
   * @param work_id Work identifier
   * @param key Key pointer for target
   * @param key_len Length of key in bytes
   * @param values Set of pointer,length pairs for associated value memory.
   * First in vector is attached to the key.
   * @param in_work_request Open protocol request message
   * @param in_work_request_len Open protocol request message length
   * @param new_root Set to true if a new root value was created
   * @param response_buffers Buffers to transmit, in order, for response
   *
   * @return ADO plugin response. S_ERASE_TARGET to erase the target pair on return
   */
  virtual status_t do_work(const uint64_t              work_id,
                           const char*                 key,
                           const size_t                key_len,
                           IADO_plugin::value_space_t& values,
                           const void*                 in_work_request,
                           const size_t                in_work_request_len,
                           const bool                  new_root,
                           response_buffer_vector_t&   response_buffers) = 0;

  /**
   * Upcall initial launch event
   *
   * @param auth_id Authentication identifier
   * @param pool_name Name of pool
   * @param pool_size Size of pool in bytes
   * @param pool_flags Pool creation/open flags
   * @param memory_type Memory type (0=DRAM, 1=PM)
   * @param expected_obj_count Client-provided object count
   * @param params Plugin parameters
   *
   */
  virtual void launch_event(const uint64_t                  auth_id,
                            const std::string&              pool_name,
                            const size_t                    pool_size,
                            const unsigned int              pool_flags,
                            const unsigned int              memory_type,
                            const size_t                    expected_obj_count,
                            const std::vector<std::string>& params) {}

  /**
   * Upcall for operational events
   *
   * @param op Event
   */
  virtual void notify_op_event(ADO_op op) {} /* see ADO::OP_XXX */


  /**
   * Notify ADO of a MCAS-level clustering event
   *
   * @param sender Message sender
   * @param type Message type
   * @param message Message content
   */
  virtual void cluster_event(const std::string& sender,
			     const std::string& type,
			     const std::string& message) {}


  /* note:     FLAGS_CREATE_ONLY = 0x4, */
  enum : int {
    FLAGS_ADO_LIFETIME_UNLOCK = 0x21,
    FLAGS_NO_IMPLICIT_UNLOCK  = 0x22,
  };

  enum : uint64_t {
    CONFIG_SHARD_INC_REF = 0x1,
    CONFIG_SHARD_DEC_REF = 0x2,
  };

  typedef struct {
    void*  key;
    size_t key_len;
    void*  value;
    size_t value_len;
  } kv_reference_t;

  /* Reference vector holds pointer to value memory held key-ptr,
     key-size, val-ptr, val-size for all pairs */
  class Reference_vector {
   public:
    Reference_vector(const Reference_vector& src)
        : _count(src._count), _value_memory_array(src._value_memory_array), _value_memory_size(src._value_memory_size)
    {
    }

    Reference_vector(const size_t count, kv_reference_t* value_memory_array, const size_t value_memory_size)
        : _count(count), _value_memory_array(value_memory_array), _value_memory_size(value_memory_size)
    {
    }

    Reference_vector(const size_t count, void* value_memory_array, const size_t value_memory_size)
        : _count(count), _value_memory_array(static_cast<kv_reference_t*>(value_memory_array)),
          _value_memory_size(value_memory_size)
    {
    }

    explicit Reference_vector() : _count(0), _value_memory_array(nullptr), _value_memory_size(0) {}

    Reference_vector& operator=(const Reference_vector&) = default;

    static size_t size_required(size_t num_pairs)
    {
      return sizeof(Reference_vector) + (num_pairs * sizeof(kv_reference_t));
    }

    inline size_t                value_memory_size() const { return _value_memory_size; }
    inline const kv_reference_t* ref_array() const { return _value_memory_array; }
    inline size_t                count() const { return _count; }
    inline void*                 value_memory() const { return static_cast<void*>(_value_memory_array); }

   private:
    size_t          _count;
    kv_reference_t* _value_memory_array;
    size_t          _value_memory_size;
  } __attribute__((packed));

  struct Callback_table {
    /**
     * Create a new key-value pair. Implicitly take a lock (default releases at
     *end of ADO invoke)
     *
     * @param work_id Work identifier from ADO invocation. This is
     *                used to manage the lock. If work_id is 0
     *                then a permanent lock is taken.
     * @param key_name Name of key
     * @param value_size Size of value
     * @param flags Optional IADO_plugin::FLAGS_ADO_LIFETIME_UNLOCK (retains for lifetime of pool)
     * @param out_value_addr [out] Newly created value address
     * @param out_key_ptr [optional] Out address of key (fixed for duration of
     *held lock)
     * @param out_key_handle [optional] Key handle for subsequent unlock
     *
     * @return : S_OK, E_ALREADY_EXISTS
     **/
    std::function<status_t(const uint64_t              work_id,
                           const std::string&          key_name,
                           const size_t                value_size,
                           const int                   flags,
                           void*&                      out_value_addr,
                           const char**                out_key_ptr,
                           component::IKVStore::key_t* out_key_handle)>
        create_key;

    /**
     * Open existing key-value pair. Implicitly take a lock (default releases at
     *end of ADO invoke)
     *
     * @param work_id Work identifier from ADO invocation. This is
     *                used to manage the lock. If work_id is 0
     *                then a permanent lock is taken.
     * @param key_name Name of key
     * @param flags Optional IADO_plugin::FLAGS_ADO_LIFETIME_UNLOCK
     * @param out_value_addr [out] Value address
     * @param out_value_len [out] Size of value
     * @param out_key_ptr [optional] Out address of key (fixed for duration of
     *held lock)
     * @param out_key_handle [optional] Key handle for subsequent unlock
     *
     * @return : S_OK, E_INVAL (result from IKVStore::lock)
     **/
    std::function<status_t(const uint64_t              work_id,
                           const std::string&          key_name,
                           const int                   flags,
                           void*&                      out_value_addr,
                           size_t&                     out_value_len,
                           const char**                out_key_ptr,
                           component::IKVStore::key_t* out_key_handle)>
       open_key;

    /**
     * Erase existing key-value pair from store and index
     *
     * @param key_name Name of key
     *
     * @return : S_OK, E_INVAL or result from IKVStore::erase
     **/
    std::function<status_t(const std::string& key_name)>
        erase_key;

    /**
     * Resize existing value (may perform memcpy)
     *
     * @param work_id Work identifier from ADO invocation
     * @param key_name Name of key
     * @param new_value_size Size to resize to
     * @param out_value_addr [out] New value address (relocation may have
     *occurred)
     *
     * @return : S_OK, E_INVAL, or error from IKVStore::lock or IKVStore::resize
     **/
    std::function<status_t(const uint64_t     work_id,
                           const std::string& key_name,
                           const size_t       new_value_size,
                           void*&             out_new_value_addr)>
        resize_value;

    /**
     * Allocate pool memory. Memory is not attached to key.
     *
     * @param size Size to allocate in bytes
     * @param alignment_hint Hint for alignment (cannot be guaranteed)
     * @param out_value_addr [out] Pointer to allocated region
     *
     * @return : S_OK, E_INVAL, or result from IKVStore::allocate_pool_memory
     **/
    std::function<status_t(const size_t size,
                           const size_t alignment_hint,
                           void*& out_new_addr)>
        allocate_pool_memory;

    /**
     * Free pool memory. Memory should not be referenced/attached.
     *
     * @param size Size of memory to free
     * @param out_value_addr [out] Pointer to allocated region
     *
     * @return : S_OK, or result from IKVStore::free_pool_memory
     **/
    std::function<status_t(const size_t size, const void* addr)>
        free_pool_memory;

    /**
     * Get vector of all key-value references.  Optionally filter based on
     * write timestamp. The vector is actually
     * held in value memory and should be free by ADO plugin.
     *
     * @param t_begin Optional time begin constraint (zero for no constraint)
     * @param t_end Optional time end constraint (zero for no constraint)
     * @param out_vector [out] Pointer to vector of references
     *              held in pool memory which must be freed from ADO
     *
     * @return : S_OK, E_INSUFFICIENT_SPACE
     **/
    std::function<status_t(const common::epoch_time_t t_begin,
                           const common::epoch_time_t t_end,
                           Reference_vector& out_vector)>
        get_reference_vector;

    /**
     * Scan for key (requires index configured)
     *
     * @param work_id Work identifier from ADO invocation
     * @param key_expression Regular expression
     * @param begin_position Position to search from
     * @param find_type (see kvindex_itf.h)
     * @param [out] Position of match or point at which max comparisons reached.
     * @param [out] Matched key
     *
     * @return : S_OK, E_NO_INDEX, E_MAX_REACHED (need to recall with new
     *offset)
     **/
    std::function<status_t(const std::string&     key_expression,
                           const offset_t         begin_position,
                           const IKVIndex::find_t find_type,
                           offset_t&              out_matched_position,
                           std::string&           out_matched_key)>
        find_key;

    /**
     * Get pool information, e.g., size, flags, expected_obj_count.
     * Note: this information is currently for *only* when the pool was
     * created. If the pool is resized, or reopened, then some of this
     * information may be out of date.
     *
     * @param out_result [out] std::string in JSON format
     *
     * @return : S_OK or E_FAIL
     */
    std::function<status_t(std::string& out_result)>
        get_pool_info;

    /**
     * Iterate on pool key-value pairs
     *
     * @param t_begin Optional time begin constraint (zero for no constraint)
     * @param t_end Optional time end constraint (zero for no constraint)
     * @param iterator [inout] Iterator handle. If zero, open iterator and
     * advance to the first element.
     * @param reference [out] Reference information for key-value pair
     *
     * @return S_OK on success and valid reference, E_INVAL (bad iterator),
     *   E_OUT_OF_BOUNDS (when attempting to dereference out of bounds
     *   E_ITERATOR_DISTURBED (when writes have been made since last iteration)
     */
    std::function<status_t(const common::epoch_time_t             t_begin,
                           const common::epoch_time_t             t_end,
                           component::IKVStore::pool_iterator_t&  iterator,
                           component::IKVStore::pool_reference_t& reference)>
        iterate;

    /**
     * Explicitly unlock a key-value pair
     *
     * @param work_id Work identifier from ADO invocation
     * @param key_handle Lock handle attained from open/create
     *
     * @return : S_OK or E_INVAL if not locked with FLAGS_NO_IMPLICIT_UNLOCK
     */
    std::function<status_t(const uint64_t work_id,
                           const component::IKVStore::key_t key_handle)>
        unlock;


    /**
     * Send configuration option to shard
     */
    std::function<status_t(const uint64_t option)>
        configure;
  };

  /**------------------------------------------------------------------------------
       Call back API wrappers - used to support optional parameters on
     call-backs
     ------------------------------------------------------------------------------
  */
  inline status_t cb_create_key(const uint64_t              work_id,
                                const std::string&          key_name,
                                const size_t                value_size,
                                const int                   flags,
                                void*&                      out_value_addr,
                                const char**                out_key_ptr    = nullptr,
                                component::IKVStore::key_t* out_key_handle = nullptr)
  {
    return _cb.create_key(work_id, key_name, value_size, flags, out_value_addr, out_key_ptr, out_key_handle);
  }


  inline status_t cb_open_key(const uint64_t              work_id,
                              const std::string&          key_name,
                              const int                   flags,
                              void*&                      out_value_addr,
                              size_t&                     out_value_len,
                              const char**                out_key_ptr    = nullptr,
                              component::IKVStore::key_t* out_key_handle = nullptr)
  {
    return _cb.open_key(work_id, key_name, flags, out_value_addr, out_value_len, out_key_ptr, out_key_handle);
  }

  inline status_t cb_open_key(const uint64_t              work_id,
                              const std::string&          key_name,
                              void*&                      out_value_addr,
                              size_t&                     out_value_len,
                              const char**                out_key_ptr    = nullptr,
                              component::IKVStore::key_t* out_key_handle = nullptr)
{
    return _cb.open_key(work_id, key_name, 0, out_value_addr, out_value_len, out_key_ptr, out_key_handle);
  }

  inline status_t cb_erase_key(const std::string& key_name) { return _cb.erase_key(key_name); }

  inline status_t cb_resize_value(const uint64_t     work_id,
                                  const std::string& key_name,
                                  const size_t       new_value_size,
                                  void*&             out_new_value_addr)
  {
    return _cb.resize_value(work_id, key_name, new_value_size, out_new_value_addr);
  }

  inline status_t cb_allocate_pool_memory(const size_t size, const size_t alignment_hint, void*& out_new_addr)
  {
    return _cb.allocate_pool_memory(size, alignment_hint, out_new_addr);
  }

  inline status_t cb_free_pool_memory(const size_t size, const void* addr) { return _cb.free_pool_memory(size, addr); }

  inline status_t cb_get_reference_vector(const common::epoch_time_t t_begin,
                                          const common::epoch_time_t t_end,
                                          Reference_vector&  out_vector)
  {
    return _cb.get_reference_vector(t_begin, t_end, out_vector);
  }

  inline status_t cb_find_key(const std::string&     key_expression,
                              const offset_t         begin_position,
                              const IKVIndex::find_t find_type,
                              offset_t&              out_matched_position,
                              std::string&           out_matched_key)
  {
    return _cb.find_key(key_expression, begin_position, find_type, out_matched_position, out_matched_key);
  }

  inline status_t cb_get_pool_info(std::string& out_result) { return _cb.get_pool_info(out_result); }

  inline status_t cb_iterate(const common::epoch_time_t             t_begin,
                             const common::epoch_time_t             t_end,
                             component::IKVStore::pool_iterator_t&  iterator,
                             component::IKVStore::pool_reference_t& reference)
  {
    return _cb.iterate(t_begin, t_end, iterator, reference);
  }

  inline status_t cb_unlock(const uint64_t work_id, const component::IKVStore::key_t key_handle)
  {
    return _cb.unlock(work_id, key_handle);
  }

  inline status_t cb_configure(const std::uint32_t options)
  {
    return _cb.configure(options);
  }

  /**
   * Register callbacks, so the plugin can perform KV-pair operations (sent to
   * shard to perform)
   *
   * @param create_key Callback function to create a new key in the current pool
   * @param erase_key Callback function to erase a key from the current pool
   */
  void register_callbacks(const Callback_table& callbacks) { _cb = callbacks; }

  /**
   * Request graceful shutdown. This will be called by ADO process on shutdown.
   *
   *
   * @return S_OK on success
   */
  virtual status_t shutdown() = 0;

 protected:
  IADO_plugin::Callback_table _cb; /*< callback functions */
};

/**
 * ADO interface.  This is actually a proxy interface communicating with the
 * external process.
 *
 */
class IADO_proxy : public component::IBase {
 public:
  // clang-format off
  DECLARE_INTERFACE_UUID(0xbbbfa389,0x1665,0x4e5b,0xa1b1,0x3c,0xff,0x4a,0x5e,0xe2,0x63);
  // clang-format on

  using work_id_t = uint64_t; /*< work handle/identifier */

  /* ADO-to-SHARD (and vice versa) protocol */

  /**
   * Send boostrap event to ADO
   *
   * @param opened_existing True if open, false if create
   *
   * @return S_OK on success
   */
  virtual status_t bootstrap_ado(bool opened_existing) = 0;

  /**
   * Send operation to ADO
   *
   * @return S_OK on success
   */
  virtual status_t send_op_event(ADO_op op) = 0;


  /**
   * Send a clustering event to ADO
   *
   * @param sender Sender UUID
   * @param type Type/command
   * @param content Content of message
   *
   * @return S_OK on success
   */
  virtual status_t send_cluster_event(const std::string& sender,
				      const std::string& type,
				      const std::string& content) = 0;

  /**
   * Send a memory map request to ADO
   *
   * @param token Token from MCAS expose_memory call
   * @param size Size of the memory
   * @param value_vaddr Virtual address as mapped by shard (may be different in
   * ADO)
   *
   * @return S_OK on success
   */
  virtual status_t send_memory_map(uint64_t token, size_t size, void* value_vaddr) = 0;

  /**
   * Send a work request to the ADO
   *
   * @param work_request_key Unique request identifier
   * @param key Pointer to key
   * @param key_len Length of key in bytes
   * @param value_addr Pointer to value (as mapped by shard)
   * @param value_len Length of value in bytes
   * @param detached_value_addr Pointer to detached value (as mapped by shard)
   * @param detached_value_len Length of detached value in bytes
   * @param invocation_data Data representing the work request (may be binary or
   * string)
   * @param invocation_len Length of data representing work
   * @param new_root Set true if a new root value was created
   *
   * @return S_OK on success
   */
  virtual status_t send_work_request(const uint64_t work_request_key,
                                     const char*    key,
                                     const size_t   key_len,
                                     const void*    value_addr,
                                     const size_t   value_len,
                                     const void*    detached_value,
                                     const size_t   detached_value_len,
                                     const void*    invocation_data,
                                     const size_t   invocation_len,
                                     const bool     new_root) = 0;

  /**
   * Check for completion of work
   *
   * @param out_work_request_key [out] Work request identifier of completed work
   * @param out_status [out] Status
   * @param response_buffers [out] Response buffers
   *
   * @return True if bytes received
   */
  virtual bool check_work_completions(uint64_t&                                         out_work_request_key,
                                      status_t&                                         out_status,
                                      component::IADO_plugin::response_buffer_vector_t& response_buffers) = 0;

  /**
   * Get callback buffer withouth interpreting
   *
   * @param out_buffer Recieved buffer
   *
   * @return S_OK or E_EMPTY
   */
  virtual status_t recv_callback_buffer(Buffer_header*& out_buffer) = 0;

  /**
   * Free callback buffer
   *
   * @param buffer Recieved buffer
   *
   */
  virtual void free_callback_buffer(void* buffer) = 0;

  /**
   * Check for table operations (e.g., create_key)
   *
   * @param buffer Message buffer
   * @param work_request_id [out] Work request identifier of completed work
   * @param op [out] Operation type (see ado_proto.fbs)
   * @param key [out] Name of corresponding key
   * @param value_len [out] Size in bytes (optional)
   *
   * @return True if message interpreted as table op
   */
  virtual bool check_table_ops(const void*        buffer,
                               uint64_t&          work_request_id,
                               component::ADO_op& op,
                               std::string&       key,
                               size_t&            value_len,
                               size_t&            value_alignment,
                               void*&             addr) = 0;

  /**
   * Check for index operations (e.g., find)
   *
   * @param buffer Message buffer
   * @param key_expression [out] Search expression
   * @param begin_pos [out] Position of match if successfull
   * @param find_type Type of search
   * @param max_comp Maximum numbfer of comparisons to make
   *
   * @return True if message interpreted as index op
   */
  virtual bool check_index_ops(const void*  buffer,
                               std::string& key_expression,
                               offset_t&    begin_pos,
                               int&         find_type,
                               uint32_t     max_comp) = 0;
  /**
   * Check for vector operations
   *
   * @param buffer Message buffer
   * @param t_begin Begin time constraint Zero for none.
   * @param t_end End time constraint. Zero for none.
   *
   * @return True if message interpreted as vector op
   */
  virtual bool check_vector_ops(const void* buffer, common::epoch_time_t& t_begin, common::epoch_time_t& t_end) = 0;

  /**
   * Check for vector operations
   *
   * @param buffer Message buffer
   *
   * @return True if message interpreted as pool info op
   */
  virtual bool check_pool_info_op(const void* buffer) = 0;

  /**
   * Check for iteration
   *
   * @param buffer Message buffer
   * @param t_begin Begin time constraint Zero for none.
   * @param t_end End time constraint. Zero for none.
   * @param iterator Iteration handle
   *
   * @return True if message interpreted as pool info op
   */
  virtual bool check_iterate(const void*                           buffer,
                             common::epoch_time_t&                 t_begin,
                             common::epoch_time_t&                 t_end,
                             component::IKVStore::pool_iterator_t& iterator) = 0;

  /**
   * Check for op event responses
   *
   * @param buffer Message buffer
   * @param op Operation
   *
   * @return True if message interpreted as op event response
   */
  virtual bool check_op_event_response(const void* buffer, component::ADO_op& op) = 0;

  /**
   * Check for unlock request
   *
   * @param buffer Message buffer
   * @param work_id Work identifier
   * @param key_handle Key handle for unlock
   *
   * @return True if message interpreted as unlock request
   */
  virtual bool check_unlock_request(const void* buffer, uint64_t& work_id, component::IKVStore::key_t& key_handle) = 0;

  /**
   * Send response to unlock operation
   *
   * @param status Status code
   *
   */
  virtual void send_unlock_response(const status_t status) = 0;


  /**
   * @brief      Check for configuration request
   *
   * @param[in]  buffer   Message buffer
   * @param      options  Configuration options
   */
  virtual bool check_configure_request(const void* buffer, uint64_t& options) = 0;

  /**
   * @brief      Send configure response
   *
   * @param[in]  status Status code
   */
  virtual void send_configure_response(const status_t status) =  0;

  /**
   * Send response to ADO for table operation
   *
   * @param s Status code
   * @param value_addr [optional] Address of new
   * @param value_len [optional] Length of value in bytes
   * @param key_ptr [out] Reference to key
   * @param key_handle [out] Key handle for unlocking explicitly
   *
   */
  virtual void send_table_op_response(const status_t             status,
                                      const void*                value_addr     = nullptr,
                                      size_t                     value_len      = 0,
                                      const char*                key_ptr        = nullptr,
                                      component::IKVStore::key_t out_key_handle = nullptr) = 0;

  /**
   * Send response to ADO for index operation
   *
   * @param status Status code
   * @param matched_position Offset of the match
   * @param matched_key Key of the match
   *
   */
  virtual void send_find_index_response(const status_t     status,
                                        const offset_t     matched_position,
                                        const std::string& matched_key) = 0;

  /**
   * Send a vector of key-value pointers
   *
   * @param status Status code
   * @param rv Reference vector
   *
   */
  virtual void send_vector_response(const status_t status, const IADO_plugin::Reference_vector& rv) = 0;

  /**
   * Send iteration response
   *
   * @param status Status code
   * @param iterator Iterator handle (updated)
   * @param reference Reference result
   */
  virtual void send_iterate_response(const status_t                              status,
                                     const component::IKVStore::pool_iterator_t  iterator,
                                     const component::IKVStore::pool_reference_t reference) = 0;

  /**
   * Send a pool info response
   *
   */
  virtual void send_pool_info_response(const status_t status, const std::string& info) = 0;

  /**
   * Indicate whether ADO has shutdown
   *
   *
   * @return
   */
  virtual bool has_exited() = 0;

  /**
   * Request graceful shutdown
   *
   *
   * @return S_OK on success
   */
  virtual status_t shutdown() = 0;

  /**
   * Get pool id proxy is associated with
   *
   *
   * @return Pool id
   */
  virtual component::IKVStore::pool_t pool_id() const = 0;

  /**
   * Get ado id proxy is associated with
   *
   *
   * @return ado id
   */
  virtual std::string ado_id() const = 0;

  /**
   * Get the name of the pool
   *
   * @return pool name
   */
  virtual const std::string& pool_name() const = 0;

  /**
   * Add a key-value pair for deferred unlock
   *
   * @param work_request_id Work request identifier
   * @param key Key handle
   */
  virtual void add_deferred_unlock(const uint64_t work_request_id, const component::IKVStore::key_t key) = 0;

  /**
   * Updates a key-value pair deferred unlock
   *
   * @param work_request_id Work request identifier
   * @param key Key/lock handle
   *
   * @return S_OK or E_NOT_FOUND
   */
  virtual status_t update_deferred_unlock(const uint64_t work_request_id, const component::IKVStore::key_t key) = 0;

  /**
   * Retrive (and clear) keys that need to be unlock on associated pool
   *
   * @param work_request_id Work request identifier
   * @param keys Out vector of keys
   */
  virtual void get_deferred_unlocks(const uint64_t work_request_id, std::vector<component::IKVStore::key_t>& keys) = 0;

  /**
   * Add a key-value pair for unlock after the life of the ADO
   *
   * @param key Key handle
   */
  virtual void add_life_unlock(const component::IKVStore::key_t key) = 0;

  /**
   * Check if key handle already in deferred list
   *
   * @param work_request_id Work request
   * @param key Key handle
   *
   * @return True if exists as deferred lock
   */
  virtual bool check_for_implicit_unlock(const uint64_t work_key, const component::IKVStore::key_t key) = 0;

  /**
   * Remove life lock
   *
   * @param key Key handle
   *
   * @return S_OK or E_NOT_FOUND
   */
  virtual status_t remove_life_unlock(const component::IKVStore::key_t key) = 0;

  /**
   * Release life locks
   *
   */
  virtual void release_life_locks() = 0;
};

/**
 * ADO manager interface.  This is actually a proxy interface communicating with
 * an external process. The ADO manager has a "global" view of the system and
 * can coordinate / schedule resources that are being consumed by the ADO
 * processes.
 */
class IADO_manager_proxy : public component::IBase {
 public:
  // clang-format off
  DECLARE_INTERFACE_UUID(0xaaafa389,0x1665,0x4e5b,0xa1b1,0x3c,0xff,0x4a,0x5e,0xe2,0x63);
  // clang-format on

  using shared_memory_token_t = uint64_t; /*< token identifying shared memory for mcas module */

  /**
   * Launch ADO process.  This method must NOT block.
   *
   * @param auth_id Authentication identifier
   * @param debug_level Debug level
   * @param itf KV-store interface
   * @param pool_id Pool identifier
   * @param filename Location of the executable
   * @param args Command line arguments to pass
   * @param shm_token Token to pass to ADO to use to map value memory into
   * process space.
   * @param value_memory_numa_zone NUMA zone to which the value memory resides
   * @param sla Placeholder for some type of SLA/QoS requirements specification.
   *
   * @return Proxy interface, with reference count 1. Use release_ref() to
   * destroy.
   */
  virtual IADO_proxy* create(const uint64_t              auth_id,
			     const unsigned              debug_level,
                             component::IKVStore*        itf,
                             component::IKVStore::pool_t pool_id,
                             const std::string&          pool_name,
                             const size_t                pool_size,
                             const unsigned int          pool_flags,
                             const uint64_t              expected_obj_count,
                             const std::string&          filename,
                             std::vector<std::string>&   args,
                             numa_node_t                 value_memory_numa_zone,
                             SLA*                        sla = nullptr) = 0;

  /**
   * Wait for process to exit.
   *
   * @param ado_proxy Handle to proxy object
   *
   * @return S_OK on success or E_BUSY.
   */
  virtual bool has_exited(IADO_proxy* ado_proxy) = 0;

  /**
   * Shutdown ADO process
   *
   * @param ado Interface pointer to proxy
   *
   * @return S_OK on success
   */
  virtual status_t shutdown_ado(IADO_proxy* ado) = 0;
};

class IADO_manager_proxy_factory : public component::IBase {
 public:
  // clang-format off
  DECLARE_INTERFACE_UUID(0xfacfa389,0x1665,0x4e5b,0xa1b1,0x3c,0xff,0x4a,0x5e,0xe2,0x63);
  // clang-format on

  virtual IADO_manager_proxy* create(unsigned debug_level, unsigned shard, std::string cores, float cpu_num) = 0;
};

class IADO_proxy_factory : public component::IBase {
 public:
  // clang-format off
  DECLARE_INTERFACE_UUID(0xfacbb389,0x1665,0x4e5b,0xa1b1,0x3c,0xff,0x4a,0x5e,0xe2,0x63);
  // clang-format on

  virtual IADO_proxy* create(const uint64_t              auth_id,
			     const unsigned              debug_level,
                             component::IKVStore*        itf,
                             component::IKVStore::pool_t pool_id,
                             const std::string&          pool_name,
                             const size_t                pool_size,
                             const unsigned int          pool_flags,
                             const uint64_t              expected_obj_count,
                             const std::string&          filename,
                             std::vector<std::string>&   args,
                             std::string                 cores,
                             int                         memory,
                             float                       cpu_num,
                             numa_node_t                 numa_zone) = 0;
};

#pragma GCC diagnostic pop

}  // namespace component

#endif  // __API_ADO_ITF_H__
