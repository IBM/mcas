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

#ifndef __MCAS_SHARD_H__
#define __MCAS_SHARD_H__

#ifndef __cplusplus
#error Cpp file
#endif

#include <api/ado_itf.h>
#include <api/components.h>
#include <api/fabric_itf.h>
#include <api/kvindex_itf.h>
#include <api/kvstore_itf.h>
#include <common/byte_span.h>
#include <common/cpu.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/spsc_bounded_queue.h>
#include <common/string_view.h>
#include <common/perf/tm_fwd.h>

#include <csignal> /* sig_atomic_t */
#include <list>
#include <memory>
#include <string>
#include <map>
#include <set>
#include <thread>
#include <unordered_map>
#include <future>

#include "ado_map.h"
#include "cluster_messages.h"
#include "config_file.h"
#include "connection_handler.h"
#include "fabric_transport.h"
#include "mcas_config.h"
#include "pool_manager.h"
#include "range.h"
#include "security.h"
#include "task_key_find.h"
#include "types.h"

#include <nupm/mcas_mod.h>
#include <xpmem.h>

namespace common
{
struct profiler;
}

namespace signals
{
  extern volatile sig_atomic_t sigint;
}

namespace mcas
{
class Connection_handler;

/* Adapter point */
using Shard_transport = Fabric_transport;

class Shard : public Shard_transport, private common::log_source {
 private:
  static constexpr size_t TWO_STAGE_THRESHOLD = KiB(128); /* above this two stage protocol is used */
  static constexpr const char *const _cname = "Shard";
  static constexpr const char *const flush_enable_key = "FLUSH_ENABLE";

  using byte_span = common::byte_span;

 private:
  struct rename_info_t {
    rename_info_t(const component::IKVStore::pool_t pool_, const std::string &from_, const std::string &to_)
        : pool(pool_),
          from(from_),
          to(to_)
    {
    }

    component::IKVStore::pool_t pool;
    std::string                 from;
    std::string                 to;
  };

  struct space_lock_info_t {
    using transport_t = mcas::Connection_base;
    memory_registered<transport_t> mr;
    unsigned                       count;

    explicit space_lock_info_t(memory_registered<transport_t> &&mr_) noexcept : mr(std::move(mr_)), count(0) {}
  };

  struct lock_info_t : public space_lock_info_t {
    component::IKVStore::pool_t pool;
    component::IKVStore::key_t  key;
    size_t                      value_size;
    lock_info_t(component::IKVStore::pool_t      pool_,
                component::IKVStore::key_t       key_,
                std::size_t                      value_size_,
                memory_registered<transport_t> &&mr_) noexcept
        : space_lock_info_t(std::move(mr_)),
          pool(pool_),
          key(key_),
          value_size(value_size_)
    {
    }
    lock_info_t(const lock_info_t &) = delete;
    lock_info_t &operator=(const lock_info_t &) = delete;
  };

  struct pool_desc_t {
    std::string  name;
    size_t       size;
    unsigned int flags;
    size_t       expected_obj_count;
    bool         opened_existing;
    void*        base_addr;
  };

  using pool_t              = component::IKVStore::pool_t;
  using buffer_t            = Shard_transport::buffer_t;
  using index_map_t         = std::unordered_map<pool_t, component::Itf_ref<component::IKVIndex>>;
  using locked_value_map_t  = std::unordered_map<const void* , lock_info_t>;
  using spaces_shared_map_t = std::map<range<std::uint64_t>, space_lock_info_t>;
  using rename_map_t        = std::unordered_map<const void* , rename_info_t>;
  using task_list_t         = std::list<Shard_task* >;

 public:
  using string_view = common::string_view;

  Shard(const Config_file & config_file,
        unsigned            shard_index,
        const std::string & dax_config,
        unsigned            debug_level,
        bool                forced_exit,
        const char *        profile_file,
        bool                triggered_profile);

  Shard(const Shard &) = delete;
  Shard &operator=(const Shard &) = delete;

  ~Shard() {}

  inline void get_future() { return _thread.get(); }
  inline bool exiting() const { return _thread_exit; }

  inline void signal_exit() /*< signal main loop to exit */
  {
    _thread_exit = true;
  }

  inline void send_cluster_event(const std::string &sender, const std::string &type, const std::string &content)
  {
    _cluster_signal_queue.send_message(sender, type, content);
  }

 private:
  void thread_entry(const std::string &backend,
                    const std::string &index,
                    const std::string &dax_config,
                    unsigned           debug_level,
                    const std::string  ado_cores,
                    float              ado_core_number,
                    const char *       profile_main_loop,
                    bool               triggered_profile);

  /* locked values are those from put_direct and get_direct */
  void add_locked_value_shared(const pool_t                         pool_id,
                               component::IKVStore::key_t           key,
                               void *                               target,
                               size_t                               target_len,
                               memory_registered<Connection_base> &&mr);
  void release_locked_value_shared(const void *target);
  void add_locked_value_exclusive(const pool_t                         pool_id,
                                  component::IKVStore::key_t           key,
                                  void *                               target,
                                  size_t                               target_len,
                                  memory_registered<Connection_base> &&mr);
  void release_locked_value_exclusive(const void *target);

  void add_space_shared(const range<std::uint64_t> &range, memory_registered<Connection_base> &&mr);
  void release_space_shared(const range<std::uint64_t> &range);

  void add_pending_rename(const pool_t pool_id, const void *target, const std::string &from, const std::string &to);
  void release_pending_rename(const void *target);

  inline void add_target_keyname(const void *target, const std::string& skey) {
    _target_keyname_map[target] = skey;
  }

  inline std::string release_target_keyname(const void *target) {
    auto r = _target_keyname_map[target];
    _target_keyname_map.erase(target);
    return r;
  }


  void initialize_components(const std::string &backend,
                             const std::string &mm_plugin,
                             const std::string &dax_config,
                             unsigned           debug_level,
                             const std::string  ado_cores,
                             float              ado_core_number);

  void check_for_new_connections();

  void main_loop(common::profiler &);

  /* message processing functions */
  void process_message_pool_request(Connection_handler *handler, const protocol::Message_pool_request *msg);
  void process_message_IO_request(Connection_handler *handler, const protocol::Message_IO_request *msg);
  void process_info_request(Connection_handler *handler, const protocol::Message_INFO_request *msg, common::profiler &pr);
  void process_ado_request(Connection_handler *handler, const protocol::Message_ado_request *msg);
  void process_put_ado_request(Connection_handler *handler, const protocol::Message_put_ado_request *msg);
  void process_messages_from_ado();
  status_t process_configure(const protocol::Message_IO_request *msg);

  /* response handling functions */
  void io_response_put_advance(Connection_handler *handler, const protocol::Message_IO_request *msg, buffer_t *iob);
  void io_response_put_locate(Connection_handler *handler, const protocol::Message_IO_request *msg, buffer_t *iob);
  void io_response_put_release(Connection_handler *handler, const protocol::Message_IO_request *msg, buffer_t *iob);
  void io_response_get_locate(Connection_handler *handler, const protocol::Message_IO_request *msg, buffer_t *iob);
  void io_response_get_release(Connection_handler *handler, const protocol::Message_IO_request *msg, buffer_t *iob);
  void io_response_put(Connection_handler *handler, const protocol::Message_IO_request *msg, buffer_t *iob);
  void io_response_get(Connection_handler *handler, const protocol::Message_IO_request *msg, buffer_t *iob);
  void io_response_erase(Connection_handler *handler, const protocol::Message_IO_request *msg, buffer_t *iob);
  void io_response_configure(Connection_handler *handler, const protocol::Message_IO_request *msg, buffer_t *iob);
  void io_response_locate(Connection_handler *handler, const protocol::Message_IO_request *msg, buffer_t *iob);
  void io_response_release(Connection_handler *handler, const protocol::Message_IO_request *msg, buffer_t *iob);
  void io_response_release_with_flush(Connection_handler *handler, const protocol::Message_IO_request *msg, buffer_t *iob);

  void close_all_ado();

  void process_tasks(unsigned &idle);

  void service_cluster_signals();

  void signal_ado(const char * tag,
                  Connection_handler * handler,
                  const uint64_t client_request_id,
                  const component::IKVStore::pool_t pool,
                  const std::string& key,
                  const component::IKVStore::lock_type_t lock_type,
                  const bool is_get_response = false);

  void signal_ado_async_nolock(const char * tag,
                               Connection_handler * handler,
                               const uint64_t client_request_id,
                               const component::IKVStore::pool_t pool,
                               const std::string& key);

  static protocol::Message_IO_response * prepare_response(const Connection_handler *           handler_,
                                                          buffer_t *                           iob_,
                                                          uint64_t                             request_id,
                                                          int                                  status_);

  static void respond(Connection_handler *                 handler,
                      buffer_t *                           iob,
                      const protocol::Message_IO_request * msg,
                      int                                  status,
                      const char *                         func);

  component::IKVIndex *lookup_index(const pool_t pool_id)
  {
    if (_index_map) {
      auto search = _index_map->find(pool_id);
      if (search == _index_map->end()) return nullptr;
      return search->second.get();
    }
    else
      return nullptr;
  }

  void add_index_key(const pool_t pool_id, const std::string &k)
  {
    auto index = lookup_index(pool_id);
    if (index) index->insert(k);
  }

  void remove_index_key(const pool_t pool_id, const std::string &k)
  {
    auto index = lookup_index(pool_id);
    if (index) index->erase(k);
  }

  inline void add_task_list(Shard_task *task) { _tasks.push_back(task); }

  inline size_t session_count() const { return _handlers.size(); }

  struct sg_result {
    std::vector<Protocol::Message_IO_response::locate_element> sg_list;
    std::uint64_t mr_low;
    std::uint64_t mr_high;
    std::uint64_t excess_length;
  };

  auto offset_to_sg_list(
    range<std::uint64_t> t
    , const std::vector<byte_span> &region_breaks
  ) -> sg_result;

  inline bool ado_enabled() const { return (_i_ado_mgr && (_ado_plugins.size() > 0)); }

  /* ADO signalling helpers */
  inline bool ado_signal_enabled() const { return ado_enabled() && (_ado_signal_mask != Ado_signal::NONE); }
  inline bool ado_signal_post_put() const { return ado_enabled() && (_ado_signal_mask & Ado_signal::POST_PUT); }
  inline bool ado_signal_post_get() const { return ado_enabled() && (_ado_signal_mask & Ado_signal::POST_GET); }
  inline bool ado_signal_post_erase() const { return ado_enabled() && (_ado_signal_mask & Ado_signal::POST_ERASE); }

  inline auto get_ado_interface(pool_t pool_id) { return _ado_pool_map.get_proxy(pool_id); }

  status_t conditional_bootstrap_ado_process(component::IKVStore *       kvs,
                                             Connection_handler *        handler,
                                             component::IKVStore::pool_t pool_id,
                                             component::IADO_proxy *&    ado,
                                             pool_desc_t &               desc);

  /* per-shard statistics */
  component::IMCAS::Shard_stats _stats alignas(8);

  void dump_stats()
  {
    PINF("------------------------------------------------");
    PINF("| Shard Statistics                             |");
    PINF("------------------------------------------------");
    PINF("PUT count          : %lu", _stats.op_put_count);
    PINF("GET count          : %lu", _stats.op_get_count);
    PINF("PUT_DIRECT count   : %lu", _stats.op_put_direct_count);
    PINF("GET_DIRECT count   : %lu", _stats.op_get_direct_count);
    PINF("GET 2-stage count  : %lu", _stats.op_get_twostage_count);
    PINF("ERASE count        : %lu", _stats.op_erase_count);
    PINF("ADO count          : %lu (enabled=%s)", _stats.op_ado_count, ado_enabled() ? "yes" : "no");
    PINF("Failed count       : %lu", _stats.op_failed_request_count);
    PINF("Session count      : %lu", session_count());
    PINF("------------------------------------------------");
  }

 private:
  struct work_request_t {
    Connection_handler *             handler;
    component::IKVStore::pool_t      pool;
    component::IKVStore::key_t       key_handle;
    const char *                     key_ptr;
    size_t                           key_len;
    component::IKVStore::lock_type_t lock_type;
    uint64_t                         request_id; /* original client request */
    uint32_t                         flags;

    inline bool is_async() const { return flags & component::IMCAS::ADO_FLAG_ASYNC; }
  };

  class Work_request_allocator {
   private:
    static constexpr size_t NUM_ELEMENTS = WORK_REQUEST_ALLOCATOR_COUNT;
    std::vector<work_request_t *> _free;
    /* "_all" apparently exists only to own the work_request_t elements */
    std::vector<std::unique_ptr<work_request_t>> _all;

   public:
    Work_request_allocator() : _free{}, _all{}
    {
      for (size_t i = 0; i < NUM_ELEMENTS; i++) {
        auto wr =
            std::unique_ptr<work_request_t>(static_cast<work_request_t *>(aligned_alloc(64, sizeof(work_request_t))));
        if (!wr) {
          throw std::bad_alloc();
        }
        _free.push_back(wr.get());
        _all.push_back(std::move(wr));
      }
    }

    virtual ~Work_request_allocator() {}

    inline work_request_t *allocate()
    {
      if (_free.size() == 0) throw General_exception("Work_request_allocator exhausted");
      auto wr = _free.back();
      _free.pop_back();
      return wr;
    }

    inline void free_wr(work_request_t *wr) { _free.push_back(wr); }

  } _wr_allocator;

  using ado_pool_map_t =
      std::unordered_map<component::IKVStore::pool_t,
                         std::pair<component::IADO_proxy *, Connection_handler *>>;

  using work_request_key_t = uint64_t;

  static inline work_request_t *request_key_to_record(work_request_key_t key)
  {
    return reinterpret_cast<work_request_t *>(key);
  }

  inline const std::string &net_addr() const { return _net_addr; }

  /* Shard class members */
  const std::string                                 _net_addr;
  const unsigned int                                _port;
  std::unique_ptr<index_map_t>                      _index_map; /* depends on _i_kvstore therefore should be cleaned up first */
  bool                                              _thread_exit;
  bool                                              _forced_exit;
  unsigned                                          _core;
  size_t                                            _max_message_size;
  component::Itf_ref<component::IKVStore>           _i_kvstore;
  component::Itf_ref<component::IADO_manager_proxy> _i_ado_mgr;    /*< null indicate non-ADO mode */
  Ado_pool_map                                      _ado_pool_map; /*< maps open pool handles to ADO proxy */
  Ado_map                                           _ado_map;      /*< managing the pool name to ADO proxy */
  std::vector<Connection_handler *>                 _handlers;
  locked_value_map_t                                _locked_values_shared;
  locked_value_map_t                                _locked_values_exclusive;
  std::map<const void*, std::string>                _target_keyname_map;
  spaces_shared_map_t                               _spaces_shared;
  rename_map_t                                      _pending_renames;
  task_list_t                                       _tasks; /*< list of deferred tasks */
  std::set<work_request_key_t>                      _outstanding_work;
  std::vector<work_request_t *>                     _failed_async_requests;
  const std::string                                 _ado_path;
  std::vector<std::string>                          _ado_plugins;
  std::map<std::string, std::string>                _ado_params;
  Ado_signal                                        _ado_signal_mask = Ado_signal::NONE;  /* active signals for shard */
  Shard_security                                    _security; /* manages TLS authentication etc. */
  Cluster_signal_queue                              _cluster_signal_queue;
  std::string                                       _backend;
  std::string                                       _mm_plugin_path;
  std::string                                       _dax_config;
  std::future<void>                                 _thread;
};

}  // namespace mcas

#endif  // __MCAS_SHARD_H__
