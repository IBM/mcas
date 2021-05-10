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

#include "shard.h"

#include <api/components.h>
#include <api/kvindex_itf.h>
#include <common/dump_utils.h>
#include <common/env.h>
#include <common/profiler.h>
#include <common/utils.h>
#include <common/str_utils.h>
#include <sys/types.h>
#include <signal.h>
#include <pwd.h>
#include <libpmem.h>
#include <nupm/mcas_mod.h>
#include <nupm/region_descriptor.h>
#include <zlib.h>

#include <boost/numeric/conversion/cast.hpp>

#include "resource_unavailable.h"
#include "env.h"

#include <sys/types.h> /* getpid */
#include <unistd.h>

#include <algorithm> /* remove */
#include <cinttypes>
#include <cstdlib> /* system */

#include <limits>
#include <sstream>

volatile sig_atomic_t signals::sigint = 0;

using namespace mcas;
using namespace component;

static inline bool check_mcas_module()
{
  int fd = open("/dev/mcas", O_RDWR, 0666);
  close(fd);
  return (fd != -1);
}

namespace
{
class Env {
public:
  static std::string get_user_name()
  {
    uid_t          uid = geteuid();
    struct passwd *pw  = getpwuid(uid);
    if (pw) return std::string(pw->pw_name);
    return {};
  }
};

/* Unlock a key, if necessary: If is_locked(store->lock())
 * unlock when the object goes out of scope *unless*
 * responsibility has been released (by a call to release)
 */
struct locked_key
{
  common::moveable_ptr<component::IKVStore> _store;
  IKVStore::pool_t                          _pool;
  component::IKVStore::key_t                _lock_handle;

  locked_key(component::IKVStore *store_, IKVStore::pool_t pool_, component::IKVStore::key_t lh_)
    : _store(store_)
    , _pool(pool_)
    , _lock_handle(lh_)
  {}
  locked_key(const locked_key &) = delete;
  locked_key &operator=(const locked_key &) = delete;
  ~locked_key()
  {
    if ( _store ) {
      _store->unlock(_pool, _lock_handle);
    }
  }
  component::IKVStore::key_t release()
  {
    _store.release();
    return _lock_handle;
  }
};
}  // namespace

namespace mcas
{

/* global parameters */
namespace global
{
unsigned debug_level = 0;
}

static std::string default_mm_plugin(const Config_file& /*config_file*/,
                                     const std::string& backend)
{
  if(backend == "mapstore") return DEFAULT_MAPSTORE_MM_PLUGIN_PATH;
  else if(backend == "hstore") return DEFAULT_HSTORE_MM_PLUGIN_PATH;
  else if(backend == "hstore-cc") return DEFAULT_HSTORE_CC_MM_PLUGIN_PATH;
  else throw Logic_exception("invalid store");
}

Shard::Shard(const Config_file &config_file,
             const unsigned     shard_index,
             const std::string &dax_config,
             const unsigned     debug_level_,
             const bool         forced_exit,
             const char *const  profile_file_,
             const bool         triggered_profile_)
  : Shard_transport(
                    /* libfabric calls this "info::src_addr" and "info::src_addrlen" */
                    config_file.get_shard_optional(config::addr, shard_index),
                    /* libfabric calls this "info::fabric::prov_name" */
                    config_file.get_net_providers(),
                    /* libfabric calls this "info::domain::name", and also (as a separate
                       parameter) "node" */
                    config_file.get_shard_optional(config::net, shard_index),
                    config_file.get_shard_port(shard_index)),
    common::log_source(debug_level_),
    _stats{},
    _wr_allocator{},
    _net_addr(config_file.get_shard_optional(config::addr, shard_index)
              ? *config_file.get_shard_optional(config::addr, shard_index)
              : ""),
    _port(config_file.get_shard_port(shard_index)),
    _index_map(nullptr),
    _thread_exit(false),
    _forced_exit(forced_exit),
    _core(config_file.get_shard_core(shard_index)),
    _max_message_size(0),
    _i_kvstore(nullptr),
    _i_ado_mgr(nullptr),
    _ado_pool_map(debug_level_),
    _ado_map(),
    _handlers{},
    _locked_values_shared{},
    _locked_values_exclusive{},
    _target_keyname_map{},
    _spaces_shared{},
    _pending_renames{},
    _tasks{},
    _outstanding_work{},
    _failed_async_requests{},
    _ado_path(config_file.get_ado_path() ? *config_file.get_ado_path() : ""),
    _ado_plugins(config_file.get_shard_ado_plugins(shard_index)),
    _ado_params(config_file.get_shard_ado_params(shard_index)),
    _ado_signal_mask(config_file.get_shard_ado_signals(shard_index)),
    _security(config_file.security_get_cert_path(),
              config_file.security_get_key_path(),
              config_file.get_shard_optional(config::security_mode, shard_index),
              config_file.get_shard_optional(config::addr, shard_index),
              config_file.get_shard_optional(config::net, shard_index),
              config_file.get_shard_security_port(shard_index),
              debug_level_),
    _cluster_signal_queue(),
    _backend(config_file.get_shard_required(config::default_backend, shard_index)),
    _mm_plugin_path(default_mm_plugin(config_file, _backend)),
    _dax_config(dax_config),
    _thread(std::async(std::launch::async,
                       &Shard::thread_entry,
                       this,
                       _backend,
                       _mm_plugin_path,
                       dax_config,
                       debug_level(),
                       config_file.get_shard_ado_cores(shard_index),
                       config_file.get_shard_ado_core_number(shard_index),
                       profile_file_,
                       triggered_profile_))
{
}

void Shard::thread_entry(const std::string& backend,
                         const std::string& mm_plugin_path,
                         const std::string& dax_config,
                         const unsigned     debug_level,
                         const std::string  ado_cores,
                         const float        ado_core_num,
                         const char *const  profile_main_loop_,
                         const bool         triggered_profile_)
{
  (void) _max_message_size;  // unused
  CPLOG(2, "Shard: %u worker thread entered.", _core);

  /* pin thread */
  cpu_mask_t mask;
  mask.add_core(_core);
  if (set_cpu_affinity_mask(mask) == -1) {
    PLOG("%s: bad mask parameter", __FILE__);
  }

  CPLOG(2, "Shard: CPU_MASK thread %p configured with cpu mask: [%s]",
        common::p_fmt(this), mask.string_form().c_str());

  try {
    try {
      initialize_components(backend,
                            mm_plugin_path,
                            dax_config, debug_level, ado_cores, ado_core_num);
    }
    catch (const General_exception &e) {
      PERR("Shard component initialization failed: %s.", e.cause());
      throw;
    }
    catch (const std::exception &e) {
      PERR("Shard component initialization failed: %s.", e.what());
      throw;
    }

    common::profiler p(profile_main_loop_, !triggered_profile_);
    main_loop(p);
  }
  catch (const General_exception &e) {
    PERR("Shard component execution failed: %s.", e.cause());
  }
  catch (const std::exception &e) {
    PERR("Shard component execution failed: %s.", e.what());
  }

  /* main_loop sets _thread_exit true, but it will not be called on early failure */
  _thread_exit = true;

  CPLOG(2, "Shard: %u worker thread exited.", _core);
}

void Shard::initialize_components(const std::string &backend,
                                  const std::string &mm_plugin_path,
                                  const std::string &dax_config,
                                  unsigned debug_level,
                                  const std::string ado_cores,
                                  float ado_core_num)
{
  using namespace component;

  /* STORE */
  {
    IBase *comp;

    if (backend == "mapstore")
      comp = load_component("libcomponent-mapstore.so", mapstore_factory);
    else if (backend == "hstore")
      comp = load_component("libcomponent-hstore.so", hstore_factory);
    else if (backend == "hstore-cc")
      comp = load_component("libcomponent-hstore-cc.so", hstore_factory);
    else
      throw General_exception("unrecognized backend (%s)", backend.c_str());

    CPINF(1, "Shard: using store backend (%s)", backend.c_str());

    if (!comp) throw General_exception("unable to initialize mcas backend component");

    auto fact = make_itf_ref(static_cast<IKVStore_factory *>(comp->query_interface(IKVStore_factory::iid())));
    assert(fact);

    if (backend == "hstore" || backend == "hstore-cc") {
      if (dax_config.empty()) throw General_exception("hstore backend requires dax configuration");

      _i_kvstore.reset(fact->create(debug_level,
                                    {
                                     {+component::IKVStore_factory::k_debug, std::to_string(debug_level)},
                                     {+component::IKVStore_factory::k_dax_config, dax_config}
                                    }
                                    ));
    }
    else { /* mapstore */
      _i_kvstore.reset(fact->create(debug_level,
                                    {
                                     {+component::IKVStore_factory::k_mm_plugin_path, mm_plugin_path}
                                    }));
    }
  }

  /* ADO manager proxy */
  {
    IBase *comp = load_component("libcomponent-adomgrproxy.so", ado_manager_proxy_factory);
    if (comp) {
      auto fact = make_itf_ref(static_cast<IADO_manager_proxy_factory *>
                               (comp->query_interface(IADO_manager_proxy_factory::iid())));
      assert(fact);

      _i_ado_mgr.reset(fact->create(debug_level, _core, ado_cores, ado_core_num));

      if (_i_ado_mgr == nullptr)
        throw General_exception("Instantiation of ADO manager failed unexpectedly.");

      PMAJOR("ADO manager created.");
    }
    else {
      PMAJOR("ADO not found and thus not enabled.");
    }
  }
}

void Shard::service_cluster_signals()
{
  return;

  Cluster_message *cmsg;

  while (_cluster_signal_queue.recv_message(cmsg)) {

    CPLOG(2,"Shard: (%p) got message !! (sender=%s,type=%s,content=%s)",
          common::p_fmt(this),
          cmsg->_sender.c_str(), cmsg->_type.c_str(), cmsg->_content.c_str());

    assert(cmsg);

    /* if we have ADOs then forward to them */
    for (auto &proxy : _ado_map) {
      CPLOG(2, "Shard: sending cluster event to ADO (%p)...", common::p_fmt(proxy.second));

      if (proxy.second->send_cluster_event(cmsg->_sender, cmsg->_type, cmsg->_content) != S_OK)
        throw General_exception("send_cluster_event to ADO failed");
    }

    delete cmsg;
  }
}


//#define DEBUG_LIVENESS // use this to show how live the shard loop threads are
#define LIVENESS_DURATION 10000
#define LIVENESS_SHARDS 18

void Shard::main_loop(common::profiler &pr_)
{
  using namespace mcas::protocol;

  std::ostringstream ss;
  ss << "shard-" << _core;
  pthread_setname_np(pthread_self(), ss.str().c_str());
  assert(_i_kvstore);

  static constexpr uint64_t CHECK_CONNECTION_INTERVAL     = 1000;
  static constexpr uint64_t CHECK_CLUSTER_SIGNAL_INTERVAL = 10000;
  static constexpr uint64_t OUTPUT_DEBUG_INTERVAL         = 10000000;
  static constexpr unsigned SESSIONS_EMPTY_USLEEP         = 50000;

  Connection_handler::action_t action;

#ifdef DEBUG_LIVENESS
  static std::mutex score_board_lock;
  static uint64_t score_board[LIVENESS_SHARDS] = {0};
#endif

  unsigned idle            = 0;
  uint64_t tick alignas(8) = 0;

  for (; _thread_exit == false; ++idle, ++tick) {
#ifdef DEBUG_LIVENESS
    assert(_core < LIVENESS_SHARDS);
    /* show report of liveness */
    if(tick % LIVENESS_DURATION == 0) {
      std::lock_guard<std::mutex> g(score_board_lock);
      score_board[_core] = tick;

      std::stringstream sstr;
      if(_core == 0 || _core == 8) {
        unsigned i;
        for(i=0;i<LIVENESS_SHARDS;i++) {
          sstr << " " << score_board[i] / LIVENESS_DURATION;
        }
        sstr << " :" << i;
        PMAJOR("Live: [%s ]", sstr.str().c_str());
      }
    }
#endif

    /* graceful exit on sigint */
    if(signals::sigint > 0) {
      CPLOG(2, "Shard: received SIGINT");
      _thread_exit = true;
    }
    else if (_handlers.empty()) { /* if there are no sessions, sleep thread */
      usleep(SESSIONS_EMPTY_USLEEP);
      try {
        check_for_new_connections();
      }
      catch (const std::exception &e) {
        PERR("Shard: cannot get new connection: %s", e.what());
        _thread_exit = true;
      }
      service_cluster_signals();
      continue;
    }

    /* check for new connections or sleep on none */
    if (tick % CHECK_CONNECTION_INTERVAL == 0) {
      try {
        check_for_new_connections();
      }
      catch (const std::exception &e) {
        PERR("Shard: cannot get new connection: %s", e.what());
        _thread_exit = true;
      }
    }

    /* periodic cluster signal handling */
    {
      if (tick % CHECK_CLUSTER_SIGNAL_INTERVAL == 0) {
        service_cluster_signals();
      }
    }

    /* output memory usage for debug level > 0 */
    if (debug_level() > 0) {
      if (tick % OUTPUT_DEBUG_INTERVAL == 0)
        CPINF(2, "Shard_ado: port(%u) '#memory' %s", _port, common::get_DRAM_usage().c_str());
    }

    {
      std::vector<Connection_handler *> pending_close;

      _stats.client_count = boost::numeric_cast<uint16_t>(_handlers.size()); /* update stats client count */

      assert(_handlers.size() < 1000);

      /* iterate connection handlers (each connection is a client session) */
      for (const auto handler : _handlers) {

        /* issue tick, unless we are stalling */
        auto tick_response = handler->tick();

        if(tick_response == mcas::Connection_handler::TICK_RESPONSE_WAIT_SECURITY) {
          /* first tick, complete initialization */
          handler->configure_security(_security.ipaddr(),
                                      _security.port(),
                                      _security.cert_path(),
                                      _security.key_path());
        }
        /* Close session, this will occur if the client shuts down (cleanly or
         * not). Also close sessions in response to SIGINT */
        else if ((tick_response == mcas::Connection_handler::TICK_RESPONSE_CLOSE) ||
                 (signals::sigint > 0)) {
          idle = 0;

          /* close all open pools belonging to session  */
          CPLOG(1, "Shard: forcing pool closures");

          /* iterate open pool handles, close them and associated ADO processes
           */
          auto& pool_set = handler->pool_manager().open_pool_set();

          for (auto &p : pool_set) {
            auto pool_id = p.first;

            /* close ADO process on pool close */
            if (ado_enabled()) {
              {
                /* decrement reference to ADO proxy, clean up
                   when zero */
                auto ado_itf = get_ado_interface(pool_id);

                CPLOG(2, "Shard: check for ADO close ref count=%u", ado_itf->ref_count());

                if (ado_itf->ref_count() == 1) {
                  ado_itf->shutdown();
                  _ado_map.remove(ado_itf);

                  if (_i_kvstore->close_pool(pool_id) != S_OK)
                    throw Logic_exception("failed to close pool");
                }

                ado_itf->release_ref();
              }

              _ado_pool_map.release(pool_id);
            }

            CPLOG(2, "Shard: closed pool handle %lx for connection close request", pool_id);
          }

          CPLOG(1,"Shard: closing connection %p", common::p_fmt(handler));
          pending_close.push_back(handler);
        } // TICK_RESPONSE_CLOSE

        /* process ALL deferred actions */
        int get_pending_iter = 0;
        while (handler->get_pending_action(action)) {
          idle = 0;
          (void)get_pending_iter;
          assert(get_pending_iter++ < 1000);

          switch (action.op) {
#if 0
          case Connection_handler::action_type::ACTION_RELEASE_VALUE_LOCK_EXCLUSIVE:
            CPLOG(2, "releasing esxclusive value lock (%p)", action.parm);
            release_locked_value_exclusive(action.parm);
            release_pending_rename(action.parm);
            break;
#endif
          case Connection_handler::action_type::ACTION_RELEASE_VALUE_LOCK_SHARED:
            CPLOG(2, "releasing shared value lock (%p)", action.parm);
            release_locked_value_shared(action.parm);
            break;
          default:
            throw Logic_exception("unexpected action type %d", int(action.op));
          }
        }

        /* A process which cannot handle the top queue message due to
         * lack of resource may throw resource_unavailable, which will
         * leave the protocol::Message on the queue for later
         * handling.
         */
        try {

          /* collect ONE available messages ; don't collect them ALL, they just keep coming! */
          if (const protocol::Message *p_msg = handler->peek_pending_msg()) {

            idle = 0;
            assert(p_msg);
            /* "split" accepts responsibility for the *preceding* code. Not exactly intuitive. */
            switch (p_msg->type_id()) {
            case MSG_TYPE::IO_REQUEST:
              process_message_IO_request(handler, static_cast<const protocol::Message_IO_request *>(p_msg));
              break;
            case MSG_TYPE::ADO_REQUEST:
              process_ado_request(handler, static_cast<const protocol::Message_ado_request *>(p_msg));
              break;
            case MSG_TYPE::PUT_ADO_REQUEST:
              process_put_ado_request(handler, static_cast<const protocol::Message_put_ado_request *>(p_msg));
              break;
            case MSG_TYPE::POOL_REQUEST:
              process_message_pool_request(handler, static_cast<const protocol::Message_pool_request *>(p_msg));
              break;
            case MSG_TYPE::INFO_REQUEST:
              process_info_request(handler, static_cast<const protocol::Message_INFO_request *>(p_msg), pr_);
              break;
            default:
              throw General_exception("unrecognizable message type");
            }
            handler->free_buffer(handler->pop_pending_msg());
          }
        }
        catch (const resource_unavailable &e) {
          PWRN("%s: short of buffers in 'handler' processing: %s", __func__, e.what());
        }
        catch (const std::exception &e) {
          PWRN("%s: exception in 'handler' processing: %s", __func__, e.what());
          throw;
        }
      }  // iteration of handlers

      /* handle messages send back from ADO */
      try {
        if(ado_enabled())
          process_messages_from_ado();
      }
      catch (const resource_unavailable &e) {
        PWRN("short of buffers in 'ADO' processing: %s", e.what());
      }
      catch (const std::exception &e) {
        PWRN("%s: exception in 'ADO' processing: %s", __func__, e.what());
        throw;
      }

      {
        /* handle tasks */
        process_tasks(idle);
      }

      /* handle pending close sessions */
      assert(pending_close.size() < 1000);

      /* process closures */
      {
        for (auto &h : pending_close) {
          _handlers.erase(std::remove(_handlers.begin(), _handlers.end(), h), _handlers.end());
          CPLOG(2, "Shard: deleting handler (%p)", common::p_fmt(h));

          assert(h);
          delete h;

          CPLOG(2, "Shard: #remaining handlers (%lu)", _handlers.size());

          if (_handlers.empty() && _forced_exit) {
            CPLOG(1, "Shard: forcing exit..");
            _thread_exit = true;
          }
        }
      }
    }
  }

  {
    close_all_ado();
  }

  CPLOG(1, "Shard: shard (%p) exited", common::p_fmt(this));
}

void Shard::process_message_pool_request(Connection_handler *handler,
                                         const protocol::Message_pool_request *msg)
{
  handler->msg_recv_log(msg, __func__);
  using namespace component;
  // validate auth id
  assert(msg->op());

  /* allocate response buffer */
  auto response_iob = handler->allocate_send();
  assert(response_iob);
  assert(response_iob->base());
  memset(response_iob->iov->iov_base, 0, response_iob->iov->iov_len);

  Pool_manager &pool_mgr = handler->pool_manager();

  protocol::Message_pool_response *response =
    new (response_iob->base()) protocol::Message_pool_response(handler->auth_id());

  assert(response->version() == protocol::PROTOCOL_VERSION);
  response->set_status(S_OK);

  try {
    /* handle operation */
    switch (msg->op())
      {
      case mcas::protocol::OP_CREATE: {
        static unsigned count = 0;
        count++;

        CPINF(1,"POOL CREATE: op=%u name=%s size=%lu obj-count=%lu (%u) base_addr=0x%lx",
              msg->op(), msg->pool_name(), msg->pool_size(), msg->expected_object_count(),
              count, msg->base_addr());

        const std::string pool_name = msg->pool_name();

        IKVStore::pool_t pool;
        if (pool_mgr.check_for_open_pool(pool_name, pool)) {
          if (msg->flags() & IMCAS::ADO_FLAG_CREATE_ONLY) {
            if (debug_level() > 0)
              PWRN("request to create pool denied, create only specified on existing pool");
            response->pool_id = IKVStore::POOL_ERROR;
            response->set_status(E_FAIL);
          }
          else {
            pool_mgr.add_reference(pool);
          }
        }
        else {
          pool = pool_mgr.create_and_register_pool(_i_kvstore.get(),
                                                   msg->pool_name(),
                                                   msg->pool_size(),
                                                   msg->expected_object_count(),
                                                   msg->flags());

          if (pool == IKVStore::POOL_ERROR) {
            response->pool_id = 0;
            response->set_status(E_FAIL);
            PWRN("unable to create pool (%s)", pool_name.c_str());
          }
          else {
            response->pool_id = pool;
            response->set_status(S_OK);
          }

          CPINF(1, "OP_CREATE: new pool id: %lx", pool);

          /* check for ability to pre-register memory with RDMA stack */
          nupm::region_descriptor regions;
          status_t             hr;
          if ((hr = _i_kvstore->get_pool_regions(pool, regions)) == S_OK) {
            for (auto &r : regions.address_map()) {
              CPLOG(2, "region: %p %lu MiB", ::base(r), REDUCE_MB(::size(r)));
              /* pre-register memory region with RDMA */
#if 0
              handler->ondemand_register(r.iov_base, r.iov_len);
#endif
              handler->ondemand_register(common::make_const_byte_span(r));
            }
          }
          else {
            CPLOG(2, "pool region query NOT supported, using on-demand");
          }
        }

        if (pool && ado_enabled()) { /* if ADO is enabled start ADO process */
          IADO_proxy *ado  = nullptr;
          pool_desc_t desc = {pool_name, msg->pool_size(), msg->flags(),
                              msg->expected_object_count(), false,
                              reinterpret_cast<void*>(msg->base_addr())};

          conditional_bootstrap_ado_process(_i_kvstore.get(), handler, pool, ado, desc);
        }
        static unsigned count2 = 0;
        count2++;

        CPINF(1, "POOL CREATE: OK, pool_id=%lx (%u)", pool, count2);

      } break;
      case mcas::protocol::OP_OPEN: {
        CPINF(1, "POOL OPEN: name=%s", msg->pool_name());

        IKVStore::pool_t pool;
        const std::string pool_name(msg->pool_name());

        /* check that pool is not already open */
        if (pool_mgr.check_for_open_pool(pool_name, pool)) {

          CPLOG(2,"Shard: reusing existing open pool (%p)", reinterpret_cast<void *>(pool));
          /* if pool already open just increment reference */
          pool_mgr.add_reference(pool);
          response->pool_id = pool;
        }
        else {

          /* pool is not open yet, open and register it */
          pool = pool_mgr.open_and_register_pool(_i_kvstore.get(), msg->pool_name());

          if (pool == IKVStore::POOL_ERROR) {
            response->pool_id = 0;
            response->set_status(E_INVAL);
          }
          else {
            response->pool_id = pool;
            response->set_status(S_OK);
          }
        }
        CPINF(2, "POOL OPEN: pool id: %lx", pool);

        if (pool != IKVStore::POOL_ERROR && ado_enabled()) { /* if ADO is enabled start ADO process */
          IADO_proxy *ado  = nullptr;

          pool_desc_t desc = {pool_name, msg->pool_size(), msg->flags(),
                              msg->expected_object_count(), true,
                              reinterpret_cast<void*>(msg->base_addr())};

          conditional_bootstrap_ado_process(_i_kvstore.get(), handler, pool, ado, desc);
        }
      } break;
      case mcas::protocol::OP_CLOSE: {
        CPINF(1, "POOL CLOSE: pool_id=%lx", msg->pool_id());

        if (!pool_mgr.is_pool_open(msg->pool_id())) {
          response->set_status(E_INVAL);
        }
        else {
          /* release pool reference, if its zero, we can close pool for real */
          if (pool_mgr.release_pool_reference(_i_kvstore.get(), msg->pool_id())) {
            CPLOG(1, "Shard: pool reference now zero. pool_id=%lx", msg->pool_id());

            /* close ADO process on pool close */
            if (ado_enabled()) {
              {
                auto ado_itf = make_itf_ref(get_ado_interface(msg->pool_id()));

                if (ado_itf->ref_count() == 1) {
                  /* ADO has is being released */
                  ado_itf->shutdown();
                  _ado_map.remove(ado_itf.get());
                }
              }

              // HACK
              _ado_pool_map.release(msg->pool_id());
            }

            auto rc = _i_kvstore->close_pool(msg->pool_id());
            
            if (debug_level() && rc != S_OK)
              PWRN("Shard: close_pool result:%d", rc);
            
            response->set_status(rc);
          }
          else {
            response->set_status(S_OK);
          }
        }
      } break;
      case mcas::protocol::OP_DELETE: {
        CPINF(1, "POOL DELETE pool_id=%lx (name %s)", msg->pool_id(), msg->pool_name());

        if (msg->pool_id() > 0 && pool_mgr.is_pool_open(msg->pool_id())) {
          CPLOG(2, "POOL DELETE by handle: pool_id=%lx", msg->pool_id());

          try {
            if (pool_mgr.pool_reference_count(msg->pool_id()) == 1) {
              CPLOG(2,"POOL DELETE reference count is 1 deleting for real");

              auto pool_name = pool_mgr.pool_name(msg->pool_id());

              if (!pool_mgr.release_pool_reference(_i_kvstore.get(), msg->pool_id()))
                throw Logic_exception("invalid pool reference count");

              /* notify ADO if needed */
              if (ado_enabled()) {
                auto ado_itf = get_ado_interface(msg->pool_id());

                /* send message to ADO, but perform closure only
                   when a response is given back from the ADO.
                   we can't block here because the shard
                   thread must keep going to avoid cross-client
                   degradation */
                if (ado_itf->send_op_event(ADO_op::POOL_DELETE) != S_OK)
                  throw General_exception("send_op_event to ADO failed");
              }
              else {
                /* close and delete pool */
                _i_kvstore->close_pool(msg->pool_id());

                try {
                  response->set_status(_i_kvstore->delete_pool(pool_name));
                }
                catch (...) {
                  PWRN("Shard: pool delete failed");
                  response->set_status(E_FAIL);
                }
              }
            }
            else {
              response->set_status(E_BUSY);
            }
          }
          catch (const std::invalid_argument &e) {
            throw e;
          }
        }
        /* try delete by pool name */
        else {
          CPLOG(2, "POOL DELETE by name: name=%s", msg->pool_name());

          IKVStore::pool_t pool;
          const auto       pool_name = msg->pool_name();

          response->pool_id = 0;
          /* check if pool is still open; return error if it is */
          if (pool_mgr.check_for_open_pool(pool_name, pool)) {
            if (debug_level() > 2) PWRN("Shard: pool delete on pool that is still open");

            response->set_status(IKVStore::E_ALREADY_OPEN);
          }
          else {
            response->set_status(_i_kvstore->delete_pool(msg->pool_name()));
          }
        }
      } break;
      default:
        throw Protocol_exception("%s - bad operation (msg->op = %d)", __func__, msg->op());
      }
  }
  catch (std::exception &e) {
    PERR("Unhandled exception processing a request OP(%d): %s", msg->op(), e.what());
  }

  /* trim response length */
  response_iob->set_length(response->msg_len());

  /* finally, send response */
  handler->post_response(response_iob, response, __func__);
}

void Shard::add_locked_value_shared(const pool_t                         pool_id,
                                    component::IKVStore::key_t           key,
                                    void *                               target,
                                    size_t                               target_len,
                                    memory_registered<Connection_base> &&mr)
{
  auto it = _locked_values_shared.emplace(std::piecewise_construct, std::forward_as_tuple(target),
                                          std::forward_as_tuple(pool_id, key, target_len, std::move(mr)));
  ++it.first->second.count;
}

void Shard::add_locked_value_exclusive(const pool_t                         pool_id,
                                       component::IKVStore::key_t           key,
                                       void *                               target,
                                       size_t                               target_len,
                                       memory_registered<Connection_base> &&mr)
{
  auto it = _locked_values_exclusive.emplace(std::piecewise_construct, std::forward_as_tuple(target),
                                             std::forward_as_tuple(pool_id, key, target_len, std::move(mr)));
  ++it.first->second.count;
}

void Shard::release_locked_value_shared(const void *target)
{
  auto i = _locked_values_shared.find(target); /* search by target address */
  if (i == _locked_values_shared.end())
    throw Logic_exception("%s: bad target; value never locked? (%p)", __func__, target);

  if (i->second.count == 1) {
    static bool do_flush = common::env_value<bool>(flush_enable_key, true);
    _i_kvstore->unlock(i->second.pool, i->second.key, do_flush ? IKVStore::UNLOCK_FLAGS_FLUSH : 0);
    _locked_values_shared.erase(i);
  }
  else {
    i->second.count--;
  }
}

void Shard::release_locked_value_exclusive(const void *target)
{
  auto i = _locked_values_exclusive.find(target); /* search by target address */
  if (i == _locked_values_exclusive.end())
    throw Logic_exception("%s bad target; value never locked? (%p)", __func__, target);

  if (i->second.count == 1) {
    static bool do_flush = common::env_value<bool>(flush_enable_key, true);
    _i_kvstore->unlock(i->second.pool, i->second.key, do_flush ? IKVStore::UNLOCK_FLAGS_FLUSH : 0);
    _locked_values_exclusive.erase(i);
  }
  else {
    i->second.count--;
  }
}

void Shard::add_space_shared(const range<std::uint64_t> &range_, memory_registered<Connection_base> &&mr_)
{
  auto i = _spaces_shared
    .emplace(std::piecewise_construct, std::forward_as_tuple(range_), std::forward_as_tuple(std::move(mr_)))
    .first;

  ++i->second.count;

  CPLOG(2, "%s: [0x%" PRIx64 "..0x%" PRIx64 ") count %u", __func__, range_.first, range_.second, i->second.count);
}

void Shard::release_space_shared(const range<std::uint64_t> &range_)
{
  auto i = _spaces_shared.find(range_); /* search by max offset */
  if (i == _spaces_shared.end()) {
    throw Logic_exception("%s: bad target; space never located? (%" PRIx64 ":%" PRIx64 ")", __func__, range_.first,
                          range_.second);
  }

  CPLOG(2, "%s: [0x%" PRIx64 "..0x%" PRIx64 ") count %u", __func__, range_.first, range_.second, i->second.count);

  --i->second.count;

  if (i->second.count == 0) {
    _spaces_shared.erase(i);
  }
}

/* note, target address is used because it is unique for the shard */
void Shard::add_pending_rename(const pool_t pool_id, const void *target, const std::string &from, const std::string &to)
{
  CPLOG(2, "added pending rename %p %s->%s", target, from.c_str(), to.c_str());

  assert(_pending_renames.find(target) == _pending_renames.end());

  _pending_renames.emplace(std::piecewise_construct, std::forward_as_tuple(target),
                           std::forward_as_tuple(pool_id, from, to));
}

namespace
{
/* Several callers to lock only want to know whether the lock succeeded.
 * Translate all non-success value to E_FAIL, to simplify that test.
 */
bool is_locked(status_t rc)
{
  switch (rc) {
  case S_OK:
    return true;
  case S_OK_CREATED:
    return true;
  case E_FAIL:
    PWRN("%s failed to lock value: F_FAIL", __func__);
    return false;
  case E_LOCKED:
    PWRN("%s failed to lock value: F_LOCKED", __func__);
    return false;
  case IKVStore::E_KEY_NOT_FOUND:
    PWRN("%s failed to lock value: E_KEY_NOT_FOUND", __func__);
    return false;
  case IKVStore::E_TOO_LARGE:
    PWRN("%s failed to lock value: E_TOO_LARGE", __func__);
    dump_backtrace();
    return false;
  case E_NOT_SUPPORTED:
    PWRN("%s failed to lock value: E_NOT_SUPPORTED", __func__);
    return false;
  default:
    PWRN("%s failed to lock value: %d", __func__, rc);
    return false;
  };
}
}

void Shard::release_pending_rename(const void *target)
{
  try {
    auto info = _pending_renames.at(target);

    CPLOG(2, "renaming (%s) to (%s)", info.from.c_str(), info.to.c_str());

    void *                     value;
    size_t                     value_len = 8;
    component::IKVStore::key_t keyh;

    /* we do the lock/unlock first, because there might not be a prior
       object so this will create one on demand. */
    {
      if (! is_locked(_i_kvstore->lock(info.pool, info.to, IKVStore::STORE_LOCK_WRITE, value, value_len, keyh)))
        throw Logic_exception("%s lock failed", __func__);
    }
    if (_i_kvstore->unlock(info.pool, keyh) != S_OK) /* no flush needed */
      throw Logic_exception("%s unlock failed", __func__);

    if (_i_kvstore->swap_keys(info.pool, info.from, info.to) != S_OK)
      throw Logic_exception("%s swap_keys failed", __func__);

    if (_i_kvstore->erase(info.pool, info.from) != S_OK)
      throw Logic_exception("%s erase failed", __func__);

    _pending_renames.erase(target);

    /* now make available in the index */
    add_index_key(info.pool, info.to);
  }
  catch (std::out_of_range &err) {
    /* silent exception; there may not be a rename for this object
       if the release is coming from a get_direct
    */
  }
}

protocol::Message_IO_response * Shard::prepare_response(const Connection_handler *handler_,
                                                        buffer_t *iob_,
                                                        uint64_t request_id,
                                                        int status_)
{
  auto response =
    new (iob_->base()) protocol::Message_IO_response(iob_->length(), handler_->auth_id(), request_id);
  response->set_status(status_);

  iob_->set_length(response->base_message_size());
  return response;
}

void Shard::respond(Connection_handler *handler_,
                    buffer_t *iob_,
                    const protocol::Message_IO_request *msg_,
                    int status_,
                    const char * func_)
{
  auto response = prepare_response(handler_, iob_, msg_->request_id(), status_);
  handler_->post_response(iob_, response, func_);  // issue IO request response
}

/////////////////////////////////////////////////////////////////////////////
//   GET LOCATE    //
/////////////////////
void Shard::io_response_get_locate(Connection_handler *handler,
                                   const protocol::Message_IO_request *msg,
                                   buffer_t *iob)
{
  CPLOG(2, "GET_LOCATE: (%p) key=(%.*s) value_len=0z%zx request_id=%lu", common::p_fmt(this),
        static_cast<int>(msg->key_len()), msg->key(), msg->get_value_len(), msg->request_id());

  /* open memory */
  assert(msg->pool_id() > 0);

  auto status = S_OK;

  std::string k = msg->skey();

  /* lock value */
  component::IKVStore::key_t key_handle;
  void *                     target     = nullptr;
  size_t                     target_len = 0;
  status_t rc = _i_kvstore->lock(msg->pool_id(), k, IKVStore::STORE_LOCK_READ, target, target_len, key_handle);

  if ( ! is_locked(rc) ) { status = E_FAIL; }

  if (key_handle == component::IKVStore::KEY_NONE) {
    PWRN("%s failed to lock value returned KEY_NONE", __func__);
    status = E_FAIL;
  }

  if (status != S_OK) {
    respond(handler, iob, msg, status, __func__);
    ++_stats.op_failed_request_count;
  }
  else {
    locked_key lk(_i_kvstore.get(), msg->pool_id(), key_handle);

    assert(target);
    auto pool_id = msg->pool_id();

    std::uint64_t key = 0;
    try  {
      memory_registered<Connection_base> mr(debug_level(), handler, target, target_len, 0, 0);
      key = mr.key();
      /* register clean and deregister tasks for value */
      add_locked_value_shared(pool_id, lk.release(), target, target_len, std::move(mr));

      /* record key for signaling */
      if (ado_signal_post_get())
        add_target_keyname(target, k);
    }
    catch ( const std::exception &e ) {
      PWRN("%s failed: %s", __func__, e.what());
      status = E_FAIL;
    }

    auto response  = prepare_response(handler, iob, msg->request_id(), status);
    response->addr = reinterpret_cast<std::uint64_t>(target);
    response->key  = key;
    response->set_data_len_without_data(target_len);

    handler->post_send_buffer(iob, response, __func__);

    /* update stats */
    _stats.op_get_direct_count++;
  }
}

/////////////////////////////////////////////////////////////////////////////
//   GET RELEASE   //
/////////////////////
void Shard::io_response_get_release(Connection_handler *handler,
                                    const protocol::Message_IO_request *msg,
                                    buffer_t *iob)
{
  auto target = reinterpret_cast<const void *>(msg->addr);
  CPLOG(2, "GET_RELEASE: (%p) addr=(%p) request_id=%lu", common::p_fmt(this),
        target, msg->request_id());

  int status = S_OK;
  try {
    release_locked_value_shared(target);
  }
  catch (const Logic_exception &) {
    status = E_INVAL;
  }
  ++_stats.op_get_count;

  if (ado_signal_post_get()) {
    auto skey = release_target_keyname(target); /* recover key and remove entry from map */
    signal_ado("post-get",
               handler,
               msg->request_id(),
               msg->pool_id(),
               skey,
               IKVStore::lock_type_t::STORE_LOCK_READ);
    /* note: client will be signalled on return of this ADO call,
       therefore if the ADO operation stalls, the client will be stalled too.
    */
  }
  else {
    respond(handler, iob, msg, status, __func__);
  }
}

/////////////////////////////////////////////////////////////////////////////
//   PUT ADVANCE   //
/////////////////////
void Shard::io_response_put_advance(Connection_handler *handler,
                                    const protocol::Message_IO_request *msg,
                                    buffer_t *iob)
{
  CPLOG(2, "PUT_ADVANCE: (%p) key=(%.*s) value_len=%zu request_id=%lu", common::p_fmt(this),
        static_cast<int>(msg->key_len()), msg->key(), msg->get_value_len(), msg->request_id());

  /* open memory */
  assert(msg->pool_id() > 0);

  /* can't support dont stomp flag */
  if (msg->flags() & IKVStore::FLAGS_DONT_STOMP) {
    PWRN("PUT_ADVANCE failed IKVStore::FLAGS_DONT_STOMP not viable");
    _stats.op_failed_request_count++;
    respond(handler, iob, msg, E_INVAL, __func__);
  }
  else {
    auto status = S_OK;

    std::string actual_key = msg->skey();
    std::string k("___pending_");
    k += actual_key; /* we embed the actual key for recovery purposes */

    /* create (if needed) and lock value */
    component::IKVStore::key_t key_handle;
    void *                     target     = nullptr;
    size_t                     target_len = msg->get_value_len();
    assert(target_len > 0);
    status_t rcx = _i_kvstore->lock(msg->pool_id(), k, IKVStore::STORE_LOCK_WRITE, target, target_len, key_handle);

    if ( ! is_locked(rcx) || key_handle == component::IKVStore::KEY_NONE) {
      PWRN("PUT_ADVANCE failed to lock value");
      status = E_FAIL;
    }

    locked_key lk(_i_kvstore.get(), msg->pool_id(), key_handle);

    if (target_len != msg->get_value_len()) {
      PWRN("existing entry length does NOT equal request length");
      status = E_INVAL;
    }

    if (status != S_OK) {
      respond(handler, iob, msg, status, __func__);
      ++_stats.op_failed_request_count;
    }
    else {
      auto pool_id = msg->pool_id();

      // memory_registered<component::IFabric_connection> mr =
      // make_memory_registered(debug_level(), handler, target, target_len, 0,
      // 0);
      std::uint64_t key = 0;
      try
        {
          memory_registered<Connection_base> mr(debug_level(), handler, target, target_len, 0, 0);
          key = mr.key();

          /* register clean and rename tasks for value */
          add_locked_value_exclusive(pool_id, lk.release(), target, target_len, std::move(mr));
          add_pending_rename(pool_id, target, k, actual_key);

          /* record key for signaling */
          if (ado_signal_post_put())
            add_target_keyname(target, k);
        }
      catch ( const std::exception &e )
        {
          PLOG("%s failed: %s", __func__, e.what());
          status = E_FAIL;
        }

      auto response  = prepare_response(handler, iob, msg->request_id(), status);
      response->addr = reinterpret_cast<std::uint64_t>(target);
      response->key  = key;

      handler->post_send_buffer(iob, response, __func__);

      /* update stats */
      _stats.op_put_direct_count++;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////
//   PUT LOCATE    //
/////////////////////
void Shard::io_response_put_locate(Connection_handler *handler,
                                   const protocol::Message_IO_request *msg,
                                   buffer_t *iob)
{
  CPLOG(2, "PUT_LOCATE: (%p) key=(%.*s) value_len=0x%zu request_id=%lu", common::p_fmt(this),
        static_cast<int>(msg->key_len()), msg->key(), msg->get_value_len(), msg->request_id());

  /* open memory */
  assert(msg->pool_id() > 0);

  /* can't support dont stomp flag */
  if (msg->flags() & IKVStore::FLAGS_DONT_STOMP) {
    PWRN("PUT_ADVANCE failed IKVStore::FLAGS_DONT_STOMP not viable");
    _stats.op_failed_request_count++;
    respond(handler, iob, msg, E_INVAL, __func__);
  }
  else {
    auto status = S_OK;

    std::string actual_key = msg->skey();
    std::string k("___pending_");
    k += actual_key; /* we embed the actual key for recovery purposes */

    /* create (if needed) and lock value */
    component::IKVStore::key_t key_handle;
    void *                     target     = nullptr;
    size_t                     target_len = msg->get_value_len();
    assert(target_len > 0);

    /* The initiative to unlock lies with the caller if status returns S_OK, else it lies with us. */
    status_t rc = _i_kvstore->lock(msg->pool_id(), k, IKVStore::STORE_LOCK_WRITE, target, target_len, key_handle);

    if ( ! is_locked(rc) ) { status = E_FAIL; }

    if (key_handle == component::IKVStore::KEY_NONE) {
      PWRN("%s failed to lock value returned KEY_NONE", __func__);
      status = E_INVAL;
    }

    if (status != S_OK) {
      respond(handler, iob, msg, status, __func__);
      ++_stats.op_failed_request_count;
    }
    else {
      locked_key lk(_i_kvstore.get(), msg->pool_id(), key_handle);
      assert(target);
      auto pool_id = msg->pool_id();

      std::uint64_t key = 0;
      try  {
        memory_registered<Connection_base> mr(debug_level(), handler, target, target_len, 0, 0);
        key = mr.key();
        /* register clean and rename tasks for value */
        add_locked_value_exclusive(pool_id, lk.release(), target, target_len, std::move(mr));
        add_pending_rename(pool_id, target, k, actual_key);

        if (ado_signal_post_put())
          add_target_keyname(target, actual_key);
      }
      catch ( const std::exception &e ) {
        PLOG("%s failed: %s", __func__, e.what());
        status = E_FAIL;
      }

      auto response  = prepare_response(handler, iob, msg->request_id(), status);
      response->addr = reinterpret_cast<std::uint64_t>(target);
      response->key  = key;

      handler->post_send_buffer(iob, response, __func__);

      /* update stats */
      _stats.op_put_direct_count++;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////
//   PUT RELEASE   //
/////////////////////
void Shard::io_response_put_release(Connection_handler *handler, const protocol::Message_IO_request *msg, buffer_t *iob)
{
  auto target = reinterpret_cast<const void *>(msg->addr);
  CPLOG(2, "PUT_RELEASE: (%p) addr=(%p) request_id=%lu", common::p_fmt(this),
        target, msg->request_id());

  int status = S_OK;

  try {
    release_locked_value_exclusive(target);
    release_pending_rename(target);
  }
  catch (const Logic_exception &) {
    status = E_INVAL;
  }

  ++_stats.op_put_count;

  if (ado_signal_post_put()) {
    auto skey = release_target_keyname(target); /* recover key and remove entry from map */
    signal_ado("post-put",
               handler,
               msg->request_id(),
               msg->pool_id(),
               skey,
               IKVStore::lock_type_t::STORE_LOCK_READ);
    /* note: client will be signalled on return of this ADO call,
       therefore if the ADO operation stalls, the client will be stalled too.
    */
  }
  else {
    respond(handler, iob, msg, status, __func__);
  }
}

/////////////////////////////////////////////////////////////////////////////
//   PUT           //
/////////////////////
void Shard::io_response_put(Connection_handler *handler, const protocol::Message_IO_request *msg, buffer_t *iob)
{
  /* for basic 'puts' we have to do a memcpy - to support "in-place"
     puts for larger data, we use a two-stage operation
  */

  if (debug_level() > 2) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
    PMAJOR("PUT: (%p) key=(%.*s) value=(%.*s ...) len=(%zu)", common::p_fmt(this), int(msg->key_len()),
           msg->key(), (min(int(msg->get_value_len()), 20)), msg->value(), msg->get_value_len());
#pragma GCC diagnostic pop
  }
  {
    int status = S_OK;
    if (UNLIKELY(msg->is_scbe())) {
      // short-circuit backend - testing only
      CPLOG(2, "PUT: short-circuited backend");
    }
    else {
      const std::string key = msg->skey();

      status = _i_kvstore->put(msg->pool_id(), key, msg->value(), msg->get_value_len(), msg->flags());

      if (debug_level() > 2) {
        if (status == E_ALREADY_EXISTS) {
          PWRN("kvstore->put returned E_ALREADY_EXISTS");
          _stats.op_failed_request_count++;
        }
        else {
          PLOG("kvstore->put returned %d", status);
        }
      }

      add_index_key(msg->pool_id(), key);

      /* optional ado signaling of put event */
      if (ado_signal_post_put()) {
        signal_ado("post-put",
                   handler,
                   msg->request_id(),
                   msg->pool_id(),
                   key,
                   IKVStore::lock_type_t::STORE_LOCK_READ);
        /* note: client will be signalled on return of this ADO call,
           therefore if the ADO operation stalls, the client will be stalled too.
        */
      }
    }
    /* update stats */
    ++_stats.op_put_count;

    if (!ado_signal_post_put())
      respond(handler, iob, msg, status, __func__);
  }
}

/////////////////////////////////////////////////////////////////////////////
//   GET           //
/////////////////////
void Shard::io_response_get(Connection_handler *handler, const protocol::Message_IO_request *msg, buffer_t *iob)
{
  if (debug_level() > 2)
    PMAJOR("GET: (%p) (request=%lu,buffer_size=%zu) key=(%.*s) ",
           common::p_fmt(this), msg->request_id(),
           msg->get_value_len(), int(msg->key_len()), msg->key());

  if (msg->is_scbe()) {
    CPLOG(2, "GET: short-circuited backend");

    respond(handler, iob, msg, S_OK, __func__);
  }
  else {
    /* Maximum length of a speculative get. If the value length exceeds,
     * GET_DIRECT_THRESHOLD, the memcpy will have been wasted. Do not
     * specify a larger value that you are willing to re-fetch in that
     * (A get with a callback to a client-controlled allocator would avoid
     * wasting the memcpy by consulting the client before running the
     * memcpy.)
     */
    static constexpr std::size_t GET_DIRECT_THRESHOLD = KiB(128);
    static_assert(GET_DIRECT_THRESHOLD <= TWO_STAGE_THRESHOLD, "get_direct threshold must not exceed a single-message data size");
    std::string k = msg->skey();

    if ( ! ado_signal_post_get() )
    {
      auto response = prepare_response(handler, iob, msg->request_id(), S_OK);
      /* Maximum possible size for buffer */
      std::size_t data_len = GET_DIRECT_THRESHOLD;
      status_t rc = _i_kvstore->get_direct(msg->pool_id(), k, response->data(), data_len);
       /* If got the whole value */
      if ( rc == S_OK && data_len <= GET_DIRECT_THRESHOLD )
      {
        /* value can fit in message buffer, copy */
        CPLOG(2, "Shard: performing memcpy for very small get");

        response->set_data_len(data_len);
        iob->set_length(response->msg_len());
        handler->post_response(iob, response, __func__);

        _stats.op_get_count++;
        return;
      }
    }

    ::iovec value_out{nullptr, 0};

    component::IKVStore::key_t key_handle;

    status_t rc = _i_kvstore->lock(msg->pool_id(),
                                   k,
                                   IKVStore::STORE_LOCK_READ,
                                   value_out.iov_base,
                                   value_out.iov_len,
                                   key_handle);

    if ( ! is_locked(rc) || key_handle == component::IKVStore::KEY_NONE) { /* key not found */
      CPLOG(2, "Shard: locking value failed");
      respond(handler, iob, msg, rc, __func__);
      ++_stats.op_failed_request_count;
    }
    else {
      locked_key lk(_i_kvstore.get(), msg->pool_id(), key_handle);

      CPLOG(2, "Shard: locked OK: value_out=%p (%.*s ...) value_out_len=%lu", value_out.iov_base,
            boost::numeric_cast<int>(min(boost::numeric_cast<int>(value_out.iov_len), 20)),
            static_cast<char *>(value_out.iov_base), value_out.iov_len);

      assert(value_out.iov_len);
      assert(value_out.iov_base);

      /*
       * The value is returned in one of three places:
       *   (1) ! direct *and* below TWO_STAGE_THRESHOLD     : adjoining the
       * message On completion the single buffer_t will be resturned as with any
       * response (2) ! direct *and* would fit in a receive buffer : two buffers
       * in one packet (what is a packet?) On completion of the single post: (a)
       * the value mus be unlocked, or at least moved to a queue from which it
       * will eventually be pulled and unlocked (b) the second buffer_t must be
       * recovered and deleted (3) direct or would not fit in a buffer :
       * separately posted in immediately following message On completion of the
       * second post (a) the value mus be unlocked, or at least moved to a queue
       * from which it will eventually be pulled and unlocked (b) the buffer_t
       * must be recovered and deleted
       *
       * If the choice is not (1), how does the client know that? Must it know
       * about the value of TWO_STAGE_THRESHOLD?
       */
      bool is_direct = msg->is_direct();

      /* optimize based on size */
      if (!is_direct && (value_out.iov_len < TWO_STAGE_THRESHOLD)) {

        /* value can fit in message buffer, let's copy instead of
           performing two-part DMA */
        CPLOG(2, "Shard: performing memcpy for small get");

        if(ado_signal_post_get()) {

          /* actually, this will override the memcpy optimization and
             do a dual buffer send. */
          _i_kvstore->unlock(msg->pool_id(), lk.release(), IKVStore::UNLOCK_FLAGS_FLUSH);

          signal_ado("post-get",
                     handler,
                     msg->request_id(),
                     msg->pool_id(),
                     k,
                     IKVStore::lock_type_t::STORE_LOCK_READ,
                     true /* special 'get' response */);
        }
        else {
          auto response = prepare_response(handler, iob, msg->request_id(), S_OK);
          response->copy_in_data(value_out.iov_base, value_out.iov_len);
          iob->set_length(response->msg_len());

          _i_kvstore->unlock(msg->pool_id(), lk.release(), IKVStore::UNLOCK_FLAGS_FLUSH);

          handler->post_response(iob, response, __func__);
        }

        _stats.op_get_count++;
      }
      else {
        CPLOG(2, "Shard: get using two stage get response (value_out_len=%lu)", value_out.iov_len);

        /* check if client has allocated sufficient space */
        if (msg->get_value_len() < value_out.iov_len) {
          _i_kvstore->unlock(msg->pool_id(), lk.release()); /* no flush needed ? */
          PWRN("Shard: client posted insufficient space for get operation.");
          ++_stats.op_failed_request_count;
          respond(handler, iob, msg, E_INSUFFICIENT_SPACE, __func__);
        }
        else {
          try {
            memory_registered<Connection_base> mr(debug_level(), handler, value_out.iov_base, value_out.iov_len, 0, 0);

            auto desc = mr.desc();
            auto response = prepare_response(handler, iob, msg->request_id(), S_OK);

            response->set_data_len_without_data(value_out.iov_len);
            assert(response->get_status() == S_OK);

            /* register clean up task for value */
            add_locked_value_shared(msg->pool_id(), lk.release(), value_out.iov_base, value_out.iov_len, std::move(mr));

            if (!is_direct && (value_out.iov_len <= (handler->IO_buffer_size() - response->base_message_size()))) {
              CPLOG(2, "posting response header and value together");

              /* post both buffers together in same response packet */
              if(ado_signal_post_get()) {

                signal_ado("post-get",
                           handler,
                           msg->request_id(),
                           msg->pool_id(),
                           k,
                           IKVStore::lock_type_t::STORE_LOCK_READ,
                           true);

                /* note: client will be signalled on return of this ADO call,
                   therefore if the ADO operation stalls, the client will be stalled too.
                */
              }
              else {
                handler->post_response_with_value(iob, value_out, desc, response, __func__);
              }
            }
            else {
              /* client should have used GET_LOCATE */
              respond(handler, iob, msg, component::IKVStore::E_TOO_LARGE, __func__);
            }
          }
          catch ( const std::exception &e ) {
            PLOG("%s failed: %s", __func__, e.what());
            respond(handler, iob, msg, E_FAIL, __func__);
          }
          _stats.op_get_twostage_count++;
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////
//   ERASE         //
/////////////////////
void Shard::io_response_erase(Connection_handler *handler, const protocol::Message_IO_request *msg, buffer_t *iob)
{
  std::string key = msg->skey();

  status_t status;

  if (ado_signal_post_erase()) {

    /* actually it may not have been erased by the time the
       notification goes through.  We're just sending an
       asynchronous notification of erase to the ADO.
       if we want a pre-erase with guarantees we'll have to
       support a "deferred" erase (similar to unlock).
    */
    signal_ado_async_nolock("post-erase",
                            handler,
                            msg->request_id(),
                            msg->pool_id(),
                            key);
  }

  status = _i_kvstore->erase(msg->pool_id(), key);

  if (status == S_OK) {
    remove_index_key(msg->pool_id(), key);
  }
  else {
    _stats.op_failed_request_count++;
  }

  _stats.op_erase_count++;

  respond(handler, iob, msg, status, __func__);
}

/////////////////////////////////////////////////////////////////////////////
//   CONFIGURE     //
/////////////////////
void Shard::io_response_configure(Connection_handler *handler, const protocol::Message_IO_request *msg, buffer_t *iob)
{
  if (debug_level() > 1) PMAJOR("Shard: pool CONFIGURE (%s)", msg->cmd());
  respond(handler, iob, msg, process_configure(msg), __func__);
}

void Shard::process_message_IO_request(Connection_handler *handler, const protocol::Message_IO_request *msg)
  try {
    handler->msg_recv_log(msg, __func__);
    using namespace component;

    const auto iob = handler->allocate_send();
    assert(iob);

    ++_stats.op_request_count;
    switch (msg->op()) {
    case protocol::OP_PUT_LOCATE:
      io_response_put_locate(handler, msg, iob);
      break;
    case protocol::OP_PUT_RELEASE:
      io_response_put_release(handler, msg, iob);
      break;
    case protocol::OP_GET_LOCATE:
      io_response_get_locate(handler, msg, iob);
      break;
    case protocol::OP_GET_RELEASE:
      io_response_get_release(handler, msg, iob);
      break;
    case protocol::OP_LOCATE:
      io_response_locate(handler, msg, iob);
      break;
    case protocol::OP_RELEASE:
      io_response_release(handler, msg, iob);
      break;
    case protocol::OP_RELEASE_WITH_FLUSH:
      io_response_release_with_flush(handler, msg, iob);
      break;
    case protocol::OP_PUT:
      io_response_put(handler, msg, iob);
      break;
    case protocol::OP_GET:
      io_response_get(handler, msg, iob);
      break;
    case protocol::OP_ERASE:
      io_response_erase(handler, msg, iob);
      break;
    case protocol::OP_CONFIGURE:
      io_response_configure(handler, msg, iob);
      break;
    default:
      throw Protocol_exception("operation not implemented");
    }
  }
  catch ( std::exception &e )
    {
      PWRN("%s: exception in op %i handling", __func__, int(msg->op()));
      throw;
    }

namespace
{
using byte_span = common::byte_span;
auto region_breaks(const std::vector<byte_span> regions_) -> std::vector<byte_span>
{
  std::vector<byte_span> region_breaks;

  std::size_t offset = 0;
  for (const auto &i : regions_) {
    offset += ::size(i);
    region_breaks.push_back(common::make_byte_span(::base(i), offset));
  }

  return region_breaks;
}
}  // namespace

auto Shard::offset_to_sg_list(range<std::uint64_t> t,
                              const std::vector<byte_span> &region_breaks_) -> sg_result
{
  CPLOG(2, "region break count %zu", region_breaks_.size());
  for (const auto &e : region_breaks_) {
    CPLOG(2, "region break %p len 0x%zx", ::base(e), ::size(e));
  }

  auto it_begin = std::upper_bound(region_breaks_.begin(),
                                   region_breaks_.end(),
                                   t.first,
                                   [](const uint64_t a, const byte_span &b) {
                                     return a < reinterpret_cast<std::uint64_t>(::data(b));
                                   });

  auto it_end   = std::upper_bound(region_breaks_.begin(),
                                   region_breaks_.end(),
                                   t.second,
                                   [](const uint64_t a, const byte_span &b) {
                                     return a < reinterpret_cast<std::uint64_t>(::data(b));
                                   });

  CPLOG(2, "it_begin %zu it_end %zu", it_begin - region_breaks_.begin(), it_end - region_breaks_.begin());

  /* beginning and ending offsets, within the beginning and ending regions,
   * respectively */
  auto begin_off = t.first - (it_begin == region_breaks_.begin() ? 0 : ::size(*(it_begin - 1)));
  auto end_off   = t.second - (it_end == region_breaks_.begin() ? 0 : ::size(*(it_end - 1)));

  std::vector<::iovec> transfer;
  auto                 mr_low  = std::numeric_limits<std::uint64_t>::max();
  auto                 mr_high = std::numeric_limits<std::uint64_t>::min();

  CPLOG(2, "initial begin_off 0x%" PRIx64 " end_off 0x%" PRIx64 " mr_low 0x%" PRIx64 " mr_high 0x%" PRIx64, begin_off,
        end_off, mr_low, mr_high);

  /* the range from t_begin to t_end may be contained in discontigous memory.
   * build a scatter-gather list.
   */
  std::vector<protocol::Message_IO_response::locate_element> sg_list;

  /* Entries before the last */
  while (it_begin != it_end) {
    /* range to fetch must fit within a single region, This one does not. */
    assert(::data(*it_begin));

    CPLOG(2, "loop iov_base %p iov_len 0x%zx begin_off %zu", ::base(*it_begin), ::size(*it_begin), begin_off);

    const auto m_low  = reinterpret_cast<std::uint64_t>(::data(*it_begin) + begin_off);
    const auto m_high = reinterpret_cast<std::uint64_t>(::data_end(*it_begin));
    mr_low            = std::min(mr_low, m_low);
    mr_high           = std::max(mr_high, m_high);

    CPLOG(2, "loop m_low 0x%" PRIu64 " m_high 0x%" PRIu64 " mr_low 0x%" PRIu64 " mr_high 0x%" PRIu64, m_low, m_high,
          mr_low, mr_high);

    sg_list.push_back(protocol::Message_IO_response::locate_element{m_low, m_high - m_low});
    begin_off = 0;
  }

  /* last entry */
  assert(::base(*it_begin));

  CPLOG(2, "final iov_base %p iov_len 0x%zx begin_off %zu", ::base(*it_begin), ::size(*it_begin), begin_off);

  auto m_low = reinterpret_cast<std::uint64_t>(::data(*it_begin) + begin_off);
  /* end of last element, not to exceed the pools memory element */
  auto excess_length = ::size(*it_begin) < end_off ? end_off - ::size(*it_begin) : 0;
  auto m_high        = reinterpret_cast<std::uint64_t>(::data(*it_begin) + end_off - excess_length);
  mr_low             = std::min(mr_low, m_low);
  mr_high            = std::max(mr_high, m_high);

  CPLOG(2, "final m_low 0x%" PRIx64 " m_high 0x%" PRIx64 " mr_low 0x%" PRIx64 " mr_high 0x%" PRIx64 " size 0x%" PRIx64,
        m_low, m_high, mr_low, mr_high, m_high - m_low);

  sg_list.push_back(protocol::Message_IO_response::locate_element{m_low, m_high - m_low});
  return sg_result{ std::move(sg_list), mr_low, mr_high, excess_length };
}

/////////////////////////////////////////////////////////////////////////////
//   LOCATE        //
/////////////////////
void Shard::io_response_locate(Connection_handler *handler, const protocol::Message_IO_request *msg, buffer_t *iob)
{
  range<std::uint64_t> t(msg->get_offset(), msg->get_offset() + msg->get_size());
  CPLOG(2, "LOCATE: (%p) offset 0x%zx size 0x%zx request_id=%lu", common::p_fmt(this), msg->get_offset(),
        msg->get_size(), msg->request_id());

  nupm::region_descriptor regions;
  auto status = _i_kvstore->get_pool_regions(msg->pool_id(), regions);

  if (status == S_OK) {
    const auto rb = region_breaks(regions.address_map());
    auto sgr = offset_to_sg_list(t, rb);
    /* Register the entire range */
    std::uint64_t key = 0;
    try
      {
        memory_registered<Connection_base> mr(debug_level(),
                                              handler,
                                              reinterpret_cast<void *>(sgr.mr_low),
                                              sgr.mr_high - sgr.mr_low,
                                              0,
                                              0);
        key = mr.key();
        /* register deregister task for space */
        add_space_shared(range<std::uint64_t>(t.first, t.second - sgr.excess_length), std::move(mr));
      }
    catch ( const std::exception &e )
      {
        PLOG("%s failed: %s", __func__, e.what());
        status = E_FAIL;
      }

    /* respond, with the scatter-gather list as "data" */
    auto response = prepare_response(handler, iob, msg->request_id(), status);
    if ( status == S_OK )
      {
        response->copy_in_data(&sgr.sg_list.front(), sgr.sg_list.size() * sizeof *sgr.sg_list.begin());
        iob->set_length(response->msg_len());
        response->key = key;
        handler->post_send_buffer(iob, response, __func__);
      }
    else
      {
        respond(handler, iob, msg, status, __func__);
      }

    /* update stats */
    ++_stats.op_get_direct_offset_count;
  }
  else {
    respond(handler, iob, msg, status, __func__);
  }
}

/////////////////////////////////////////////////////////////////////////////
//   RELEASE       //
/////////////////////
void Shard::io_response_release(Connection_handler *handler, const protocol::Message_IO_request *msg, buffer_t *iob)
{
  range<std::uint64_t> t(msg->get_offset(), msg->get_offset() + msg->get_size());
  CPLOG(2, "RELEASE: (%p) offset 0x%zx size %zu request_id=%lu", common::p_fmt(this), t.first,
        msg->get_size(), msg->request_id());

  int status = S_OK;
  try {
    release_space_shared(t);
  }
  catch (const Logic_exception &e) {
    CPLOG(2, "%s: RELEASE: (%p) [0x%" PRIx64 "..0x%" PRIx64 ") error %s", __func__, common::p_fmt(this),
          t.first, t.second, e.cause());
    status = E_INVAL;
  }
  respond(handler, iob, msg, status, __func__);
}

/////////////////////////////////////////////////////////////////////////////
//   RELEASE_WITH_FLUSH   //
////////////////////////////
void Shard::io_response_release_with_flush(Connection_handler *handler, const protocol::Message_IO_request *msg, buffer_t *iob)
{
  const char *tag = "RELEASE_WITH_FLUSH";
  range<std::uint64_t> t(msg->get_offset(), msg->get_offset() + msg->get_size());
  CPLOG(2, "%s: (%p) offset 0x%zx size %zu request_id=%lu", tag, common::p_fmt(this), t.first,
        msg->get_size(), msg->request_id());

  nupm::region_descriptor regions;
  auto status = _i_kvstore->get_pool_regions(msg->pool_id(), regions);

  if (status == S_OK) {
    const auto rb = region_breaks(regions.address_map());

    auto sgr = offset_to_sg_list(t, rb);
    try {
      for ( const auto &e : sgr.sg_list )
        {
          auto s = _i_kvstore->flush_pool_memory(msg->pool_id(), reinterpret_cast<const void *>(e.addr), e.len);
          if ( status == S_OK ) {
            status = s;
          }
        }

      release_space_shared(t);
    }
    catch (const Logic_exception &e) {
      CPLOG(2, "%s: %s: (%p) [0x%" PRIx64 "..0x%" PRIx64 ") error %s", tag, __func__, common::p_fmt(this),
            t.first, t.second, e.cause());
      status = E_INVAL;
    }
  }
  respond(handler, iob, msg, status, __func__);
}

void Shard::process_info_request(Connection_handler *handler, const protocol::Message_INFO_request *msg, common::profiler &pr_)
{
  handler->msg_recv_log(msg, __func__);

  if (msg->type() == protocol::INFO_TYPE_FIND_KEY) {
    CPLOG(1, "Shard: INFO request INFO_TYPE_FIND_KEY (%s)", msg->c_str());

    if (_index_map == nullptr) { /* index does not exist */
      PLOG("Shard: cannot perform regex request, no index!! use "
           "configure('AddIndex::VolatileTree') or similar for dynamic loading ");
      const auto                       iob      = handler->allocate_send();
      protocol::Message_INFO_response *response = new (iob->base()) protocol::Message_INFO_response(handler->auth_id());

      response->set_status(E_INVAL);
      handler->post_send_buffer(iob, response, __func__);
      return;
    }

    try {
      add_task_list(new Key_find_task(msg->c_str(),
                                      msg->offset,
                                      handler,
                                      _index_map->at(msg->pool_id()).get(),
                                      debug_level()));
    }
    catch (...) {
      const auto                       iob      = handler->allocate_send();
      protocol::Message_INFO_response *response = new (iob->base()) protocol::Message_INFO_response(handler->auth_id());

      response->set_status(E_INVAL);
      handler->post_send_buffer(iob, response, __func__);
      return;
    }

    return; /* response is not issued straight away */
  }

  const auto iob = handler->allocate_send();
  assert(iob);

  CPLOG(1, "Shard: INFO request type:0x%X", msg->type());

  /* stats request handler */
  if (msg->type() == protocol::INFO_TYPE_GET_STATS) {
    protocol::Message_stats *response = new (iob->base()) protocol::Message_stats(handler->auth_id(), _stats);
    response->set_status(S_OK);
    iob->set_length(sizeof(protocol::Message_stats));

    if (debug_level() > 1) dump_stats();

    handler->post_send_buffer(iob, response, __func__);
  }

  /* info requests */
  protocol::Message_INFO_response *response = new (iob->base()) protocol::Message_INFO_response(handler->auth_id());

  if (msg->type() == component::IKVStore::Attribute::COUNT) {
    response->set_value(_i_kvstore->count(msg->pool_id()));
    response->set_status(S_OK);
    pr_.start();
  }
  else if (msg->type() == component::IKVStore::Attribute::VALUE_LEN) {
    std::vector<uint64_t> v;
    std::string           key = msg->key();
    auto hr = _i_kvstore->get_attribute(msg->pool_id(), component::IKVStore::Attribute::VALUE_LEN, v, &key);
    response->set_status(hr);

    if (hr == S_OK && v.size() == 1) {
      response->set_value(v[0]);
    }
    else {
      PWRN("_i_kvstore->get_attribute failed for value_len (key=%s)", key.c_str());
      response->set_value(0);
    }
    CPLOG(1, "Shard: INFO reqeust INFO_TYPE_VALUE_LEN rc=%d val=%lu", hr, response->value_numeric());
  }
  else {
    std::vector<uint64_t> v;
    std::string           key = msg->key();
    auto                  hr =
      _i_kvstore->get_attribute(msg->pool_id(), static_cast<component::IKVStore::Attribute>(msg->type()), v, &key);
    response->set_status(hr);

    if (hr == S_OK && v.size() == 1) {
      response->set_value(v[0]);
    }
    else {
      /* crc32 we can do here also */
      if (msg->type() == component::IKVStore::Attribute::CRC32) {
        response->set_status(S_OK);
        void *                     p     = nullptr;
        size_t                     p_len = 0;
        component::IKVStore::key_t key_handle;
        status_t rc = _i_kvstore->lock(msg->pool_id(), key, component::IKVStore::STORE_LOCK_READ, p, p_len, key_handle);

        if ( ! is_locked(rc) || key_handle == component::IKVStore::KEY_NONE) {
          response->set_status(E_FAIL);
          response->set_value(0);
        }
        else {
          locked_key lk(_i_kvstore.get(), msg->pool_id(), key_handle);
          /* do CRC */
          uint32_t crc = uint32_t(crc32(0, static_cast<const Bytef *>(p), uInt(p_len)));
          response->set_status(S_OK);
          response->set_value(crc);
        }
      }
      else {
        PWRN("_i_kvstore->get_attribute failed");
        response->set_status(E_FAIL);
        response->set_value(0);
      }
    }
    CPLOG(1, "Shard: INFO reqeust INFO_TYPE_VALUE_LEN rc=%d val=%lu", hr, response->value());
  }

  iob->set_length(response->base_message_size());
  handler->post_send_buffer(iob, response, __func__);
}

void Shard::process_tasks(unsigned &idle)
{
 retry:
  for (task_list_t::iterator i = _tasks.begin(); i != _tasks.end(); i++) {
    auto t = *i;
    assert(t);
    idle = 0;

    status_t s = t->do_work();
    if (s != component::IKVStore::S_MORE) {
      auto handler      = t->handler();
      auto response_iob = handler->allocate_send();
      assert(response_iob);
      protocol::Message_INFO_response *response =
        new (response_iob->base()) protocol::Message_INFO_response(handler->auth_id());

      if (s == S_OK) {
        response->set_value(response_iob->length(), t->get_result(), t->get_result_length(), t->matched_position());
#if 0
        response->offset = t->matched_position();
#endif
        response->set_status(S_OK);
        response_iob->set_length(response->message_size());
      }
      else if (s == E_FAIL) {
        response->set_status(E_FAIL);
        response_iob->set_length(response->base_message_size());
      }
      else {
        throw Logic_exception("unexpected task condition");
      }

      handler->post_send_buffer(response_iob, response, __func__);
      _tasks.erase(i);

      goto retry;
    }
  }
}

void Shard::check_for_new_connections()
{
  /* new connections are transferred from the connection handler
     to the shard thread */
  Connection_handler *handler;

  static int connections = 1;
  while ((handler = get_new_connection()) != nullptr) {
    CPLOG(2, "Shard: processing new connection (%p) total %d",
          common::p_fmt(handler), connections);
    connections++;
    _handlers.push_back(handler);
  }
}

status_t Shard::process_configure(const protocol::Message_IO_request *msg)
{
  using namespace component;

  std::string command(msg->cmd());

  /* dynamic loading of secondary index.  loading this
     way (as oppposed to via shard configuration) will
     cause a rebuild by iterating the key space in the main
     storage engine
     e.g. AddIndex::rbtree, AddIndex::customtree
  */
  if (command.substr(0, 10) == "AddIndex::") {
    std::string index_str = command.substr(10);

    if(index_str == "VolatileTree") /* backwards compatability */
      index_str = "rbtree";

    /* derive name of component library from the AddIndex:: parameter */
    std::string dll_string = "libcomponent-index-";
    dll_string += index_str;
    dll_string += ".so";

    if (_index_map == nullptr)
      _index_map.reset(new index_map_t());

    /* create index component and put into shard index map */
    IBase *comp = load_component(dll_string.c_str(), index_factory);
    if (!comp) {
      PWRN("unable to load %s as secondary index", dll_string.c_str());
      return E_FAIL;
    }

    auto factory = make_itf_ref(static_cast<IKVIndex_factory *>(comp->query_interface(IKVIndex_factory::iid())));
    assert(factory);

    /* note, the second parameter to the factory, here "dynamic" should be passed to the constructor
       and used to control rebuilding */
    auto index = component::make_itf_ref(factory->create_dynamic(_dax_config));
    assert(index);

    /* save mapping of pool id to index instance */
    auto p = _index_map->insert(std::make_pair(msg->pool_id(), std::move(index))).first;
    factory.reset(nullptr);

    CPLOG(1, "Shard: rebuilding secondary index ...");

    status_t hr = S_OK;

    /* optionally, iterate key space for rebuilding */
    if(index && index->iterate_key_space_on_load()) {
      if ((hr = _i_kvstore->map_keys(msg->pool_id(),
                                     [p](const std::string &key)
                                     {
                                       p->second->insert(key);
                                       return 0;
                                     })) != S_OK) {

        /* alternative when map_keys method optimization is not supported on main engine */
        hr = _i_kvstore->map(msg->pool_id(),
                             [p](const void *key,
                                 const size_t key_len, const void *,  // value
                                 const size_t) {  // value_len
                               std::string k(static_cast<const char *>(key), key_len);
                               p->second->insert(k);
                               return 0;
                             });
      }
    }

    return hr;
  }
  else if (command == "RemoveIndex::") {
    try {
      _index_map->erase(msg->pool_id());
      CPLOG(1, "Shard: removed index on pool (%lx)", msg->pool_id());
    }
    catch (...) {
      return E_BAD_PARAM;
    }

    return S_OK;
  }
  else {
    PWRN("unknown configure command (%s)", command.c_str());
    return E_BAD_PARAM;
  }

  return S_OK;
}

}  // namespace mcas
