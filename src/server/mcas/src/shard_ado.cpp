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

//#define PROFILE_POST_ADO
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wold-style-cast"

/*< #included in shard.cpp */
#include <api/ado_itf.h>
#include <common/errors.h>
#include <common/exceptions.h>
#include <common/cycles.h>
#include <nupm/mcas_mod.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>
#include <libpmem.h>

#include <cstdint> /* PRIu64 */
#include <sstream>
#include <fstream>

#include "config_file.h"
#include "mcas_config.h"
#include "resource_unavailable.h"

#ifdef PROFILE
#include <gperftools/profiler.h>
#endif

//#define SHORT_CIRCUIT_ADO_HANDLING

namespace mcas
{
bool check_xpmem_kernel_module()
{
  int fd = open("/dev/xpmem", O_RDWR, 0666);
  close(fd);
  return (fd != -1);
}

status_t Shard::conditional_bootstrap_ado_process(component::IKVStore*        kvs,
                                                  Connection_handler*         handler,
                                                  component::IKVStore::pool_t pool_id,
                                                  component::IADO_proxy*&     ado,
                                                  pool_desc_t&                desc)
{
  assert(pool_id);
  assert(kvs);
  assert(handler);

  /* ADO processes are instantiated on a per-pool basis.  First
     check if an ADO process already exists.
  */
  bool bootstrap = true;

  auto proxy = _ado_pool_map.get_proxy(pool_id);

  if (proxy == nullptr) {
    if (!_ado_map.has_ado_for_pool(desc.name)) {
      /* need to launch new ADO process */
      std::vector<std::string> args;
      args.push_back("--plugins");

      std::string plugin_str;
      for (auto& plugin : _ado_plugins) {
        args.push_back(plugin);
        plugin_str += plugin + ",";
      }
      plugin_str = plugin_str.substr(0, plugin_str.size() - 1);

      for (auto& ado_param : _ado_params) {
        args.push_back("--param");
        args.push_back("'{" + ado_param.first + ":" + ado_param.second + "}'");
      }

      /* add parameter passing ipaddr */
      std::string net_addr = _net_addr;
      args.push_back("--param");
      args.push_back("'{net:" + net_addr + "," + std::to_string(_port) + "}'");

      PMAJOR("Shard: Launching with ADO path: (%s)", _ado_path.c_str());
      PMAJOR("Shard: ADO plugins: (%s)", plugin_str.c_str());

      ado = _i_ado_mgr->create(handler->auth_id(), debug_level(), kvs, pool_id,
                               desc.name,                // pool name
                               desc.size,                // pool_size,
                               desc.flags,               // const unsigned int pool_flags,
                               desc.expected_obj_count,  // const uint64_t expected_obj_count,
                               _ado_path, args, 0);

      CPLOG(2, "ADO process launched OK.");

      _ado_map.add_ado_for_pool(desc.name, ado);
    }
    else {
      ado       = _ado_map.get_ado_for_pool(desc.name);
      bootstrap = false;
    }
  }
  else {
    ado = proxy;
    ado->add_ref();
    bootstrap = false;
  }

  assert(ado);

  /* save handle to ADO instance (ref counted) */
  _ado_pool_map.add(pool_id, ado, handler);

  /* conditionally bootstrap ADO */
  if (bootstrap) {
    auto rc = ado->bootstrap_ado(desc.opened_existing);
    if (rc != S_OK) {
      return rc;
    }

    if (_backend == "mapstore" && !check_xpmem_kernel_module()) {
      PERR("mapstore with ADO requires XPMEM kernel module");
      throw Logic_exception("no XPMEM kernel module");
    }
    else if (!nupm::check_mcas_kernel_module()) {
      PERR("%s with ADO requires MCAS kernel module", _backend.c_str());
      throw Logic_exception("no MCAS kernel module");
    }

    /* exchange memory mapping information */
    {
      std::vector<::iovec> regions;
      auto                 rc = _i_kvstore->get_pool_regions(pool_id, regions);

      if (rc != S_OK) {
        PWRN("cannot get pool regions; unable to map to ADO");
        return rc;
      }

      for (auto& r : regions) {
        r.iov_len = round_up_page(r.iov_len);

        // Don't think we need this - DW
        // touch_pages(r.iov_base, r.iov_len); /* pre-fault pages */

        if (_backend == "mapstore") {
          /* uses XPMEM kernel module */
          xpmem_segid_t seg_id = ::xpmem_make(r.iov_base, r.iov_len, XPMEM_PERMIT_MODE, reinterpret_cast<void*>(0666));
          if (seg_id == -1) throw Logic_exception("xpmem_make failed unexpectedly");
          ado->send_memory_map(std::uint64_t(seg_id), r.iov_len, r.iov_base);
        }
        else {
          /* uses MCAS kernel module */
          /* generate a token for the mapping - TODO: remove exposed memory */
          uint64_t token = reinterpret_cast<uint64_t>(r.iov_base);

          nupm::revoke_memory(token); /* move any prior registration; TODO clean up when ADO goes */

          if (nupm::expose_memory(token, r.iov_base, r.iov_len) != S_OK)
            throw Logic_exception("nupm::expose_memory failed unexpectedly");

          ado->send_memory_map(token, r.iov_len, r.iov_base);
        }

        CPLOG(2, "Shard_ado: exposed region: %p %lu", r.iov_base, r.iov_len);
      }
    }

#if defined(PROFILE) && defined(PROFILE_POST_ADO)
    PLOG("Starting profiler");
    ProfilerStart("post_ado_launch.prof");
#endif
  } /* end of bootstrap */

  return S_OK;
}

void Shard::process_put_ado_request(Connection_handler* handler, const protocol::Message_put_ado_request* msg)
{
  handler->msg_recv_log(msg, __func__);
  using namespace component;

  IADO_proxy*     ado = nullptr;
  status_t        rc;
  IKVStore::key_t key_handle = 0;
  auto            locktype   = IKVStore::STORE_LOCK_WRITE;
  void*           value      = nullptr;
  size_t          value_len  = 0;
  const char*     key_ptr    = nullptr;
  bool            new_root   = false;

  const auto error_func = [&](const char* message) {
    auto response_iob = handler->allocate_send();
    auto response     = new (response_iob->base())
    protocol::Message_ado_response(response_iob->length(), E_FAIL, handler->auth_id(), msg->request_id());

    response->append_response(const_cast<char*>(message), strlen(message), 0 /* layer id */);

    response->set_status(E_INVAL);
    response_iob->set_length(response->message_size());
    handler->post_send_buffer(response_iob, response, __func__);
  };

#ifdef SHORT_CIRCUIT_ADO_HANDLING
  error_func("ADO!SC");
  return;
#endif

  if (!_i_ado_mgr) {
    error_func("ADO!NOT_ENABLED(put)");
    return;
  }

  /* ADO should already be running */
  ado = _ado_pool_map.get_proxy(msg->pool_id());
  if (!ado) throw General_exception("ADO is not running");

  if (msg->value_len() == 0) {
    error_func("ADO!ZERO_VALUE_LEN");
    return;
  }

  /* option ADO_FLAG_NO_OVERWRITE means that we don't copy
     value in if the key-value already exists */
  bool value_already_exists = false;
  if ((msg->flags & IMCAS::ADO_FLAG_NO_OVERWRITE) || (msg->flags & IMCAS::ADO_FLAG_DETACHED)) {
    std::vector<uint64_t> answer;
    std::string           key(msg->key());

    if (_i_kvstore->get_attribute(msg->pool_id(), IKVStore::Attribute::VALUE_LEN, answer, &key) !=
        IKVStore::E_KEY_NOT_FOUND) {
      /* already exists */
      value_already_exists = true;
    }
  }

  /* if ADO_FLAG_DETACHED and we need to create root value */
  if ((msg->flags & IMCAS::ADO_FLAG_DETACHED) && (msg->root_val_len > 0)) {
    value_len = msg->root_val_len;

    status_t s = _i_kvstore->lock(msg->pool_id(), msg->key(), locktype, value, value_len, key_handle, &key_ptr);
    if (s < S_OK) {
      error_func("ADO!ALREADY_LOCKED");
      return;
    }
    if (key_handle == IKVStore::KEY_NONE) throw Logic_exception("lock gave KEY_NONE");

    new_root = (s == S_OK_CREATED) ? true : false;
  }

  void*  detached_val_ptr = nullptr;
  size_t detached_val_len = 0;

  /* NOTE: this logic needs reviewing to ensure appropriate
     semantics for different flag combinations */

  if (msg->flags & IMCAS::ADO_FLAG_DETACHED) {
    auto size_to_allocate = round_up(msg->value_len(), 8);

    /* detached value request, means the value is put but is not assigned to key root ptr  */
    rc = _i_kvstore->allocate_pool_memory(msg->pool_id(), size_to_allocate, 8, /* alignment */
                                          detached_val_ptr);
    if (rc != S_OK) {
      PWRN("allocate_pool_memory for detached value failed (len=%lu, rc=%d)", size_to_allocate, rc);
      error_func("ADO!OUT_OF_MEMORY");
      return;
    }
    detached_val_len = size_to_allocate;
    memcpy(detached_val_ptr, msg->value(), msg->value_len());

    CPLOG(2, "Shard_ado: allocated detached memory (%p,%lu)", detached_val_ptr, detached_val_len);
  }
  else if (value_already_exists && (msg->flags & IMCAS::ADO_FLAG_NO_OVERWRITE)) {
    /* do nothing, drop through */
  }
  else {
    /* write value passed with invocation message */
    rc = _i_kvstore->put(msg->pool_id(), msg->key(), msg->value(), msg->value_len());
    if (rc != S_OK) throw Logic_exception("put_ado_invoke: put failed");
  }

  /*------------------------------------------------------------------
    Lock kv pair if needed, then create a work request and send to
    the ADO process via UIPC
  */
  if (!value) { /* now take the lock if not already locked */
    if (_i_kvstore->lock(msg->pool_id(), msg->key(), locktype, value, value_len, key_handle, &key_ptr) != S_OK) {
      error_func("ADO!ALREADY_LOCKED(key)");
      return;
    }
    if (key_handle == IKVStore::KEY_NONE) throw Logic_exception("lock gave KEY_NONE");
  }

  CPLOG(2, "Shard_ado: locked KV pair (value=%p, value_len=%lu)", value, value_len);

  /* register outstanding work */
  auto wr = _wr_allocator.allocate();
  *wr     = {handler, msg->pool_id(), key_handle, key_ptr, msg->get_key_len(), locktype, msg->request_id(), msg->flags};

  auto wr_key = reinterpret_cast<work_request_key_t>(wr); /* pointer to uint64_t */
  _outstanding_work.insert(wr_key);

  wmb();

  /* now send the work request */
  ado->send_work_request(wr_key, key_ptr, msg->get_key_len(), value, value_len, detached_val_ptr, detached_val_len,
                         msg->request(), msg->request_len(), new_root);

  CPLOG(2, "Shard_ado: sent work request (len=%lu, key=%lx)", msg->request_len(), wr_key);
}

void Shard::process_ado_request(Connection_handler* handler, const protocol::Message_ado_request* msg)
{
  try {
    // PLOG("%s: enter", __func__);
    //    PNOTICE("invoke ADO recv (rid=%lu)", msg->request_id());

    handler->msg_recv_log(msg, __func__);
    using namespace component;

    IADO_proxy* ado;

    const auto error_func = [&](status_t status, const char* message) {
      auto response_iob = handler->allocate_send();
      auto response     = new (response_iob->base())
      protocol::Message_ado_response(response_iob->length(), status, handler->auth_id(), msg->request_id());
      response->append_response(const_cast<char*>(message), strlen(message), 0);
      response_iob->set_length(response->message_size());
      PLOG("%s server error message %s", __func__, message);
      handler->post_send_buffer(response_iob, response, __func__);
    };

    CPLOG(2, "Shard_ado: process_ado_request");

#ifdef SHORT_CIRCUIT_ADO_HANDLING
    error_func(E_INVAL, "ADO!SC");
    PLOG("%s server error short circuit", __func__);
    return;
#endif

    if (!ado_enabled()) {
      std::ostringstream o;
      o << "ADO!NOT_ENABLED mgr '" << (_i_ado_mgr ? "present" : "missing") << "' load count " << _ado_plugins.size();
      error_func(E_INVAL, o.str().c_str());
      PLOG("%s server error ADO!NOT_ENABLED", __func__);
      return;
    }

    if (msg->flags & IMCAS::ADO_FLAG_DETACHED) { /* not valid for plain invoke_ado */
      error_func(E_INVAL, "ADO!INVALID_ARGS");
      PLOG("%s server error ADO!INVALID_ARGS circuit", __func__);
      return;
    }

    void*  value     = nullptr;
    size_t value_len = msg->ondemand_val_len;

    /* handle ADO_FLAG_CREATE_ONLY - no invocation to ADO is made */
    if (msg->flags & IMCAS::ADO_FLAG_CREATE_ONLY) {
      std::vector<uint64_t> answer;
      std::string           key(msg->key());
      if (_i_kvstore->get_attribute(msg->pool_id(), IKVStore::Attribute::VALUE_LEN, answer, &key) !=
          IKVStore::E_KEY_NOT_FOUND) {
        error_func(E_ALREADY_EXISTS, "ADO!ALREADY_EXISTS");
        PLOG("%s server error ADO!ALREADY_EXISTS", __func__);
        if (debug_level() > 1) PWRN("process_ado_request: ADO_FLAG_CREATE_ONLY, key already exists");
        return;
      }

      IKVStore::key_t key_handle;
      auto locktype = (msg->flags & IMCAS::ADO_FLAG_READ_ONLY) ? IKVStore::STORE_LOCK_READ : IKVStore::STORE_LOCK_WRITE;

      status_t s = _i_kvstore->lock(msg->pool_id(), msg->key(), locktype, value, value_len, key_handle);
      if (s < S_OK) {
        std::stringstream ss;
        ss << "ADO!ALREADY_LOCKED(" << msg->key() << ")";
        error_func(E_LOCKED, ss.str().c_str());
        if (debug_level() > 1) PWRN("process_ado_request: key already locked (ADO_FLAG_CREATE_ONLY)");
        PLOG("%s server error lock", __func__);
        return;
      }

      /* zero memory */
      pmem_memset(value, 0, value_len, 0);

      /* unlock key-value pair because we are not invoking ADO */
      if (_i_kvstore->unlock(msg->pool_id(), key_handle) != S_OK)
        throw Logic_exception("unable to unlock after lock");

      /* copy value address into response */
      auto response_iob = handler->allocate_send();
      auto response     = new (response_iob->base())
        protocol::Message_ado_response(response_iob->length(), S_OK, handler->auth_id(), msg->request_id());

      response->append_response(&value, sizeof(value), 0 /* layer id */);
      response->set_status(S_OK);
      response_iob->set_length(response->message_size());
      PLOG("%s server response count %zu", __func__, response->get_response_count());
      handler->post_send_buffer(response_iob, response, __func__);

      return;  // end of ADO_FLAG_CREATE_ONLY condition
    }

    /*  ADO should already be running */
    ado = _ado_pool_map.get_proxy(msg->pool_id());
    assert(ado);

    /* get key-value pair */
    IKVStore::key_t key_handle = IKVStore::KEY_NONE;
    const char*     key_ptr    = nullptr;
    auto            locktype   = IKVStore::STORE_LOCK_NONE;
    status_t        s          = S_OK;

    /* if this is associated with a key-value pair, we have to lock */
    if (msg->key_len > 0) {
      locktype = (msg->flags & IMCAS::ADO_FLAG_READ_ONLY) ? IKVStore::STORE_LOCK_READ : IKVStore::STORE_LOCK_WRITE;
      s        = _i_kvstore->lock(msg->pool_id(), msg->key(), locktype, value, value_len, key_handle, &key_ptr);

      if (s < S_OK) {
        std::stringstream ss;
        ss << "ADO!ALREADY_LOCKED(" << msg->key() << ")";
        error_func(E_LOCKED, ss.str().c_str());
        if (debug_level() > 1) PWRN("process_ado_request: key already locked");
        return;
      }

      if (key_handle == IKVStore::KEY_NONE) throw Logic_exception("lock gave KEY_NONE");

      CPLOG(2, "Shard_ado: locked KV pair (value=%p, value_len=%lu)", value, value_len);
    }

    /* register outstanding work */
    auto wr = _wr_allocator.allocate();
    *wr     = {handler, msg->pool_id(), key_handle, key_ptr, msg->get_key_len(), locktype, msg->request_id(), msg->flags};

    auto wr_key = reinterpret_cast<work_request_key_t>(wr); /* pointer to uint64_t */
    _outstanding_work.insert(wr_key);                       /* save request by index on key-handle */

    /* now send the work request */
    ado->send_work_request(wr_key, key_ptr, msg->get_key_len(), value, value_len, nullptr, /* no payload */
                           0, msg->request(), msg->request_len(), (s == S_OK_CREATED));

    CPLOG(2, "Shard_ado: sent work request (len=%lu, key=%lx, key_ptr=%p)",
          msg->request_len(), wr_key, static_cast<const void*>(key_ptr));

    /* for "asynchronous" calls we don't send a message
       for "synchronous call" we don't send a response to the client
       until the work completion has been picked up.  Of course this
       gives synchronous semantics on the client side.  We may need to
       extend this to asynchronous semantics for longer ADO
       operations */
  }
  catch (const Exception& e) {
    PLOG("%s: Exception %s enter", __func__, e.cause());
  }
  catch (const std::exception& e) {
    PLOG("%s: exception %s", __func__, e.what());
  }
}


void Shard::close_all_ado()
{
  PLOG("Shard: signalling ADOs to shutdown");
  for (auto iter = _ado_map.begin(); iter != _ado_map.end(); iter++) {
    component::IADO_proxy* ado = iter->second;
    ado->shutdown();
    delete ado;
  }
}

/**
 * Handle messages coming back from the ADO process.
 *
 */
void Shard::process_messages_from_ado()
{
  using namespace component;

  /* iterate ADO process proxies */
  auto iter = _ado_pool_map.begin();
  while (iter != _ado_pool_map.end()) {
    IADO_proxy*         ado     = std::get<0>(iter->second);
    Connection_handler* handler = std::get<1>(iter->second);
    iter++; /* OK to do this now */

    assert(ado);

    work_request_key_t                    request_key     = 0;
    status_t                              response_status = E_FAIL;
    IADO_plugin::response_buffer_vector_t response_buffers;

    /*---------------------*/
    /* ADO work completion */
    /*---------------------*/
    while (ado->check_work_completions(request_key, response_status, response_buffers)) {
      if (response_status > S_USER0 || response_status < E_ERROR_BASE) response_status = E_FAIL;

      CPLOG(2, "Shard_ado: check_work_completions(response_status=%d, response_count=%lu",
            response_status, response_buffers.size());

      auto work_item = _outstanding_work.find(request_key);
      if (work_item == _outstanding_work.end())
        throw General_exception("Shard_ado: bad work request key from ADO (0x%" PRIx64 ")", request_key);

      auto request_record = request_key_to_record(request_key);
      handler             = request_record->handler;
      assert(handler);

      if (debug_level() > 2) {
        for (const auto &r : response_buffers) {
          PLOG("Shard_ado: returning response (%p,%lu,%s)", r.ptr, r.len, r.is_pool() ? "pool" : "non-pool");
        }
      }

      _outstanding_work.erase(work_item);

      /* unlock the KV pair */
      if (request_record->key_handle != IKVStore::KEY_NONE) {

        CPLOG(2, "Shard_ado: start to unlock KV pair key=(%.*s)",
              int(request_record->key_len), request_record->key_ptr);

        if (_i_kvstore->unlock(request_record->pool, request_record->key_handle) != S_OK)
          throw Logic_exception("Shard_ado: unlock for KV after ADO work completion failed");

        CPLOG(2, "Shard_ado: unlocked KV pair (pool=%lx, key_handle=%p)", request_record->pool,
              static_cast<const void*>(request_record->key_handle));
      }

      /* unlock deferred locks, e.g., resulting from table operation create */
      {
        std::vector<IKVStore::key_t> keys_to_unlock;
        ado->get_deferred_unlocks(request_key, keys_to_unlock);
        for (auto k : keys_to_unlock) {
          if (_i_kvstore->unlock(request_record->pool, k) != S_OK) throw Logic_exception("deferred unlock failed");

          CPLOG(2, "Shard_ado: deferred unlock (%p)", static_cast<void*>(k));
        }
      }

      /* handle erasing target */
      if (response_status == IADO_plugin::S_ERASE_TARGET) {
        status_t s =
          _i_kvstore->erase(request_record->pool, std::string(request_record->key_ptr, request_record->key_len));
        if (s != S_OK)
          PWRN("Shard_ado: request to erase target failed unexpectedly (key=%s,rc=%d)", request_record->key_ptr, s);
        response_status = s;
      }

      /* for async, save failed requests */
      if (request_record->is_async()) {
        /* if the ADO operation response is bad, save it for
           later, otherwise don't do anything */
        if (response_status < S_OK) {
          if (debug_level() > 2) PWRN("Shard_ado: saving ADO completion failure");
          _failed_async_requests.push_back(request_record);
        }
        else {
          CPLOG(2, "Shard_ado: async ADO completion OK!");
        }
      }
      /* for sync, give response, unless the client is disconnected */
      else if (handler->client_connected()) {
        auto iob = handler->allocate_send();

        assert(iob->base());
        auto response_msg = new (iob->base()) protocol::Message_ado_response(iob->length(),
                                                                             response_status,
                                                                             handler->auth_id(),
                                                                             request_record->request_id);

        /* TODO: for the moment copy pool buffers in, we should
           be able to do zero copy though.
        */
        size_t appended_buffer_size = 0;

        for (auto& rb : response_buffers) {
          assert(rb.ptr);
          try
          {
            response_msg->append_response(rb.ptr, boost::numeric_cast<uint32_t>(rb.len), rb.layer_id);
          } catch ( const std::exception &e ) { PLOG("%s: exception building response: %s", __func__, e.what()); throw; }
          appended_buffer_size += rb.len;
        }

        assert(iob);
        assert(response_msg);
        iob->set_length(response_msg->message_size());

        CPLOG(2,"Shard_ado: posting ADO response");

        handler->post_send_buffer(iob, response_msg, __func__);
      }

      /* clean up response buffers that were temporarily allocated from the pool
       */
      for (auto& rb : response_buffers) {
        if (rb.is_pool_to_free()) {
          _i_kvstore->free_pool_memory(request_record->pool, rb.ptr, rb.len);
        }
      }

      /* release request record */
      _wr_allocator.free_wr(request_record);

    } /* end of while ado->check_work_completions */

    uint64_t             work_id = 0; /* maps to record of pool, key handle, lock type, request id etc. */
    ADO_op               op      = ADO_op::UNDEFINED;
    std::string          key, key_expression;
    size_t               value_len      = 0;
    size_t               align_or_flags = 0;
    void*                addr           = nullptr;
    offset_t             begin_pos      = 0;
    int                  find_type      = 0;
    uint32_t             max_comp       = 0;
    uint64_t             options        = 0;
    common::epoch_time_t t_begin = 0, t_end = 0;
    component::IKVStore::pool_iterator_t iterator   = nullptr;
    component::IKVStore::key_t           key_handle = nullptr;
    Buffer_header*                       buffer;

    /* process callbacks from ADO */
    while (ado->recv_callback_buffer(buffer) == S_OK) {
      /*-------------------------*/
      /* handle TABLE OPERATIONS */
      /*-------------------------*/
      if (ado->check_table_ops(buffer, work_id, op, key, value_len, align_or_flags, addr)) {
        switch (op) {
        case ADO_op::CREATE: {
          std::vector<uint64_t> val;

          status_t s = _i_kvstore->get_attribute(ado->pool_id(), IKVStore::VALUE_LEN, val, &key);

          if (s != IKVStore::E_KEY_NOT_FOUND) {
            if (debug_level() > 3)
              PWRN("Shard_ado: table op CREATE, key-value pair already "
                   "exists");

            if (align_or_flags & IKVStore::FLAGS_CREATE_ONLY) {
              ado->send_table_op_response(E_ALREADY_EXISTS, nullptr, 0, nullptr);
              break;
            }
          }

          goto open; /* stop compiler complaining about flow through*/
        }
        case ADO_op::OPEN:
          open : {
            CPLOG(2, "Shard_ado: received table op key create/open (%s, value_len=%lu)",
                  key.c_str(), value_len);

            IKVStore::key_t key_handle;
            void*           value = nullptr;
            const char*     key_ptr;

            bool invoke_completion_unlock = !(align_or_flags & IADO_plugin::FLAGS_ADO_LIFETIME_UNLOCK);

            status_t rc = _i_kvstore->lock(ado->pool_id(), key, IKVStore::STORE_LOCK_WRITE, value, value_len,
                                           key_handle, &key_ptr);

            if (rc < S_OK || key_handle == nullptr) { /* to fix, store should return error code */
              CPLOG(2, "Shard_ado: lock on key (%s, value_len=%lu) failed rc=%d", key.c_str(), value_len, rc);

              ado->send_table_op_response(rc);
            }
            else {
              CPLOG(2, "Shard_ado: locked KV pair (keyhandle=%p, value=%p,len=%lu) invoke_completion_unlock=%d",
                    static_cast<void*>(key_handle), value, value_len, invoke_completion_unlock);

              add_index_key(ado->pool_id(), key);

              /* auto-unlock means we add a deferred unlock that happens after
                 the ado invocation (identified by work_id) has completed. */
              if (align_or_flags & IADO_plugin::FLAGS_NO_IMPLICIT_UNLOCK) {
                CPLOG(2, "Shard_ado: locked (%s) without implicit unlock", key.c_str());
              }
              else if (invoke_completion_unlock) { /* unlock on ADO invoke
                                                      completion */
                if (work_id == 0) {
                  ado->send_table_op_response(E_INVAL);
                }
                else {
                  try {
                    ado->add_deferred_unlock(work_id, key_handle);
                  }
                  catch (const std::range_error&) {
                    PWRN("Shard_ado: too many locks");
                    ado->send_table_op_response(E_MAX_REACHED);
                  }
                }
              }
              else { /* unlock at ADO process shutdown */
                ado->add_life_unlock(key_handle);
              }

              assert(reinterpret_cast<uint64_t>(addr) <= 1);

              ado->send_table_op_response(S_OK, static_cast<void*>(value), value_len, key_ptr, key_handle);
            }
          } break;
        case ADO_op::ERASE: {
          CPLOG(2, "Shard_ado: received table op erase");
          ado->send_table_op_response(_i_kvstore->erase(ado->pool_id(), key));
          break;
        }
        case ADO_op::VALUE_RESIZE: /* resize only allowed on current work
                                      invocation target */
          {
            CPLOG(2, "Shard_ado: received table op resize value (work_id=%p)", reinterpret_cast<const void*>(work_id));

            /* for resize, we need unlock, resize, and then relock */
            auto work_item = _outstanding_work.find(work_id);

            if (work_item == _outstanding_work.end()) {
              ado->send_table_op_response(E_INVAL);
              break;
            }

            /* use the work id to get the key handle */
            work_request_t* wr      = request_key_to_record(work_id);
            const char*     key_ptr = nullptr;
            status_t        rc;

            if (!wr) throw Logic_exception("unable to get request from work_id");

            if ((rc = _i_kvstore->unlock(ado->pool_id(), wr->key_handle)) != S_OK) {
              ado->send_table_op_response(rc);
              break;
            }

            CPLOG(2, "Shard_ado: table op resize, unlocked");

            /* perform resize */
            void*  new_value      = nullptr;
            size_t new_value_len  = 0;
            auto   old_key_handle = wr->key_handle;
            rc                    = _i_kvstore->resize_value(ado->pool_id(), key, value_len, align_or_flags);

            if (_i_kvstore->lock(ado->pool_id(), key, IKVStore::STORE_LOCK_WRITE, new_value, new_value_len,
                                 wr->key_handle /* update key handle in record */, &key_ptr) != S_OK)
              throw Logic_exception("ADO OP_RESIZE request failed to relock");

            /* update deferred locks */
            if (ado->update_deferred_unlock(work_id, wr->key_handle) != S_OK) {
              if (ado->remove_life_unlock(old_key_handle) == S_OK) ado->add_life_unlock(wr->key_handle);
            }

            ado->send_table_op_response(rc, new_value, new_value_len, key_ptr);
            break;
          }
        case ADO_op::ALLOCATE_POOL_MEMORY: {

          status_t rc;
          assert(work_id == 0); /* work request is not needed */

          CPLOG(2, "Shard_ado: calling allocate_pool_memory align_or_flags=%lu size=%lu",
                align_or_flags, value_len);

          /* provide memory PM and DRAM summary */
          if (debug_level() > 0)
          {
            uint64_t     expected_obj_count = 0;
            size_t       pool_size          = 0;
            unsigned int pool_flags         = 0;

            assert(handler);
            handler->pool_manager().get_pool_info(ado->pool_id(), expected_obj_count, pool_size, pool_flags);

            std::vector<uint64_t> pu_attr;
            if(_i_kvstore->get_attribute(ado->pool_id(),
                                         IKVStore::Attribute::PERCENT_USED, pu_attr) == S_OK) {
              PLOG("Shard_ado: port(%u) '#memory' pool (%s) memory %lu%% used (%luMiB/%luMiB)",
                   _port,
                   ado->pool_name().c_str(),
                   pu_attr[0],
                   REDUCE_MB(pu_attr[0] == 0 ? 0 : pu_attr[0] * pool_size / 100),
                   REDUCE_MB(pool_size));
            }

            PLOG("Shard_ado: port(%u) '#memory' %s", _port, common::get_DRAM_usage().c_str());
          }

          void* out_addr = nullptr;
          rc             = _i_kvstore->allocate_pool_memory(ado->pool_id(), value_len, align_or_flags, out_addr);

          CPLOG(2, "Shard ado: allocated memory at %p from pool_id (%lx)", out_addr, ado->pool_id());
          CPLOG(2, "Shard_ado: allocate_pool_memory align_or_flags=%lu rc=%d addr=%p",
                align_or_flags, rc, out_addr);

          ado->send_table_op_response(rc, out_addr);
          break;
        }
        case ADO_op::FREE_POOL_MEMORY: {
          assert(work_id == 0); /* work request is not needed */

          if (value_len == 0) {
            ado->send_table_op_response(E_INVAL);
            break;
          }

          status_t rc = _i_kvstore->free_pool_memory(ado->pool_id(), addr, value_len);
          CPLOG(2, "Shard_ado : allocate_pool_memory free rc=%d", rc);

          if (rc != S_OK) PWRN("Shard_ado: Table operation OP_FREE failed");

          ado->send_table_op_response(rc);
          break;
        }
        default:
          throw Logic_exception("unknown table op code");
        }
        // end of if(check_table_ops..
      }
      /*--------------------------*/
      /* handle POOL INFO request */
      /*--------------------------*/
      else if (ado->check_pool_info_op(buffer)) {
        using namespace rapidjson;

        uint64_t     expected_obj_count = 0;
        size_t       pool_size          = 0;
        unsigned int pool_flags         = 0;

        assert(handler);
        handler->pool_manager().get_pool_info(ado->pool_id(), expected_obj_count, pool_size, pool_flags);

        std::vector<uint64_t> mt_attr;
        if (_i_kvstore->get_attribute(ado->pool_id(), IKVStore::Attribute::MEMORY_TYPE, mt_attr) != S_OK)
          throw Logic_exception("get_attributes failed on storage engine (Attribute::MEMORY_TYPE)");

        std::vector<uint64_t> pu_attr;
        bool pu_valid = (_i_kvstore->get_attribute(ado->pool_id(), IKVStore::Attribute::PERCENT_USED, pu_attr) == S_OK);

        try {
          Document doc;
          doc.SetObject();

          Value pool_size_v(pool_size);
          doc.AddMember("pool_size", pool_size_v, doc.GetAllocator());
          Value memory_type(mt_attr[0]);
          doc.AddMember("memory_type", memory_type, doc.GetAllocator());

          if(pu_valid) {
            Value percent_used(pu_attr[0]);
            doc.AddMember("percent_used", percent_used, doc.GetAllocator());
          }

          Value expected_obj_count_v(expected_obj_count);
          doc.AddMember("expected_obj_count", expected_obj_count_v, doc.GetAllocator());
          Value pool_flags_v(pool_flags);
          doc.AddMember("pool_flags", pool_flags_v, doc.GetAllocator());
          std::vector<uint64_t> v64;
          if (_i_kvstore->get_attribute(ado->pool_id(), IKVStore::Attribute::COUNT, v64) == S_OK) {
            Value obj_count_v(v64[0]);
            doc.AddMember("current_object_count", obj_count_v, doc.GetAllocator());
          }
          std::stringstream      ss;
          OStreamWrapper         osw(ss);
          Writer<OStreamWrapper> writer(osw);
          doc.Accept(writer);
          ado->send_pool_info_response(S_OK, ss.str());
        }
        catch (...) {
          throw Logic_exception("pool info JSON creation failed");
        }
      }
      else if (ado->check_op_event_response(buffer, op)) {
        switch (op) {
        case ADO_op::POOL_DELETE: {
          /* close pool, then delete */
          if ((_i_kvstore->close_pool(ado->pool_id()) != S_OK) || (_i_kvstore->delete_pool(ado->pool_name()) != S_OK))
            throw Logic_exception("unable to delete pool after POOL DELETE op event");

          CPLOG(2, "POOL DELETE op event completion");

          break;
        }
        case ADO_op::CLOSE: {
          PWRN("ignoring CLOSE from ADO");
          break;
        }
        default:
          throw Logic_exception("unknown op event (%d)", op);
        }
      }
      else if (ado->check_iterate(buffer, t_begin, t_end, iterator)) {
        component::IKVStore::pool_reference_t ref;
        if (!iterator) {
          iterator = _i_kvstore->open_pool_iterator(ado->pool_id());
        }

        if (!iterator) { /* still no iterator, component doesn't support */
          ado->send_iterate_response(E_NOT_IMPL, iterator, ref);
        }
        else {
          status_t rc;
          bool     time_match = false;
          do {
            rc = _i_kvstore->deref_pool_iterator(ado->pool_id(), iterator, t_begin, /* time constraints */
                                                 t_end, ref, time_match, true);

            if (rc == E_OUT_OF_BOUNDS) {
              _i_kvstore->close_pool_iterator(ado->pool_id(), iterator);
              break;
            }
          } while (!time_match && rc != E_INVAL); /* TODO: limit number of iterations */
          if (rc == E_INVAL) PWRN("Shard_ado: deref_pool_iterator returned E_INVAL");

          CPLOG(2, "Shard_ado: iterator timestamp (%lu seconds)", ref.timestamp.seconds());

          ado->send_iterate_response(rc, iterator, ref);
        }
      }
      else if (ado->check_vector_ops(buffer, t_begin, t_end)) {
        /* WARNING: this could block the shard thread. we may
           neeed to make it a "task" - but we can't do this
           without a map iterator that can be restarted.
        */
        /* vector operation, collect all key-value pointers */
        status_t                      rc;
        size_t                        count  = 0;
        void*                         buffer = nullptr;
        IADO_plugin::Reference_vector v;

        /* allocate memory from pool for the vector */
        if (t_begin.is_defined() || t_end.is_defined()) {
          /* we have to map first to get count */
          rc = _i_kvstore->map(
                               ado->pool_id(),
                               [&count](const void*, const size_t, const void*, const size_t, const common::tsc_time_t) -> int {
                                 count++;
                                 return 0;
                               },
                               t_begin, t_end);
          CPLOG(2, "map time constraints: count=%lu", count);
        }
        else {
          count = _i_kvstore->count(ado->pool_id());
        }

        auto buffer_size = IADO_plugin::Reference_vector::size_required(count);
        rc               = _i_kvstore->allocate_pool_memory(ado->pool_id(), buffer_size, 0, buffer);

        if (rc != S_OK) {
          ado->send_vector_response(rc, IADO_plugin::Reference_vector());
        }
        else {
          /* populate vector */
          IADO_plugin::kv_reference_t* ptr   = static_cast<IADO_plugin::kv_reference_t*>(buffer);
          size_t                       check = 0;

          if (t_begin.is_defined() && t_end.is_defined()) {
            rc = _i_kvstore->map(ado->pool_id(),
                                 [count, &check, &ptr](const void* key, const size_t key_len, const void* value,
                                                       const size_t value_len) -> int {
                                   assert(key);
                                   assert(key_len);
                                   assert(value);
                                   assert(value_len);
                                   if (check > count) return -1;
                                   ptr->key       = const_cast<void*>(key);
                                   ptr->key_len   = key_len;
                                   ptr->value     = const_cast<void*>(value);
                                   ptr->value_len = value_len;
                                   ptr++;
                                   check++;
                                   return 0;
                                 });
          }
          else {
            rc = _i_kvstore->map(
                                 ado->pool_id(),
                                 [count, &check, &ptr](const void* key, const size_t key_len, const void* value, const size_t value_len,
                                                       const common::tsc_time_t  // timestamp
                                                       ) -> int {
                                   assert(key);
                                   assert(key_len);
                                   assert(value);
                                   assert(value_len);
                                   if (check > count) return -1;
                                   ptr->key       = const_cast<void*>(key);
                                   ptr->key_len   = key_len;
                                   ptr->value     = const_cast<void*>(value);
                                   ptr->value_len = value_len;
                                   ptr++;
                                   check++;
                                   return 0;
                                 },
                                 t_begin, t_end);
          }

          ado->send_vector_response(rc, IADO_plugin::Reference_vector(count, buffer, buffer_size));
        }
      }
      else if (ado->check_index_ops(buffer, key_expression, begin_pos, find_type, max_comp)) {
        status_t rc;
        auto     i_kvindex = lookup_index(ado->pool_id());

        if (!i_kvindex) {
          PWRN("ADO index operation: no index enabled");
          ado->send_find_index_response(E_NO_INDEX, 0, "noindex");
        }
        else {
          std::string matched_key;
          offset_t    matched_pos = offset_t(-1);

          rc = i_kvindex->find(key_expression, begin_pos, IKVIndex::convert_find_type(find_type), matched_pos,
                               matched_key, MAX_INDEX_COMPARISONS);

          ado->send_find_index_response(rc, matched_pos, matched_key);
        }
      }
      else if (ado->check_unlock_request(buffer, work_id, key_handle)) {

        CPLOG(2, "ADO callback: unlock request (work_id=%lx, handle=%p", work_id, static_cast<const void*>(key_handle));

        /* unlock should fail if implicit unlock exists, i.e.  it
           should only be performed on locks taken via
           FLAGS_NO_IMPLICIT_UNLOCK */
        if (key_handle == nullptr || ado->check_for_implicit_unlock(work_id, key_handle)) {
          ado->send_unlock_response(E_INVAL);
        }
        else {
          ado->send_unlock_response(_i_kvstore->unlock(ado->pool_id(), key_handle));
        }
      }
      else if (ado->check_configure_request(buffer, options)) {
        /* ADO can change reference count on ADO process from shard */
        if (options & IADO_plugin::CONFIG_SHARD_INC_REF) ado->add_ref();
        if (options & IADO_plugin::CONFIG_SHARD_DEC_REF) ado->release_ref();

        ado->send_configure_response(S_OK);
      }
      else {
        throw Logic_exception("Shard_ado: bad op request from ADO plugin");
      }

      /* release buffer */
      ado->free_callback_buffer(buffer);
    }
  }
}

}  // namespace mcas

#pragma GCC diagnostic pop
