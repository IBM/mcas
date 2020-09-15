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

#include "ado.h"
#include "ado_proto.h"
#include "ado_ipc_proto.h"
#include "ado_proto_buffer.h"
#include "resource_unavailable.h"

#include <common/logging.h>
#include <common/exceptions.h>
#include <common/utils.h>
#include <common/dump_utils.h>
#include <api/interfaces.h>
#include <nupm/mcas_mod.h>
#include <atomic>
#include <boost/program_options.hpp>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <queue>
#include <map>
#include <sched.h>
#include <cstdio>
#include <cstdlib>
#include <sys/resource.h>
#include <sys/wait.h>
#include <sys/prctl.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <nupm/mcas_mod.h>
#include <xpmem.h>

#ifdef PROFILE
#include <gperftools/profiler.h>
#endif

using namespace component;

/* Globals */
static std::vector<std::pair<void*, size_t>> g_shared_memory_mappings;

/* Helpers */

bool check_xpmem_kernel_module()
{
  int fd = open("/dev/xpmem", O_RDWR, 0666);
  close(fd);
  return (fd != -1);
}

/**
 * Class to manage plugins
 */
class ADO_plugin_mgr
{
public:
  explicit ADO_plugin_mgr(const std::vector<std::string>& plugin_vector,
                          IADO_plugin::Callback_table cb_table)
    : _i_plugins{}
  {
    for(const auto& ppath : plugin_vector) {
      _i_plugins.push_back(make_itf_ref(static_cast<IADO_plugin*>(load_component(ppath.c_str(),
                                                                                 interface::ado_plugin))));
      if( ! _i_plugins.back() )
        throw General_exception("unable to load ADO plugin (%s)", ppath.c_str());

      _i_plugins.back()->register_callbacks(cb_table);
      PLOG("ADO: plugin loaded OK! (%s)", ppath.c_str());
    }
  }

  virtual ~ADO_plugin_mgr() {
    PLOG("ADO: plugin mgr dtor.");
  }

  void shutdown() {
    for(const auto &i: _i_plugins) i->shutdown();
  }

  status_t register_mapped_memory(void *shard_vaddr,
                                  void *local_vaddr,
                                  size_t len) {
    status_t s = S_OK;
    for(const auto &i: _i_plugins) {
      s |= i->register_mapped_memory(shard_vaddr,
                                     local_vaddr,
                                     len);
    }
    return s;
  }

  status_t do_work(const uint64_t work_key,
                   const char * key,
                   size_t key_len,
                   IADO_plugin::value_space_t& values,
                   const void *in_work_request,
                   const size_t in_work_request_len,
                   const bool new_root,
                   IADO_plugin::response_buffer_vector_t& response_buffers) {
    status_t s = S_OK;
    for(const auto &i: _i_plugins) {
      s |= i->do_work(work_key, key, key_len, values,
                      in_work_request,
                      in_work_request_len,
                      new_root,
                      response_buffers);
    }

    return s;
  }

  void launch_event(const uint64_t auth_id,
                    const std::string& pool_name,
                    const size_t pool_size,
                    const unsigned int pool_flags,
                    const unsigned int memory_type,
                    const size_t expected_obj_count,
                    const std::vector<std::string>& params) {

    for(const auto &i: _i_plugins)
      i->launch_event(auth_id,
                      pool_name,
                      pool_size,
                      pool_flags,
                      memory_type,
                      expected_obj_count,
                      params);
  }

  void notify_op_event(ADO_op op) {
    for(const auto &i: _i_plugins)
      i->notify_op_event(op);
  }

  void send_cluster_event(const std::string& sender,
                          const std::string& type,
                          const std::string& message) {
    for(const auto &i: _i_plugins)
      i->cluster_event(sender, type, message);
  }

private:
  std::vector<component::Itf_ref<IADO_plugin>> _i_plugins;
};



/**
 * Main entry point
 *
 * @param argc
 * @param argv
 *
 * @return
 */
int main(int argc, char* argv[])
{

  try
    {
      std::string plugins, channel_id;
      unsigned debug_level;
      std::string cpu_mask;
      std::vector<std::string> ado_params;
      bool use_log = false;

      try {
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()
          ("help", "Print help")
          ("plugins", po::value<std::string>(&plugins)->required(), "ADO plugins")
          ("channel_id", po::value<std::string>(&channel_id)->required(), "Channel (prefix) identifier")
          ("debug", po::value<unsigned>(&debug_level)->default_value(0), "Debug level")
          ("cpumask", po::value<std::string>(&cpu_mask), "Cores to restrict threads to (string form)")
          ("param", po::value<std::vector<std::string>>(&ado_params), "Plugin parameters")
          ("log", "Redirect output to ado.log")
          ;

        po::variables_map vm;
        try {
          po::store(po::parse_command_line(argc, argv, desc), vm);
          if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
          }
          use_log = vm.count("log") > 0;
          po::notify(vm);
        }
        catch (const po::error &e) {
          std::cerr << e.what() << std::endl;
          std::cerr << desc << std::endl;
          return -1;
        }
      }
      catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
      }

      /* configure process to die with parent */
      if(prctl(PR_SET_PDEATHSIG, SIGKILL))
        throw General_exception("prctl failed");

      /* capture output from child process */
      if(use_log) {
        int fd = ::open("ado.log", O_CREAT|O_WRONLY, 0600);
        ::dup2(fd, 1);
        ::dup2(fd, 2);
        ::close(fd);
        PMAJOR("ADO: redirected output to ado.log ...");
      }
      PMAJOR("ADO: launched");

      ADO_protocol_builder ipc(debug_level, channel_id, ADO_protocol_builder::Role::ACCEPT);
      PMAJOR("ADO: listening");

      /* Callback functions */

      auto ipc_create_key =
        [&ipc] (const uint64_t work_request_id,
                const std::string& key_name,
                const size_t value_size,
                const uint64_t flags,
                void*& out_value_addr,
                const char ** out_key_ptr,
                component::IKVStore::key_t * out_key_handle) -> status_t
        {
          status_t rc = S_OK;
          ipc.send_table_op_create(work_request_id, key_name, value_size, flags);
          ipc.recv_table_op_response(rc, out_value_addr, nullptr /* value len */, out_key_ptr, out_key_handle);
          return rc;
        };

      auto ipc_open_key =
        [&ipc] (const uint64_t work_request_id,
                const std::string& key_name,
                const uint64_t flags,
                void*& out_value_addr,
                size_t& out_value_len,
                const char** out_key_ptr,
                component::IKVStore::key_t * out_key_handle) -> status_t
        {
          status_t rc = S_OK;
          ipc.send_table_op_open(work_request_id, key_name, out_value_len, flags);
          ipc.recv_table_op_response(rc, out_value_addr, &out_value_len, out_key_ptr, out_key_handle);
          return rc;
        };

      auto ipc_erase_key =
        [&ipc] (const std::string& key_name) -> status_t
        {
          status_t rc = S_OK;
          void* na;
          ipc.send_table_op_erase(key_name);
          ipc.recv_table_op_response(rc, na);
          return rc;
        };

      auto ipc_resize_value =
        [&ipc] (const uint64_t work_request_id,
                const std::string& key_name,
                const size_t new_value_size,
                void*& out_new_value_addr) -> status_t
        {
          status_t rc = S_OK;
          ipc.send_table_op_resize(work_request_id, key_name, new_value_size);
          ipc.recv_table_op_response(rc, out_new_value_addr);
          return rc;
        };


      auto ipc_allocate_pool_memory =
        [&ipc] (const size_t size,
                const size_t alignment,
                void *&out_new_addr) -> status_t
        {
          status_t rc = S_OK;
          ipc.send_table_op_allocate_pool_memory(size, alignment);
          ipc.recv_table_op_response(rc, out_new_addr);
          return rc;
        };

      auto ipc_free_pool_memory =
        [&ipc] (const size_t size,
                const void * addr) -> status_t
        {
          status_t rc = S_OK;
          void * na;
          ipc.send_table_op_free_pool_memory(addr, size);
          ipc.recv_table_op_response(rc, na);
          return rc;
        };

      auto ipc_find_key =
        [&ipc] (const std::string& key_expression,
                const offset_t begin_position,
                const component::IKVIndex::find_t find_type,
                offset_t& out_matched_position,
                std::string& out_matched_key) -> status_t
        {
          status_t rc = S_OK;
          ipc.send_find_index_request(key_expression,
                                      begin_position,
                                      find_type);

          ipc.recv_find_index_response(rc,
                                       out_matched_position,
                                       out_matched_key);

          return rc;
        };

      auto ipc_get_reference_vector =
        [&ipc] (const common::epoch_time_t t_begin,
                const common::epoch_time_t t_end,
                IADO_plugin::Reference_vector& out_vector) -> status_t
        {
          status_t rc = S_OK;
          ipc.send_vector_request(t_begin, t_end);
          ipc.recv_vector_response(rc, out_vector);
          return rc;
        };

      auto ipc_get_pool_info =
        [&ipc] (std::string& out_response) -> status_t
        {
          status_t rc = S_OK;
          ipc.send_pool_info_request();
          ipc.recv_pool_info_response(rc, out_response);
          return rc;
        };

      auto ipc_iterate =
        [&ipc] (const common::epoch_time_t t_begin,
                const common::epoch_time_t t_end,
                component::IKVStore::pool_iterator_t& iterator,
                component::IKVStore::pool_reference_t& reference) -> status_t
        {
          status_t rc = S_OK;
          ipc.send_iterate_request(t_begin, t_end, iterator);
          ipc.recv_iterate_response(rc, iterator, reference);
          return rc;
        };

      auto ipc_unlock =
        [&ipc] (const uint64_t work_id,
                component::IKVStore::key_t key_handle) -> status_t
        {
          status_t rc = S_OK;
          if(work_id == 0 || key_handle == nullptr) return E_INVAL;
          ipc.send_unlock_request(work_id, key_handle);
          ipc.recv_unlock_response(rc);
          return rc;
        };

      auto ipc_configure = [&ipc](const uint64_t options) -> status_t
                           {
                             status_t rc = S_OK;
                             ipc.send_configure_request(options);
                             ipc.recv_configure_response(rc);
                             return rc;
                           };

      for(auto a: ado_params) { PLOG("ado_param:%s", a.c_str()); }

      /* load plugin and register callbacks */
      std::vector<std::string> plugin_vector;
      char *plugin_name = std::strtok(const_cast<char*>(plugins.c_str()), ",");
      while (plugin_name) {
        plugin_vector.push_back(plugin_name);
        plugin_name = std::strtok(NULL, ",");
      }

      /* load plugins */
      ADO_plugin_mgr plugin_mgr(plugin_vector,
                                IADO_plugin::Callback_table{
                                  ipc_create_key,
                                    ipc_open_key,
                                    ipc_erase_key,
                                    ipc_resize_value,
                                    ipc_allocate_pool_memory,
                                    ipc_free_pool_memory,
                                    ipc_get_reference_vector,
                                    ipc_find_key,
                                    ipc_get_pool_info,
                                    ipc_iterate,
                                    ipc_unlock,
                                    ipc_configure});

      /* main loop */
      unsigned long count = 0;
      bool exit = false;
      unsigned int memory_type = 0xFF;

      if(cpu_mask.empty() == false) {
        cpu_mask_t m;
        if(string_to_mask(cpu_mask, m) == S_OK) {
          if(debug_level > 0)
            PLOG("ADO process configured with cpu mask: %s", cpu_mask.c_str());

          if (set_cpu_affinity_mask(m) == -1)
            throw Logic_exception("bad mask parameter");
        }
        if( 2 < debug_level )
          {
            PLOG("CPU_MASK: ADO process mask: [%s]", m.string_form().c_str());
          }
      }

      PLOG("ADO process: main thread (%lu) debug_level:%d", pthread_self(), debug_level);

#ifdef PROFILE
      PMAJOR("ADO: starting profiler");
      ProfilerStart("/tmp/ADO_cpu_profile.prof");
#endif

      while (!exit) {

        /* main loop servicing incoming IPC requests */

        if(debug_level > 2)
          PLOG("ADO process: waiting for message (%lu)", count);

        Buffer_header * buffer = nullptr; /* recv will dequeue this */

        /* poll until there is a request, sleep on too much polling  */
        auto st = ipc.poll_recv_sleep(buffer);
        if(st != S_OK) throw Logic_exception(__FILE__ " ADO: ipc.poll_recv_sleep failed unexpectedly");
        assert(buffer);

        /*---------------------------------------*/
        /* custom IPC message protocol - staging */
        /*---------------------------------------*/
        using namespace mcas::ipc;

        if(mcas::ipc::Message::is_valid(buffer)) {

          switch(mcas::ipc::Message::type(buffer))
            {
            case mcas::ipc::MSG_TYPE_CHIRP: {
              auto chirp = reinterpret_cast<mcas::ipc::Chirp*>(buffer);

              switch(chirp->type)
                {
                case chirp_t::SHUTDOWN:
                  PMAJOR("ADO: received Shutdown chirp in %p",
                         static_cast<void *>(buffer));
                  plugin_mgr.shutdown();
                  exit = true;
                  break;
                default:
                  throw Protocol_exception("unknown chirp");
                }
              break;
            }
            case mcas::ipc::MSG_TYPE_MAP_MEMORY: {

              auto * mm = reinterpret_cast<Map_memory*>(buffer);

              assert(memory_type != 0xFF);

              /* use same as shard virtual address for the moment */
              void * mm_addr;
              if(memory_type == 1) { /* DRAM case, e.g. mapstore */

                if(!check_xpmem_kernel_module()) {
                  PERR("inaccessible XPMEM kernel module");
                  throw General_exception("inaccessible XPMEM kernel module");
                }

                xpmem_addr seg = {0,0};
                seg.apid = xpmem_get(xpmem_segid_t(mm->token),
                                     XPMEM_RDWR,
                                     XPMEM_PERMIT_MODE,
                                     reinterpret_cast<void *>(0666));
                if(seg.apid == -1)
                  throw General_exception("xpmem_get: failed unexpectedly.");

                mm_addr = xpmem_attach(seg,
                                       mm->size,
                                       reinterpret_cast<void*>(mm->shard_addr));
              }
              else {

                if(!nupm::check_mcas_kernel_module()) {
                  PERR("inaccessible MCAS kernel module");
                  throw General_exception("inaccessible MCAS kernel module");
                }

                mm_addr = nupm::mmap_exposed_memory(mm->token,
                                                    mm->size,
                                                    reinterpret_cast<void*>(mm->shard_addr));
              }
              if(mm_addr == MAP_FAILED)
                throw General_exception("mcasmod: mmap_exposed_memory failed unexpectly.");

              PMAJOR("ADO: mapped memory %lx size:%lu addr=%p", mm->token, mm->size, mm_addr);

              /* record mapping information for clean up */
              g_shared_memory_mappings.push_back(std::make_pair(mm_addr, mm->size));

              /* register memory with plugins */
              if(plugin_mgr.register_mapped_memory(mm->shard_addr, mm_addr, mm->size) != S_OK)
                throw General_exception("calling register_mapped_memory on ADO plugin failed");

              break;
            }
            case mcas::ipc::MSG_TYPE_WORK_REQUEST:  {

              component::IADO_plugin::response_buffer_vector_t response_buffers;
              auto * wr = reinterpret_cast<Work_request*>(buffer);

              if(debug_level > 1)
                PLOG("ADO process: RECEIVED Work_request: key=(%p:%.*s) value=%p "
                     "value_len=%lu invocation_len=%lu detached_value=%p (%.*s) len=%lu new=%d",
                     static_cast<const void *>(wr->get_key()),
                     int(wr->get_key_len()),
                     wr->get_key(),
                     wr->get_value_addr(),
                     wr->value_len,
                     wr->invocation_data_len,
                     wr->get_detached_value_addr(),
                     int(wr->detached_value_len),
                     static_cast<char *>(wr->get_detached_value_addr()),
                     wr->detached_value_len,
                     wr->new_root);

              auto work_request_id = wr->work_key;

              IADO_plugin::value_space_t values;
              values.append(wr->get_value_addr(),wr->value_len);
              if(wr->detached_value_len > 0) {
                assert(wr->get_detached_value_addr() != nullptr);
                values.append(wr->get_detached_value_addr(), wr->detached_value_len);
              }

              /* forward to plugins */
              status_t rc =
                plugin_mgr.do_work(work_request_id,
                                   wr->get_key(),
                                   wr->get_key_len(),
                                   values,
                                   wr->get_invocation_data(),
                                   wr->invocation_data_len,
                                   wr->new_root,
                                   response_buffers);

              /* pass back response data */
              ipc.send_work_response(rc,
                                     work_request_id,
                                     response_buffers);

              break;
            }
            case mcas::ipc::MSG_TYPE_BOOTSTRAP_REQUEST:  {

              auto boot_req = reinterpret_cast<Bootstrap_request*>(buffer);
              std::string pool_name(boot_req->pool_name, boot_req->pool_name_len);

              if(debug_level > 2)
                PLOG("ADO process: bootstrap_request: (%s, %lu, %u, %u, %lu)",
                     pool_name.c_str(), boot_req->pool_size,
                     boot_req->pool_flags, boot_req->memory_type, boot_req->expected_obj_count);

              memory_type = boot_req->memory_type;

              /* call the plugin */
              plugin_mgr.launch_event(boot_req->auth_id,
                                      pool_name,
                                      boot_req->pool_size,
                                      boot_req->pool_flags,
                                      boot_req->memory_type,
                                      boot_req->expected_obj_count,
                                      ado_params);

              ipc.send_bootstrap_response();

              break;
            }
            case mcas::ipc::MSG_TYPE_OP_EVENT: {
              auto event = reinterpret_cast<Op_event*>(buffer);
              if(debug_level > 1)
                PLOG("ADO_process: received op event (%s)",
                     to_str(event->op).c_str());

              /* invoke plugin then return completion */
              plugin_mgr.notify_op_event(event->op);
              ipc.send_op_event_response(event->op);

              break;
            }
            case mcas::ipc::MSG_TYPE_CLUSTER_EVENT: {
              auto event = reinterpret_cast<Cluster_event*>(buffer);
              PLOG("ADO_process: received cluster event (%s,%s,%s)",
                   event->sender(), event->type(), event->message());

              plugin_mgr.send_cluster_event(event->sender(),
                                            event->type(),
                                            event->message());
              break;
            }
            default: {
              throw Logic_exception("ADO_process: unknown mcas::ipc message type");
            }
            }

          ipc.free_ipc_buffer(buffer);
          count++;
        }
      } // end of while loop

      PMAJOR("ADO: exiting.");

      /* clean up: free shared memory mappings */
      /* TODO do we need to unregister with kernel module ? */
      for(auto& mp : g_shared_memory_mappings) {
        if(::munmap(mp.first, mp.second) != 0)
          throw Logic_exception("unmap of shared memory failed");
      }

#ifdef PROFILE
      ProfilerStop();
      PMAJOR("ADO: stopped profiler");
#endif

      return 0;
    }
  catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }
  catch (const Exception &e) {
    std::cerr << e.cause() << std::endl;
    return -1;
  }

  return -1;
}
