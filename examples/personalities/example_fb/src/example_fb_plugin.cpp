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
#include <libpmem.h>
#include <common/logging.h>
#include <common/utils.h>
#include <common/dump_utils.h>
#include <common/cycles.h>
#include <api/interfaces.h>
#include <string.h>
#include "example_fb_plugin.h"

#include <flatbuffers/flatbuffers.h>
#include <example_fb_proto_generated.h>

using namespace flatbuffers;
using namespace example_fb_protocol;

int debug_level = 3;

status_t ADO_example_fb_plugin::register_mapped_memory(void * shard_vaddr,
                                                       void * local_vaddr,
                                                       size_t len)
{
  PLOG("ADO_example_fb_plugin: register_mapped_memory (%p, %p, %lu)",
       shard_vaddr, local_vaddr, len);
  /* we would need a mapping if we are not using the same virtual
     addresses as the Shard process */

  return S_OK;
}

inline void * copy_flat_buffer(FlatBufferBuilder& fbb)
{
  auto fb_len = fbb.GetSize();
  void * ptr = ::malloc(fb_len);
  memcpy(ptr, fbb.GetBufferPointer(), fb_len);
  return ptr;
}

status_t ADO_example_fb_plugin::do_work(const uint64_t work_key,
                                        const char * key,
                                        size_t key_len,
                                        IADO_plugin::value_space_t& values,
                                        const void *in_work_request,
                                        const size_t in_work_request_len,
                                        bool new_root,
                                        response_buffer_vector_t& response_buffers)
{
  using namespace flatbuffers;

  auto value = values[0].ptr;
  //  auto value_len = values[0].len;
  auto detached_value = values[1].ptr;
  auto detached_value_len = values[1].len;

  auto root = static_cast<ADO_example_fb_plugin_root *>(value);
  if(new_root) {
    root->init();
  }
  else {
    root->check_recovery();
  }

  auto msg = GetMessage(in_work_request);
  auto txid = msg->transaction_id();

  // Put
  if(msg->element_as_PutRequest()) {
    auto pr = msg->element_as_PutRequest();

    if(debug_level > 0)
      PMAJOR("Got put request (%s,%s)",
             pr->key()->str().c_str(),
             pr->value()->str().c_str());

    size_t value_to_free_len = 0;
    auto value_to_free = root->add_version(detached_value,
                                           detached_value_len,
                                           value_to_free_len);
    if(value_to_free) {
      if(debug_level > 0)
        PMAJOR("freeing: value (%.*s)",
               (int) value_to_free_len, static_cast<char*>(value_to_free));
      cb_free_pool_memory(value_to_free_len, value_to_free);
    }

    /* create response message */
    FlatBufferBuilder fbb;
    auto req = CreateAck(fbb, S_OK);
    fbb.Finish(CreateMessage(fbb, txid, Element_Ack, req.Union()));
    response_buffers.emplace_back(copy_flat_buffer(fbb), fbb.GetSize(), response_buffer_t::alloc_type_malloc{});

    return S_OK;
  }
  // Get
  else if(msg->element_as_GetRequest()) {
    auto pr = msg->element_as_GetRequest();

    if(debug_level > 2)
      PMAJOR("Got get request (%s)",
             pr->key()->str().c_str());

    void * return_value = nullptr;
    size_t return_value_len = 0;
    cpu_time_t timestamp = 0;
    root->get_version(pr->version_index(), return_value, return_value_len, timestamp);

    if(debug_level > 0)
         PLOG("Picked version to return: @ %p (%.*s)", return_value,
              (int) return_value_len, (char *) return_value);

    FlatBufferBuilder fbb;
    auto req = CreateGetResponse(fbb, timestamp/* timestamp */, return_value_len);
    fbb.FinishSizePrefixed(CreateMessage(fbb, txid, Element_GetResponse, req.Union()));

    response_buffers.emplace_back(copy_flat_buffer(fbb), fbb.GetSize(),  response_buffer_t::alloc_type_malloc{});
    response_buffers.emplace_back(return_value, return_value_len, response_buffer_t::alloc_type_pool{});
  }
  else {
    PLOG("got something unrecognized!");
  }

  return S_OK;
}

status_t ADO_example_fb_plugin::shutdown()
{
  /* here you would put graceful shutdown code if any */
  return S_OK;
}

/**
 * Factory-less entry point
 *
 */
extern "C" void * factory_createInstance(component::uuid_t interface_iid)
{
  if(interface_iid == interface::ado_plugin)
    return static_cast<void*>(new ADO_example_fb_plugin());
  else return NULL;
}

