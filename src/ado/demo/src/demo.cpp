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

#include "demo.h"
#include <libpmem.h>
#include <api/interfaces.h>
#include <common/logging.h>
#include <cstring>

status_t ADO_demo_plugin::register_mapped_memory(void *shard_vaddr,
                                                 void *local_vaddr,
                                                 size_t len) {
  PLOG("ADO_demo_plugin: register_mapped_memory (%p, %p, %lu)", shard_vaddr,
       local_vaddr, len);
  /* we would need a mapping if we are not using the same virtual
     addresses as the Shard process */
  return S_OK;
}

status_t ADO_demo_plugin::do_work(uint64_t work_key,
                                  const char * key,
                                  size_t key_len,
                                  IADO_plugin::value_space_t& values,
                                  const void *in_work_request, /* don't use iovec because of non-const */
                                  const size_t in_work_request_len,
                                  bool new_root,
                                  response_buffer_vector_t& response_buffers)
{
  (void)key_len; // unused
  (void)in_work_request; // unused
  (void)in_work_request_len; // unused
  (void)new_root; // unused
  PLOG("ADO_demo_plugin: work_id (%lu)", work_key);
  PLOG("ADO_demo_plugin: do_work (%s, value_addr=%p, valuen_len=%lu)",
       key, values[0].ptr, values[0].len);
  PLOG("Current value: %.*s", int(values[0].len), static_cast<const char *>(values[0].ptr));
  std::memset(values[0].ptr, 'X', values[0].len);
  pmem_flush(values[0].ptr, values[0].len);

  {
    auto p = ::malloc(3);
    if ( p == nullptr )
    {
      throw std::bad_alloc();
    }
    response_buffers.emplace_back(p, 3, response_buffer_t::alloc_type_malloc{});
  }
  const auto okay = "OK!";
  ::memcpy(response_buffers[0].ptr, okay, strlen(okay));

  /* test callback */
  void *new_value_addr = nullptr;
  _cb.create_key(work_key, "newKey", 12, true, new_value_addr, nullptr, nullptr);
  std::memset(new_value_addr, 'N', 12);
  PLOG("new key created at %p", new_value_addr);
  PMAJOR("demo ADO plugin work complete");
  return S_OK;
}

status_t ADO_demo_plugin::shutdown() {
  /* here you would put graceful shutdown code if any */
  return S_OK;
}

/**
 * Factory-less entry point
 *
 */
extern "C" void *factory_createInstance(component::uuid_t interface_iid) {
  PLOG("instantiating ADO_demo_plugin");
  if (interface_iid == interface::ado_plugin)
    return static_cast<void *>(new ADO_demo_plugin());
  else
    return NULL;
}

#undef RESET_STATE
