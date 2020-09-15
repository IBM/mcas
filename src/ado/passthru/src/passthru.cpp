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

#include "passthru.h"
#include <libpmem.h>
#include <api/interfaces.h>
#include <common/logging.h>
#include <string>

status_t ADO_passthru_plugin::register_mapped_memory(void *shard_vaddr,
                                                     void *local_vaddr,
                                                     size_t len) {
  PLOG("ADO_passthru_plugin: register_mapped_memory (%p, %p, %lu)", shard_vaddr,
       local_vaddr, len);
  /* we would need a mapping if we are not using the same virtual
     addresses as the Shard process */
  return S_OK;
}

status_t ADO_passthru_plugin::do_work(uint64_t work_key,
                                      const char * key,
                                      size_t key_len,
                                      IADO_plugin::value_space_t& values,
                                      const void *in_work_request, /* don't use iovec because of non-const */
                                      const size_t in_work_request_len,
                                      bool new_root,
                                      response_buffer_vector_t& response_buffers) {
  (void)work_key; // unused
  (void)key; // unused
  (void)key_len; // unused
  (void)values; // unused
  (void)in_work_request; // unused
  (void)in_work_request_len; // unused
  (void)new_root; // unused
  (void)response_buffers; // unused

  return S_OK;
}

status_t ADO_passthru_plugin::shutdown() {
  /* here you would put graceful shutdown code if any */
  return S_OK;
}

/**
 * Factory-less entry point
 *
 */
extern "C" void *factory_createInstance(component::uuid_t interface_iid) {
  PLOG("instantiating ADO_passthru_plugin");
  if (interface_iid == interface::ado_plugin)
    return static_cast<void *>(new ADO_passthru_plugin());
  else
    return NULL;
}

#undef RESET_STATE
