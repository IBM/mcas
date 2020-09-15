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

#include "cpp_list_plugin.h"
#include <libpmem.h>
#include <api/interfaces.h>
#include <common/logging.h>
#include <common/dump_utils.h>
#include <common/type_name.h>
#include <sstream>
#include <string>
#include <ccpm/immutable_list.h>


status_t ADO_structured_plugin::register_mapped_memory(void *shard_vaddr,
                                                       void *local_vaddr,
                                                       size_t len) {
  PLOG("ADO_structured_plugin: register_mapped_memory (%p, %p, %lu)", shard_vaddr,
       local_vaddr, len);

  /* we would need a mapping if we are not using the same virtual
     addresses as the Shard process */
  return S_OK;
}

using namespace structured_ADO_protocol;
using namespace ccpm;


status_t
ADO_structured_plugin::process_putvar_command(const structured_ADO_protocol::PutVariable * command,
                                              const ccpm::region_vector_t& regions)
{
  /* just check the integrity? */
  PNOTICE("putvar: (%s)", command->container_type()->c_str());
  ccpm::Immutable_list<uint64_t> target(regions); /* checks integrity */

  target.sort();

  for(auto i: target)
    PLOG("list existing members: %lu", i);

  return S_OK;
}

status_t ADO_structured_plugin::process_invoke_command(const structured_ADO_protocol::Invoke * command,
                                                       const ccpm::region_vector_t& regions,
                                                       void*& out_work_response,
                                                       size_t& out_work_response_len)
{
  PLOG("invoke command");

  ccpm::Immutable_list<uint64_t> target(regions); /* checks integrity */

  if(command->method()->str() == "push_front") {
    using Reader = nop::StreamReader<std::stringstream>;
    nop::Deserializer<Reader> deserializer(command->serialized_params()->str());
    uint64_t val;
    if(!deserializer.Read(&val))
      throw General_exception("bad deserialization");
    PLOG("val = %lu", val);
    target.push_front(val);
  }
  else if(command->method()->str() == "sort") {
    PLOG("sorting list");
    target.sort();
  }
  else {
    PWRN("unknown invoke method");
  }

  for(auto i: target)
    PLOG("list existing members: %lu", i);

  return S_OK;
}


status_t ADO_structured_plugin::do_work(const uint64_t work_request_id,
                                        const char * key,
                                        size_t key_len,
                                        IADO_plugin::value_space_t& values,
                                        const void *in_work_request, /* don't use iovec because of non-const */
                                        const size_t in_work_request_len,
                                        bool new_root,
                                        response_buffer_vector_t& response_buffers)
{
  using namespace flatbuffers;
  using namespace structured_ADO_protocol;

  auto value = values[0].ptr;
  auto value_len = values[0].len;

  PLOG("invoke: value=%p value_len=%lu", value, value_len);

  Verifier verifier(static_cast<const uint8_t*>(in_work_request), in_work_request_len);
  if(!VerifyMessageBuffer(verifier)) {
    PMAJOR("unknown command");
  }

  auto msg = GetMessage(in_work_request);

  auto putvar_command = msg->command_as_PutVariable();
  if(putvar_command)
    return process_putvar_command(putvar_command, ccpm::region_vector_t{value, value_len});

  auto invoke_command = msg->command_as_Invoke();
  if(invoke_command)
    return process_invoke_command(invoke_command,
                                  ccpm::region_vector_t{value, value_len},
                                  value,
                                  value_len);

  PERR("unhandled command");
  return E_FAIL;
}

status_t ADO_structured_plugin::shutdown() {
  /* here you would put graceful shutdown code if any */
  return S_OK;
}

/**
 * Factory-less entry point
 *
 */
extern "C" void *factory_createInstance(component::uuid_t interface_iid) {
  PLOG("instantiating ADO_structured_plugin");
  if (interface_iid == interface::ado_plugin)
    return static_cast<void *>(new ADO_structured_plugin());
  else
    return NULL;
}

#undef RESET_STATE


