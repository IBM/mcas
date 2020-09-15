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

#include "cpp_symtab_plugin.h"
#include <libpmem.h>
#include <api/interfaces.h>
#include <common/logging.h>
#include <common/dump_utils.h>
#include <common/type_name.h>
#include <sstream>
#include <string>
#include <list>
#include <algorithm>
#include <ccpm/immutable_list.h>
#include <ccpm/immutable_string_table.h>
#include "cpp_symtab_types.h"

using namespace symtab_ADO_protocol;
using namespace ccpm;

std::vector<const char *> pointer_table;

status_t ADO_symtab_plugin::register_mapped_memory(void * shard_vaddr,
                                                   void * local_vaddr,
                                                   size_t len) {
  PLOG("ADO_symtab_plugin: register_mapped_memory (%p, %p, %lu)", shard_vaddr,
       local_vaddr, len);

  /* we would need a mapping if we are not using the same virtual
     addresses as the Shard process */
  return S_OK;
}

void ADO_symtab_plugin::launch_event(const uint64_t auth_id,
                                     const std::string& pool_name,
                                     const size_t pool_size,
                                     const unsigned int pool_flags,
                                     const unsigned int memory_type,
                                     const size_t expected_obj_count,
                                     const std::vector<std::string>& params)
{
}


status_t ADO_symtab_plugin::do_work(const uint64_t work_request_id,
                                    const char * key,
                                    size_t key_len,
                                    IADO_plugin::value_space_t& values,
                                    const void * in_work_request, /* don't use iovec because of non-const */
                                    const size_t in_work_request_len,
                                    bool new_root,
                                    response_buffer_vector_t& response_buffers)
{
  using namespace flatbuffers;
  using namespace symtab_ADO_protocol;
  using namespace cpp_symtab_personality;

  auto value = values[0].ptr;

  constexpr size_t buffer_increment = KB(32); /* granularity for memory expansion */

  //  PLOG("invoke: value=%p value_len=%lu", value, value_len);

  auto& root = *(new (value) Value_root);
  bool force_init = false;

  if(root.num_regions == 0) {
    void * buffer;
    if(cb_allocate_pool_memory(buffer_increment, 8, buffer)!=S_OK)
      throw std::runtime_error("unable to allocate new region");
    root.add_region(buffer, buffer_increment);
    force_init = true;
  }

  /* instantiate immutable string table */
  ccpm::Immutable_string_table<> string_table(root.get_regions(), force_init);


  Verifier verifier(static_cast<const uint8_t*>(in_work_request), in_work_request_len);
  if(!VerifyMessageBuffer(verifier)) {
    PMAJOR("unknown command flatbuffer");
    return E_INVAL;
  }

  auto msg = GetMessage(in_work_request);

  /* put request handling */
  auto put_request = msg->command_as_PutRequest();
  if(put_request) {
    auto str = put_request->word()->c_str();

  retry:
    try {
      auto p = string_table.add_string(str);
      pointer_table.push_back(p);
    }
    catch(const std::bad_alloc& e) {
      void * buffer;
      if(cb_allocate_pool_memory(buffer_increment, 8, buffer)!=S_OK)
        throw std::runtime_error("unable to allocate new region");
      PLOG("Expanding memory .. %p", buffer);
      string_table.expand(::iovec{buffer, buffer_increment});
      goto retry;
    }

    return S_OK;
  }

  if(msg->command_as_BuildIndex()) {
    PMAJOR("Building index...");
    std::sort(pointer_table.begin(),
              pointer_table.end(),
              []( const char *s1, const char *s2 ) -> bool
              {
                return std::strcmp( s1, s2 ) < 0;
              }
              );

    PMAJOR("Sort complete.");

    for(unsigned i=0;i<10;i++) {
      PLOG("[%u] %s", i, pointer_table[i]);
    }

    if(root.index_size) {
      assert(root.index);
      cb_free_pool_memory(root.index_size, root.index);
      root.index = nullptr;
      root.index_size = 0;
      pmem_flush(&root.index_size, sizeof(root.index_size));
    }

    root.index = nullptr;
    void * mem;
    size_t index_size = sizeof(const char *) * pointer_table.size();
    if(cb_allocate_pool_memory(index_size,
                               8,
                               mem)!=S_OK)
      throw std::runtime_error("unable to allocate memory for index");

    root.index = new (mem) std::vector<const char*>(pointer_table.size());

    for(size_t i=0;i<pointer_table.size();i++) {
      (*root.index)[i] = pointer_table[i];
    }
    root.index_size = pointer_table.size();
    pmem_flush(mem, index_size);
    pmem_flush(&root.index_size, sizeof(root.index_size));
    pmem_flush(&root.index, index_size);
    PLOG("Index (%lu entries) built OK.", root.index_size);

    for(unsigned i=0;i<10;i++) {
      PLOG("[%u-%p] %s", i, &root.index[i],  (*root.index)[i]);
    }
    PLOG("...");

    return S_OK;
  }

  auto get_symbol_request = msg->command_as_GetSymbol();
  if(get_symbol_request) {
    auto& req_word = *(get_symbol_request->word());
    PLOG("Get symbol for \"%s\"", req_word.c_str());
    PLOG("Root index size=%lu", root.index->size());
    /* binary chop - could use hash index */

    //std::size_t size, /*compare-pred*/* comp );
    auto i = std::lower_bound(root.index->begin(),
                              root.index->end(),
                              req_word.c_str(),
                              [](const char* left,
                                 const char* right) -> bool {
                                PLOG("%s--%s", left, right);
                                auto comp = strcmp(left, right) < 0;
                                //answer = const_cast<char*>(right);
                                return comp;
                              }
                              );
    if(i != root.index->end()) {
      PLOG("Found it! (%p)", *i);
      auto result = new uint64_t;
      *result = reinterpret_cast<uint64_t>(*i);
      response_buffers.emplace_back(result, sizeof(uint64_t), response_buffer_t::alloc_type_malloc{});
    }

    return S_OK;
  }

  auto get_string_request = msg->command_as_GetString();
  if(get_string_request) {
    char* sym_id = reinterpret_cast<char *>(get_string_request->symbol());
    PLOG("Request symbol:%p", sym_id);
    response_buffers.emplace_back(sym_id, strlen(sym_id), response_buffer_t::alloc_type_pool{});
    return S_OK;
  }

  PERR("unhandled command");
  return E_INVAL;
}

status_t ADO_symtab_plugin::shutdown() {
  /* here you would put graceful shutdown code if any */
  return S_OK;
}

/**
 * Factory-less entry point
 *
 */
extern "C" void *factory_createInstance(component::uuid_t interface_iid) {
  PLOG("instantiating cpp-symtab-plugin");
  if (interface_iid == interface::ado_plugin)
    return static_cast<void *>(new ADO_symtab_plugin());
  else
    return NULL;
}

#undef RESET_STATE


