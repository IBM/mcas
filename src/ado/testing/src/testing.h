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


/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __TESTING_PLUGIN_COMPONENT_H__
#define __TESTING_PLUGIN_COMPONENT_H__

#include <api/ado_itf.h>


class ADO_testing_plugin : public component::IADO_plugin
{
private:
  static constexpr bool option_DEBUG = false;

public:
  /**
   * Constructor
   *
   * @param block_device Block device interface
   *
   */
  ADO_testing_plugin() {}

  /**
   * Destructor
   *
   */
  virtual ~ADO_testing_plugin() {}

  /**
   * Component/interface management
   *
   */
  DECLARE_VERSION(0.1f);
  DECLARE_COMPONENT_UUID(0x59564581,0x9e1b,0x4811,0xbdb2,0x19,0x57,0xa0,0xa6,0x84,0x57);

  void * query_interface(component::uuid_t& itf_uuid) override {
    if(itf_uuid == component::IADO_plugin::iid()) {
      return static_cast<component::IADO_plugin*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

public:

  /* IADO_plugin */
  status_t register_mapped_memory(void * shard_vaddr,
                                  void * local_vaddr,
                                  size_t len) override;


  status_t do_work(const uint64_t work_key,
                   const char * key,
                   size_t key_len,
                   IADO_plugin::value_space_t& values,
                   const void * in_work_request, /* don't use iovec because of non-const */
                   const size_t in_work_request_len,
                   bool new_root,
                   response_buffer_vector_t& response_buffers) override;

  void launch_event(const uint64_t auth_id,
                    const std::string& pool_name,
                    const size_t pool_size,
                    const unsigned int pool_flags,
                    const unsigned int memory_type,
                    const size_t expected_obj_count,
                    const std::vector<std::string>& params) override;

  void notify_op_event(component::ADO_op op) override;

  status_t shutdown() override;

};




#endif
