#include <unistd.h>

#include <api/components.h>
#include <api/mcas_itf.h>

#include "mcas_api_wrapper.h"

using namespace component;

namespace globals
{
}

extern "C" mcas_session_t mcas_open_session_ex(const char * server_addr,
                                               const char * net_device,
                                               unsigned debug_level,
                                               unsigned patience)
{
  auto comp = load_component("libcomponent-mcasclient.so", mcas_client_factory);
  auto factory = static_cast<IMCAS_factory *>(comp->query_interface(IMCAS_factory::iid()));
  assert(factory);

  /* create instance of MCAS client session */
  mcas_session_t mcas = factory->mcas_create(debug_level /* debug level, 0=off */,
                                             patience,
                                             getlogin(),
                                             server_addr, /* MCAS server endpoint, e.g. 10.0.0.101::11911 */
                                             net_device); /* e.g., mlx5_0, eth0 */

  factory->release_ref();
  assert(mcas);
  return mcas;
}


extern "C" int mcas_close_session(const mcas_session_t session)
{
  auto mcas = static_cast<IMCAS*>(session);
  if(!session) return -1;
  
  mcas->release_ref();
  
  return S_OK;
}

extern "C" int mcas_create_pool_ex(const mcas_session_t session,
                                   const char * pool_name,
                                   const size_t size,
                                   const unsigned int flags,
                                   const uint64_t expected_obj_count,
                                   const addr_t base_addr,
                                   mcas_pool_t * out_pool_handle)                                           
{
  auto mcas = static_cast<IMCAS*>(session);
  auto pool = mcas->create_pool(std::string(pool_name),
                                size,
                                flags,
                                expected_obj_count,
                                IKVStore::Addr(base_addr));
  if(pool == IMCAS::POOL_ERROR) return -1;
  *out_pool_handle = {mcas, pool};
  return 0;
}

extern "C" int mcas_create_pool(const mcas_session_t session,
                                const char * pool_name,
                                const size_t size,
                                const mcas_flags_t flags,
                                mcas_pool_t * out_pool_handle)
{
  return mcas_create_pool_ex(session, pool_name, size, flags, 1000, 0, out_pool_handle);
}


extern "C" int mcas_open_pool_ex(const mcas_session_t session,
                                 const char * pool_name,
                                 const mcas_flags_t flags,
                                 const addr_t base_addr,
                                 mcas_pool_t * out_pool_handle)
{
  auto mcas = static_cast<IMCAS*>(session);
  auto pool = mcas->open_pool(std::string(pool_name),
                              flags,
                              IKVStore::Addr(base_addr));

  if(pool == IMCAS::POOL_ERROR) return -1;
  *out_pool_handle = {mcas, pool};  
  return 0;
}

extern "C" int mcas_open_pool(const mcas_session_t session,
                              const char * pool_name,
                              const mcas_flags_t flags,
                              mcas_pool_t * out_pool_handle)
{
  return mcas_open_pool_ex(session, pool_name, flags, 0, out_pool_handle);
}


extern "C" int mcas_close_pool(const mcas_pool_t pool)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  return mcas->close_pool(pool.handle) == S_OK ? 0 : -1;
}

extern "C" int mcas_delete_pool(const mcas_session_t session,
                                const char * pool_name)
{
  auto mcas = static_cast<IMCAS*>(session);
  return mcas->delete_pool(pool_name);
}

extern "C" int mcas_close_delete_pool(const mcas_pool_t pool)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);
  return mcas->delete_pool(poolh);
}

extern "C" int mcas_configure_pool(const mcas_pool_t pool,
                                   const char * setting)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);
  return mcas->configure_pool(poolh, std::string(setting));
}

extern "C" int mcas_put_ex(const mcas_pool_t pool,
                           const char * key,
                           const void * value,
                           const size_t value_len,
                           const unsigned int flags)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);
  return mcas->put(poolh, std::string(key), value, value_len, flags);
}

extern "C" int mcas_put(const mcas_pool_t pool,
                        const char * key,
                        const char * value,
                        const unsigned int flags)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);
  return mcas->put(poolh, std::string(key), value, strlen(value), flags);

}

extern "C" int mcas_register_direct_memory(const mcas_session_t session,
                                           const void * addr,
                                           const size_t len,
                                           mcas_memory_handle_t* out_handle)
{
  auto mcas = static_cast<IMCAS*>(session);
  auto handle = mcas->register_direct_memory(const_cast<void*>(addr), len);
  if(handle == 0) return -1;
  *out_handle = handle;
  return 0;
}

extern "C" int mcas_unregister_direct_memory(const mcas_session_t session,
                                             const mcas_memory_handle_t handle)
{
  auto mcas = static_cast<IMCAS*>(session);
  return mcas->unregister_direct_memory(static_cast<IMCAS::memory_handle_t>(handle));
}

extern "C" int mcas_put_direct_ex(const mcas_pool_t pool,
                                  const char * key,
                                  const void * value,
                                  const size_t value_len,
                                  const mcas_memory_handle_t handle,
                                  const unsigned int flags)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);
  return mcas->put_direct(poolh, key, value, value_len,
                          static_cast<IMCAS::memory_handle_t>(handle),
                          flags);
}

extern "C" int mcas_put_direct(const mcas_pool_t pool,
                               const char * key,
                               const void * value,
                               const size_t value_len)
{
  return mcas_put_direct_ex(pool, key, value, value_len, IMCAS::MEMORY_HANDLE_NONE, IMCAS::FLAGS_NONE);
}



extern "C" int mcas_async_put_ex(const mcas_pool_t pool,
                                 const char * key,
                                 const void * value,
                                 const size_t value_len,
                                 const unsigned int flags,
                                 mcas_async_handle_t * out_async_handle)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);
  IMCAS::async_handle_t handle;
  auto result = mcas->async_put(poolh, key, value, value_len, handle, flags);
  if(result == S_OK)
    *out_async_handle = {static_cast<void*>(handle), nullptr};
  return result;
}

extern "C" int mcas_async_put(const mcas_pool_t pool,
                              const char * key,
                              const char * value,
                              const unsigned int flags,
                              mcas_async_handle_t * out_async_handle)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);
  IMCAS::async_handle_t ahandle;
  auto result = mcas->async_put(poolh, key, value, strlen(value), ahandle, flags);
  if(result == S_OK)
    *out_async_handle = {static_cast<void*>(ahandle), nullptr};
  return result;
}

extern "C" int mcas_async_put_direct_ex(const mcas_pool_t pool,
                                        const char * key,
                                        const void * value,
                                        const size_t value_len,
                                        const mcas_memory_handle_t handle,
                                        const unsigned int flags,
                                        mcas_async_handle_t * out_async_handle)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);
  IMCAS::async_handle_t ahandle;
  auto result = mcas->async_put_direct(poolh, key, value, value_len,
                                       ahandle,
                                       static_cast<IMCAS::memory_handle_t>(handle),
                                       flags);

  if(result == S_OK)
    *out_async_handle = {static_cast<void*>(ahandle), nullptr};
  return result;
}

extern "C" int mcas_free_memory(const mcas_session_t session,
                                void * p)
{
  auto mcas = static_cast<IMCAS*>(session);
  return mcas->free_memory(p);
}
    


extern "C" int mcas_get(const mcas_pool_t pool,
                        const char * key,
                        void** out_value,
                        size_t* out_value_len)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);
  return mcas->get(poolh, key, *out_value, *out_value_len);
}

extern "C" int mcas_get_direct_ex(const mcas_pool_t pool,
                                  const char * key,
                                  void * out_value,
                                  size_t * inout_size_value,
                                  mcas_memory_handle_t handle)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);
  return mcas->get_direct(poolh, key, out_value, *inout_size_value,
                          static_cast<IMCAS::memory_handle_t>(handle));
}

extern "C" int mcas_get_direct(const mcas_pool_t pool,
                               const char * key,
                               void * out_value,
                               size_t * inout_size_value)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);
  return mcas->get_direct(poolh, key, out_value, *inout_size_value, IMCAS::MEMORY_HANDLE_NONE);
}


extern "C" int mcas_async_get_direct_ex(const mcas_pool_t pool,
                                        const char * key,
                                        void * out_value,
                                        size_t * inout_size_value,
                                        mcas_memory_handle_t handle,
                                        mcas_async_handle_t * out_async_handle)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);

  IMCAS::async_handle_t ahandle;
  auto result = mcas->async_get_direct(poolh, key, out_value, *inout_size_value,
                                       ahandle, static_cast<IMCAS::memory_handle_t>(handle));
  if(result == S_OK)
    *out_async_handle = {ahandle, nullptr};
  return result;
}


extern "C" int mcas_async_get_direct(const mcas_pool_t pool,
                                     const char * key,
                                     void * out_value,
                                     size_t * inout_size_value,
                                     mcas_async_handle_t * out_async_handle)
{
  return mcas_async_get_direct_ex(pool, key, out_value, inout_size_value,
                                  IMCAS::MEMORY_HANDLE_NONE, out_async_handle);
}


extern "C" int mcas_get_direct_offset_ex(const mcas_pool_t pool,
                                         const offset_t offset,
                                         void * out_buffer,
                                         size_t * inout_size,
                                         mcas_memory_handle_t handle)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);
  return mcas->get_direct_offset(poolh, offset, *inout_size, out_buffer,
                                 static_cast<IMCAS::memory_handle_t>(handle));
}

extern "C" int mcas_get_direct_offset(const mcas_pool_t pool,
                                      const offset_t offset,
                                      void * out_buffer,
                                      size_t * inout_size)
{
  return mcas_get_direct_offset_ex(pool, offset,out_buffer, inout_size, IMCAS::MEMORY_HANDLE_NONE);
}


extern "C" int mcas_async_get_direct_offset_ex(const mcas_pool_t pool,
                                               const offset_t offset,
                                               void * out_buffer,
                                               size_t * inout_size,
                                               mcas_memory_handle_t handle,
                                               mcas_async_handle_t * out_async_handle)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);
  IMCAS::async_handle_t ahandle;
  auto result = mcas->async_get_direct_offset(poolh, offset, *inout_size, out_buffer,
                                              ahandle,
                                              static_cast<IMCAS::memory_handle_t>(handle));
  
  if(result == S_OK)
    *out_async_handle = {ahandle, nullptr};
  
  return result;
}

extern "C" int mcas_async_get_direct_offset(const mcas_pool_t pool,
                                            const offset_t offset,
                                            void * out_buffer,
                                            size_t * inout_size,
                                            mcas_async_handle_t * out_async_handle)
{
  return mcas_async_get_direct_offset_ex(pool, offset,
                                         out_buffer, inout_size,
                                         IMCAS::MEMORY_HANDLE_NONE,
                                         out_async_handle);
}

extern "C" int mcas_put_direct_offset_ex(const mcas_pool_t pool,
                                         const offset_t offset,
                                         const void *const buffer,
                                         size_t * inout_size,
                                         mcas_memory_handle_t handle)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);  
  
  return mcas->put_direct_offset(poolh, offset, *inout_size, buffer,
                                 static_cast<IMCAS::memory_handle_t>(handle));
}

extern "C" int mcas_put_direct_offset(const mcas_pool_t pool,
                                      const offset_t offset,
                                      const void *const buffer,
                                      size_t * inout_size)
{
  return mcas_put_direct_offset_ex(pool, offset, buffer, inout_size,
                                   IMCAS::MEMORY_HANDLE_NONE);
}


extern "C" int mcas_async_put_direct_offset_ex(const mcas_pool_t pool,
                                               const offset_t offset,
                                               const void *const buffer,
                                               size_t * inout_size,
                                               mcas_memory_handle_t handle,
                                               mcas_async_handle_t * out_async_handle)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);
  IMCAS::async_handle_t ahandle;
  
  auto result = mcas->async_put_direct_offset(poolh, offset, *inout_size, buffer,
                                              ahandle,
                                              static_cast<IMCAS::memory_handle_t>(handle));
  if(result == S_OK)
    *out_async_handle = {ahandle, nullptr};
  
  return result;
}

extern "C" int mcas_async_put_direct_offset(const mcas_pool_t pool,
                                            const offset_t offset,
                                            const void *const buffer,
                                            size_t * inout_size,
                                            mcas_async_handle_t * out_async_handle)
{
  return mcas_async_put_direct_offset_ex(pool, offset, buffer, inout_size,
                                         IMCAS::MEMORY_HANDLE_NONE, out_async_handle);
}

extern "C" int mcas_check_async_completion(const mcas_session_t session,
                                           const mcas_async_handle_t handle)
{
  auto mcas = static_cast<IMCAS*>(session);
  return mcas->check_async_completion(static_cast<IMCAS::async_handle_t>(handle.internal));
}

extern "C" int mcas_find(const mcas_pool_t pool,
                         const char * key_expression,
                         const offset_t offset,
                         offset_t* out_matched_offset,
                         char** out_matched_key)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);
  std::string result_key;
  offset_t result_offset = 0;
  auto result = mcas->find(poolh, key_expression, offset, result_offset, result_key);
  if(result == S_OK) {
    *out_matched_offset = result_offset;
    *out_matched_key = static_cast<char*>(::malloc(result_key.size()));
    memcpy(*out_matched_key, result_key.c_str(), result_key.size());
  }    
  return result;
}

extern "C" int mcas_erase(const mcas_pool_t pool,
                          const char * key)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);
  return mcas->erase(poolh, key);
}

extern "C" int mcas_async_erase(const mcas_pool_t pool,
                                const char * key,
                                mcas_async_handle_t * handle)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);
  IMCAS::async_handle_t ahandle;
  auto result = mcas->async_erase(poolh, key, ahandle);
  if(result == S_OK)
    *handle = {ahandle, nullptr};
  return result;
}
  
extern "C" size_t mcas_count(const mcas_pool_t pool)
{
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);
  return mcas->count(poolh);
}

extern "C" int mcas_get_attribute(const mcas_pool_t pool,
                                  const char * key, 
                                  mcas_attribute attr,
                                  uint64_t** out_value,
                                  size_t* out_value_count)
{
  assert(out_value);
  assert(out_value_count);
  
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);
  std::vector<uint64_t> out_attrs;

  status_t result;
  if(key) {
    std::string k(key);
    result = mcas->get_attribute(poolh, static_cast<IMCAS::Attribute>(attr), out_attrs, &k);
  }
  else {
    result = mcas->get_attribute(poolh, static_cast<IMCAS::Attribute>(attr), out_attrs, nullptr);
  }

  auto n_elems = out_attrs.size();

  if(result == S_OK) {
    *out_value_count = n_elems;
    if(n_elems > 0) {
      *out_value = static_cast<uint64_t*>(::malloc(n_elems * sizeof(uint64_t)));
      unsigned i=0;
      for(auto x: out_attrs) {
        *out_value[i] = x;
        i++;
      }
    }
    else {
      *out_value = nullptr;
    }
  }      

  return result;
}

extern "C" void mcas_free_responses(mcas_response_array_t * out_response_vector)
{
  assert(out_response_vector);

  struct iovec* iovec_ptr = static_cast<struct iovec*>(*out_response_vector);
  while(iovec_ptr != nullptr) {
    ::free(iovec_ptr->iov_base);
    iovec_ptr++;
  }
  
  ::free(out_response_vector);
}


extern "C" int mcas_invoke_ado(const mcas_pool_t pool,
                               const char * key,
                               const void * request,
                               const size_t request_len,
                               const mcas_ado_flags_t flags,
                               const size_t value_size,
                               mcas_response_array_t * out_response_vector,
                               size_t * out_response_vector_count)
{
  using namespace std;
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);

  std::vector<IMCAS::ADO_response> responses;
  
  auto result = mcas->invoke_ado(poolh,
                                 key,
                                 request,
                                 request_len,
                                 flags,
                                 responses,
                                 value_size);

  auto n_responses = responses.size();
  if(result == S_OK) {
    *out_response_vector_count = n_responses;
    if(n_responses > 0) {
      auto rv = static_cast<struct iovec*>(::malloc((n_responses + 1) * sizeof(iovec)));
      *out_response_vector = rv;
      unsigned i = 0;
      for(auto& r : responses) {
        auto len = r.data_len();
        rv[i].iov_len = len;
        rv[i].iov_base = ::malloc(len);
        memcpy(rv[i].iov_base, r.data(), len); /* may be we can eliminate this? */
        i++;
      }
      rv[i].iov_base = nullptr; /* mark end of array */
      rv[i].iov_len = 0; 
    }
    else {
      *out_response_vector = NULL;
    }
  }
  else {
    *out_response_vector_count = 0;
    *out_response_vector = nullptr;
  }
  return result;
}


extern "C" int mcas_async_invoke_ado(const mcas_pool_t pool,
                                     const char * key,
                                     const void * request,
                                     const size_t request_len,
                                     const mcas_ado_flags_t flags,
                                     const size_t value_size,
                                     mcas_async_handle_t * handle)
{
  using namespace std;
  auto mcas = static_cast<IMCAS*>(pool.session);
  auto poolh = static_cast<IMCAS::pool_t>(pool.handle);

  using rv_t = std::vector<IMCAS::ADO_response>;
  handle->data = new rv_t;

  IMCAS::async_handle_t ahandle;
  auto result = mcas->async_invoke_ado(poolh,
                                       key,
                                       request,
                                       request_len,
                                       flags,
                                       *(reinterpret_cast<rv_t*>(handle->data)),
                                       ahandle,
                                       value_size);
  handle->internal = static_cast<void*>(ahandle);

  return result;
}

extern "C" int mcas_check_async_invoke_ado(const mcas_pool_t pool,
                                           mcas_async_handle_t * handle,
                                           mcas_response_array_t * out_response_vector,
                                           size_t * out_response_vector_count)
{
  using namespace component;
  auto mcas = static_cast<IMCAS*>(pool.session);

  if(mcas->check_async_completion(static_cast<IMCAS::async_handle_t>(handle->internal)) == S_OK) {
    if(handle->data == nullptr)
      throw General_exception("unexpected null handle data");
    auto& responses = *(reinterpret_cast<std::vector<IMCAS::ADO_response>*>(handle->data));

    auto n_responses = responses.size();
    *out_response_vector_count = n_responses;
    if(n_responses > 0) {
      auto rv = static_cast<struct iovec*>(::malloc((n_responses + 1) * sizeof(iovec)));
      *out_response_vector = rv;
      unsigned i = 0;
      for(auto& r : responses) {
        auto len = r.data_len();
        rv[i].iov_len = len;
        rv[i].iov_base = ::malloc(len);
        memcpy(rv[i].iov_base, r.data(), len); /* may be we can eliminate this? */
        i++;
      }
      rv[i].iov_base = nullptr; /* mark end of array */
      rv[i].iov_len = 0; 
    }
    else {
      *out_response_vector = NULL;
    }
    return 0;
  }

  return -1;
}


