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

extern "C" mcas_pool_t mcas_create_pool_ex(const mcas_session_t session,
                                           const char * pool_name,
                                           const size_t size,
                                           const unsigned int flags,
                                           const uint64_t expected_obj_count,
                                           const addr_t base_addr)
{
  auto mcas = static_cast<IMCAS*>(session);
  auto pool = mcas->create_pool(std::string(pool_name),
                                size,
                                flags,
                                expected_obj_count,
                                IKVStore::Addr(base_addr));
  return pool;
}

extern "C" mcas_pool_t mcas_create_pool(const mcas_session_t session,
                                        const char * pool_name,
                                        const size_t size,
                                        const mcas_flags_t flags) {
  return mcas_create_pool_ex(session, pool_name, size, flags, 1000, 0);
}


extern "C" mcas_pool_t mcas_open_pool_ex(const mcas_session_t session,
                                         const char * pool_name,
                                         const mcas_flags_t flags,
                                         const void * base_addr)
{
  auto mcas = static_cast<IMCAS*>(session);
  auto pool = mcas->open_pool(std::string(pool_name),
                              flags,
                              IKVStore::Addr(reinterpret_cast<addr_t>(base_addr)));
  return pool;  
}

extern "C" mcas_pool_t mcas_open_pool(const mcas_session_t session,
                                      const char * pool_name,
                                      const mcas_flags_t flags)
{
  return mcas_open_pool_ex(session, pool_name, flags, NULL);
}



extern "C" int mcas_close_pool(const mcas_session_t session,
                               const mcas_pool_t pool)
{
  auto mcas = static_cast<IMCAS*>(session);
  return mcas->close_pool(pool) == S_OK ? 0 : -1;
}
