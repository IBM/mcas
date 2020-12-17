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

/** 
 * C-only wrapper to MCAS client component. Mainly used for
 * foreign-function interfacing.
 * 
 */
#ifndef __MCAS_API_WRAPPER_H__
#define __MCAS_API_WRAPPER_H__

#include <stdint.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C"
{
#endif

  typedef void *        mcas_session_t; /*< handle to MCAS session */
  typedef uint32_t      mcas_flags_t;
  typedef uint64_t      mcas_pool_t; /*< handle to MCAS pool */
  typedef uint32_t      mcas_ado_flags_t;
  typedef uint64_t      addr_t;
  
  #define POOL_ERROR 0

  /* see kvstore_itf.h */
  static const mcas_flags_t FLAGS_NONE      = 0x0;
  static const mcas_flags_t FLAGS_READ_ONLY = 0x1; /* lock read-only */
  static const mcas_flags_t FLAGS_SET_SIZE    = 0x2;
  static const mcas_flags_t FLAGS_CREATE_ONLY = 0x4;  /* only succeed if no existing k-v pair exist */
  static const mcas_flags_t FLAGS_DONT_STOMP  = 0x8;  /* do not overwrite existing k-v pair */
  static const mcas_flags_t FLAGS_NO_RESIZE   = 0x10; /* if size < existing size, do not resize */
  static const mcas_flags_t FLAGS_MAX_VALUE   = 0x10;

  /* see mcas_itf.h */
  static const mcas_ado_flags_t ADO_FLAG_NONE = 0;
  /*< operation is asynchronous */
  static const mcas_ado_flags_t ADO_FLAG_ASYNC = (1 << 0);
  /*< create KV pair if needed */
  static const mcas_ado_flags_t ADO_FLAG_CREATE_ON_DEMAND = (1 << 1);
  /*< create only - allocate key,value but don't call ADO */
  static const mcas_ado_flags_t ADO_FLAG_CREATE_ONLY = (1 << 2);
  /*< do not overwrite value if it already exists */
  static const mcas_ado_flags_t ADO_FLAG_NO_OVERWRITE = (1 << 3);
  /*< create value but do not attach to key, unless key does not exist */
  static const mcas_ado_flags_t ADO_FLAG_DETACHED = (1 << 4);
  /*< only take read lock */
  static const mcas_ado_flags_t ADO_FLAG_READ_ONLY = (1 << 5);
  /*< zero any newly allocated value memory */
  static const mcas_ado_flags_t ADO_FLAG_ZERO_NEW_VALUE = (1 << 6);
  /*< internal use only: on return provide IO response */
  static const mcas_ado_flags_t ADO_FLAG_INTERNAL_IO_RESPONSE = (1 << 7);
  /*< internal use only: on return provide IO response with value buffer */
  static const mcas_ado_flags_t ADO_FLAG_INTERNAL_IO_RESPONSE_VALUE = (1 << 8);

  
  /** 
   * Open session to MCAS endpoint
   * 
   * @param server_addr Address of endpoint (e.g., 10.0.0.101::11911)
   * @param net_device Network device (e.g., mlx5_0, eth0)
   * @param debug_level Debug level (0=off)
   * @param patience Timeout patience in seconds (default 30)
   * 
   * @return Handle to mcas session
   */
  mcas_session_t mcas_open_session_ex(const char * server_addr,
                                      const char * net_device,
                                      unsigned debug_level,
                                      unsigned patience);

  inline mcas_session_t mcas_open_session(const char * server_addr,
                                          const char * net_device) {
    return mcas_open_session_ex(server_addr, net_device, 0, 30);
  }




  /** 
   * Close session
   * 
   * @param session Handle returned by mcas_open_session
   * 
   * @return 0 on success, -1 on error
   */
  int mcas_close_session(const mcas_session_t session);

  /** 
   * Create a new pool
   * 
   * @param session Session handle
   * @param pool_name Unique pool name
   * @param size Size of pool in bytes (for keys,values and metadata)
   * @param flags Creation flags
   * @param expected_obj_count Expected maximum object count (optimization)
   * @param base Optional base address
   *
   * @return Pool handle
   */
  mcas_pool_t mcas_create_pool_ex(const mcas_session_t session,
                                  const char * pool_name,
                                  const size_t size,
                                  const mcas_flags_t flags,
                                  const uint64_t expected_obj_count,
                                  const addr_t base_addr);

  mcas_pool_t mcas_create_pool(const mcas_session_t session,
                               const char * pool_name,
                               const size_t size,
                               const mcas_flags_t flags);

  /** 
   * Open existing pool
   * 
   * @param session Session handle
   * @param pool_name Pool name
   * @param flags Optional flags 
   * @param base_addr Optional base address
   * 
   * @return Pool handle
   */
  mcas_pool_t mcas_open_pool_ex(const mcas_session_t session,
                                const char * pool_name,
                                const mcas_flags_t flags,
                                const void * base_addr);

  mcas_pool_t mcas_open_pool(const mcas_session_t session,
                             const char * pool_name,
                             const mcas_flags_t flags);


  /** 
   * Close a pool
   * 
   * @param session Session handle
   * @param pool Pool handle
   * 
   * @return 0 on success, < 0 on failure
   */
  int mcas_close_pool(const mcas_session_t session,
                      const mcas_pool_t pool);

    
#ifdef __cplusplus
}
#endif

#endif // __MCAS_CLIENT_H__
