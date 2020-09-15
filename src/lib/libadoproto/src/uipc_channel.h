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



/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __CORE_UIPC_CHANNEL_H__
#define __CORE_UIPC_CHANNEL_H__

#include <common/mpmc_bounded_queue.h>
#include <common/spsc_bounded_queue.h>
#include <common/logging.h>
#include <memory>
#include <string>

struct uipc_channel
{
  virtual ~uipc_channel() {}
};

namespace core
{
namespace uipc
{

class Shared_memory;

class Channel : public uipc_channel, private common::log_source {
  /* we use the non-sleeping queue for the moment,
     with the ADO thread sleeping when the queue
     is empty. SPSC is OK since there is one shard
     thread and one ADO receiver thread.
  */
  using queue_t = common::Spsc_bounded_lfq<void*>;

  /* except that the _slab_ring is used by both
   * shard and mcas, making it an mpmc queue
   */
  using mqueue_t = common::Mpmc_bounded_lfq<void*>;

 public:
  Channel(const Channel &) = delete;

  Channel& operator=(const Channel &) = delete;

  /**
   * Master-side constructor
   *
   * @param name Name of channel
   * @param message_size Size of messages in bytes
   * @param queue_size Max elements in FIFO
   */
  Channel(const std::string &name, size_t message_size, size_t queue_size);

  /**
   * Slave-side constructor
   *
   * @param name Name of channel
   * @param message_size Size of messages in bytes
   * @param queue_size Max elements in FIFO
   */
  Channel(const std::string &name);

  /**
   * Destructor
   *
   *
   * @return
   */
  virtual ~Channel();

  /**
   * Post message onto channel
   *
   * @param msg Message pointer
   *
   * @return S_OK or E_FULL
   */
  status_t send(void* msg);

  /**
   * Receive message from channel
   *
   * @param out_msg Out message
   *
   * @return S_OK or E_EMPTY
   */
  status_t recv(void*& recvd_msg);

  /**
   * Allocate message (in shared memory) for
   * exchange on channel
   *
   *
   * @return Pointer to message
   */
  void* alloc_msg();

  /**
   * Free message allocated with alloc_msg
   *
   *
   * @return S_OK or E_INVAL
   */
  status_t free_msg(void*);

  /**
   * Used to unblock a thread waiting on a recv
   *
   */
  void unblock_threads();

  /**
   * Shutdown handling
   *
   */
  void set_shutdown() { _shutdown = true; }

  /**
   * Determine if shutdown in progress
   *
   *
   * @return True if so
   */
  bool shutdown() const { return _shutdown; }

 private:
  void initialize_data_structures();

 private:
  bool _shutdown = false;
  bool _master;
  std::string _name;
  long _slab_ring_net;
  std::unique_ptr<Shared_memory> _shmem_fifo_m2s;
  std::unique_ptr<Shared_memory> _shmem_fifo_s2m;
  std::unique_ptr<Shared_memory> _shmem_slab_ring;
  std::unique_ptr<Shared_memory> _shmem_slab;

  queue_t* _in_queue;
  queue_t* _out_queue;
  mqueue_t* _slab_ring;
};

}  // namespace uipc
}  // namespace core

#endif
