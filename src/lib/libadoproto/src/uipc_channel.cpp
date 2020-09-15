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
#include "uipc_channel.h"

#include "resource_unavailable.h"
#include "uipc_shared_memory.h"
#include <common/errors.h>
#include <common/exceptions.h>

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

#include <unistd.h>
#include <cassert>
#include <string>

namespace core
{
namespace uipc
{

#if 0
typedef common::Spsc_bounded_lfq_sleeping<
  Dataplane::Command_t, 32 /* queue size */
> command_queue_t;
#endif

Channel::Channel(const std::string &name, size_t message_size, size_t queue_size)
  : common::log_source(1)
  , _master(true)
  , _name(name)
  , _slab_ring_net(0)
  , _shmem_fifo_m2s()
  , _shmem_fifo_s2m()
  , _shmem_slab_ring()
  , _shmem_slab()
  , _in_queue(nullptr)
  , _out_queue(nullptr)
  , _slab_ring(nullptr) {
  const size_t queue_footprint = queue_t::memory_footprint(queue_size);
  size_t pages_per_queue = round_up(queue_footprint, PAGE_SIZE) / PAGE_SIZE;

  assert((queue_size != 0) && ((queue_size & (~queue_size + 1)) ==
                               queue_size));  // queue len is a power of 2
  assert(message_size % 8 == 0);

  CPLOG(1, "pages per FIFO queue: %zu", pages_per_queue);

  /* Buffer usage:
   * Primary use of the buffers is in two queues: m2s and s2m.
   * A few buffers will be allocated for threads which acquite a buffer in order
   * to complete an operation without blocking for resources (message processing,
   * shutdown).
   * Allocate 2x the buffer (m2s, s2m) count. Not quite enough to fill both buffers
   * due to the other uses.
   */
  const size_t slab_multiplier = 2;

  const size_t slab_queue_pages =
      round_up(mqueue_t::memory_footprint(queue_size * slab_multiplier), PAGE_SIZE) / PAGE_SIZE;

  CPLOG(1, "slab_queue_pages: %ld", slab_queue_pages);

  const size_t slab_pages =
      round_up(message_size * queue_size * slab_multiplier, PAGE_SIZE) / PAGE_SIZE;

  CPLOG(1, "slab_pages: %ld", slab_pages);

  const size_t total_pages = ((pages_per_queue * 2) + slab_queue_pages + slab_pages);

  CPLOG(1, "total_pages: %ld", total_pages);

  _shmem_fifo_m2s = std::make_unique<Shared_memory>(name + "-m2s", pages_per_queue);
  _shmem_fifo_s2m = std::make_unique<Shared_memory>(name + "-s2m", pages_per_queue);
  _shmem_slab_ring = std::make_unique<Shared_memory>(name + "-slabring", slab_queue_pages);
  _shmem_slab = std::make_unique<Shared_memory>(name + "-slab", slab_pages);

  _out_queue = new (_shmem_fifo_m2s->get_addr()) queue_t(
      queue_size, (static_cast<char*>(_shmem_fifo_m2s->get_addr())) + sizeof(queue_t));

  _in_queue = new (_shmem_fifo_s2m->get_addr()) queue_t(
      queue_size, (static_cast<char*>(_shmem_fifo_s2m->get_addr())) + sizeof(queue_t));

  size_t slab_slots = queue_size * slab_multiplier;
  _slab_ring = new (_shmem_slab_ring->get_addr()) mqueue_t(
      slab_slots, (static_cast<char*>(_shmem_slab_ring->get_addr())) + sizeof(mqueue_t));
  byte* slot_addr = static_cast<byte*>(_shmem_slab->get_addr());

  CPLOG(1, "buffers for %s", name.c_str());

  for (size_t i = 0; i < slab_slots; i++) {
    CPLOG(1, "  at %p", static_cast<const void*>(slot_addr));

    if ( ! _slab_ring->enqueue(slot_addr) )
    {
      throw std::runtime_error("failed to populate slab_ring");
    }
    slot_addr += message_size;
  }
}

Channel::Channel(const std::string &name)
  : common::log_source(1)
  , _master(false)
  , _name(name)
  , _slab_ring_net(0)
  , _shmem_fifo_m2s(std::make_unique<Shared_memory>(name + "-m2s"))
  , _shmem_fifo_s2m(std::make_unique<Shared_memory>(name + "-s2m"))
  , _shmem_slab_ring(std::make_unique<Shared_memory>(name + "-slabring"))
  , _shmem_slab(std::make_unique<Shared_memory>(name + "-slab"))
  , _in_queue(reinterpret_cast<queue_t*>(_shmem_fifo_m2s->get_addr()))
  , _out_queue(reinterpret_cast<queue_t*>(_shmem_fifo_s2m->get_addr()))
  , _slab_ring(reinterpret_cast<mqueue_t*>(_shmem_slab_ring->get_addr())) {

  CPLOG(1, "got fifo (m2s) @ %p - %lu bytes", _shmem_fifo_m2s->get_addr(),
        _shmem_fifo_m2s->get_size());

  CPLOG(1, "got fifo (s2m) @ %p - %lu bytes", _shmem_fifo_s2m->get_addr(),
        _shmem_fifo_s2m->get_size());

  CPLOG(1, "got slab ring @ %p - %lu bytes", _shmem_slab_ring->get_addr(),
        _shmem_slab_ring->get_size());

  CPLOG(1, "got slab @ %p - %lu bytes", _shmem_slab->get_addr(),
        _shmem_slab->get_size());

  ::usleep(500000); /* TODO hack to let master get ready - could improve with state in
                       shared memory */
}

Channel::~Channel() {
  /* don't delete queues since they were constructed on shared memory */
  CPLOG(1, "Channel %s/%s slab_ring net %ld", _name.c_str(), _master ? "master" : "slave", _slab_ring_net);
}

status_t Channel::send(void* msg) {
  assert(_out_queue);
  if (_out_queue->enqueue(msg)) {
    return S_OK;
  }
  else {
    return E_FULL;
  }
}

status_t Channel::recv(void*& recvd_msg) {
  assert(_in_queue);
  if (_in_queue->dequeue(recvd_msg))
    return S_OK;
  else
    return E_EMPTY;
}

void Channel::unblock_threads() { _in_queue->exit_threads(); }

void* Channel::alloc_msg() {
  assert(_slab_ring);
  void* msg = nullptr;
  auto st = _slab_ring->dequeue(msg);
  assert( bool(msg) == st );
  if ( ! st )
  {
    std::ostringstream o;
    o << "channel '" << _name << "' " << (_master ? "master" : "slave") << " net " << _slab_ring_net << " out of slots";
    throw resource_unavailable(o.str());
  }
  --_slab_ring_net;
  return msg;
}

status_t Channel::free_msg(void* msg) {
  assert(_slab_ring);
  assert(msg);
  /* currently no checks for validity */
  auto st = _slab_ring->enqueue(msg);
  if ( st )
  {
    ++_slab_ring_net;
  }
  return st ? S_OK : E_FULL;
}

}  // namespace uipc
}  // namespace core
