#ifndef __QUEUE_H_
#define __QUEUE_H_

#include <common/types.h>
#include <common/semaphore.h>
#include <stdlib.h> /* aligned_alloc */
#include <mutex>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Weffc++"

#if __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wclass-memaccess"  // compiler version check needed
#endif
#include <tbb/atomic.h> /* must include this */
#include <tbb/concurrent_queue.h>
#pragma GCC diagnostic pop


namespace threadipc
{

static constexpr size_t QUEUE_SIZE = 8;

enum class Operation { kill, schedule, reschedule, register_ado, wait };

struct Message {
  unsigned    shard_core = 0;
  Operation   op         = Operation::wait;
  std::string cores;
  float       core_number = -1;
  std::string ado_id;
  numa_node_t numa_zone = 0;
  Message() : cores(std::string()), ado_id(std::string()) {}
};


class Thread_ipc {

  using queue_type = tbb::concurrent_queue<Message*>;  //common::Spsc_lfq_sleeping<message *>;


 public:
  static Thread_ipc *instance();
  Thread_ipc(Thread_ipc const &) = delete;  // copy constructor deleted
  Thread_ipc &operator=(const Thread_ipc &) = delete;
  void schedule_to_mgr(unsigned int shard_core, std::string &cores, float core_number, numa_node_t numa_zone);
  void register_to_mgr(unsigned int shard_core, std::string &cores, std::string id);
  void schedule_to_ado(unsigned int shard_core, const std::string &cores, float core_number, numa_node_t numa_zone);
  void send_kill_to_ado(unsigned int shard_core);
  void get_next_ado(struct Message *&msg);
  void get_next_mgr(struct Message *&msg);
  void add_to_ado(struct Message *msg);
  void add_to_mgr(struct Message *msg);
  void cleanup();
  virtual ~Thread_ipc() { cleanup(); }
  /* pre-C++17, provide help to align instance of Thread_ipc */
#if __cplusplus < 201703L
  void *operator new(std::size_t sz) { return aligned_alloc(alignof(Thread_ipc), sz); }
#endif

 private:
  Thread_ipc();

  common::Semaphore  _queue_to_ado_sem;
  queue_type         _queue_to_ado;
  common::Semaphore  _queue_to_mgr_sem;
  queue_type         _queue_to_mgr;

  static std::mutex  _ipc_mutex;
  static Thread_ipc *_ipc;
};
}  // namespace threadipc

#endif
