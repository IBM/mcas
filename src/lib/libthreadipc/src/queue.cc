#include <threadipc/queue.h>
#include <common/utils.h>

namespace threadipc
{
// Global static pointer used to ensure a single instance of the class.
std::mutex  Thread_ipc::_ipc_mutex;
Thread_ipc *Thread_ipc::_ipc = nullptr;

/** This function is called to create an instance of the class.
    Calling the constructor publicly is not allowed. The constructor
    is private and is only called by this instance function.
*/

Thread_ipc::Thread_ipc() :
  _queue_to_ado_sem(),
  _queue_to_ado(),
  _queue_to_mgr_sem(),
  _queue_to_mgr()
{
}

Thread_ipc *Thread_ipc::instance()
{
  std::lock_guard<std::mutex> g{Thread_ipc::_ipc_mutex};
  if (!_ipc) {  // Only allow one instance of class to be generated.
    /* Thread_ipc alignment, once handled here, is now handled by Thread_ipc
     * class */
    _ipc = new Thread_ipc();
  }
  return _ipc;
}

void Thread_ipc::schedule_to_mgr(unsigned int shard_core, std::string &cores, float core_number, numa_node_t numa_zone)
{
  Message *msg     = new Message; // TO DO, use preallocated list
  msg->shard_core  = shard_core;
  msg->op          = Operation::schedule;
  msg->cores       = cores;
  msg->core_number = core_number;
  msg->numa_zone   = numa_zone;

  _queue_to_mgr.push(msg);
  _queue_to_mgr_sem.post();
}


void Thread_ipc::register_to_mgr(unsigned int shard_core, std::string &cores, std::string id)
{
  Message *msg    = new Message; // TO DO, use preallocated list
  msg->shard_core = shard_core;
  msg->op         = Operation::register_ado;
  msg->cores      = cores;
  msg->ado_id     = id;

  _queue_to_mgr.push(msg);
  _queue_to_mgr_sem.post();
}

void Thread_ipc::schedule_to_ado(unsigned int       shard_core,
                                 const std::string &cores,
                                 float              core_number,
                                 numa_node_t        numa_zone)
{
  Message *msg     = new Message;
  msg->shard_core  = shard_core;
  msg->op          = Operation::schedule;
  msg->cores       = cores;
  msg->core_number = core_number;
  msg->numa_zone   = numa_zone;
  _queue_to_ado.push(msg);
  _queue_to_ado_sem.post();
}

void Thread_ipc::send_kill_to_ado(unsigned int shard_core)
{
  Message *msg    = new Message;
  msg->shard_core = shard_core;
  msg->op         = Operation::kill;

  _queue_to_ado.push(msg);
  _queue_to_ado_sem.post();
}

void Thread_ipc::get_next_ado(struct Message*& msg)
{
  _queue_to_ado_sem.wait();
  while(!_queue_to_ado.try_pop(msg)) cpu_relax();
}

void Thread_ipc::get_next_mgr(struct Message *&msg)
{
  _queue_to_mgr_sem.wait();
  int max = 10000;
  while(!_queue_to_mgr.try_pop(msg)) {
    cpu_relax();
    if(max-- == 0) {
      throw std::runtime_error("Thread_ipc timeout");
    }
  }
}

void Thread_ipc::add_to_ado(struct Message *msg)
{
  _queue_to_ado.push(msg);
  _queue_to_ado_sem.post();
}

void Thread_ipc::add_to_mgr(struct Message *msg)
{
  _queue_to_mgr.push(msg);
  _queue_to_mgr_sem.post();
}

void Thread_ipc::cleanup()
{
  _queue_to_ado_sem.post();

  while(_queue_to_ado.unsafe_size() > 0) {
    struct Message *msg = nullptr;
    while(_queue_to_ado.try_pop(msg)) {
      assert(msg);
      delete msg;
    }
  }

  _queue_to_mgr_sem.post();

  while(_queue_to_mgr.unsafe_size() > 0) {
    struct Message *msg = nullptr;
    while(_queue_to_mgr.try_pop(msg)) {
      assert(msg);
      delete msg;
    }
  }
}


}  // namespace threadipc
