
#ifndef __ADO_MANAGER_H__
#define __ADO_MANAGER_H__

#include <common/logging.h> /* log_source */
#include <common/types.h>
#include <threadipc/queue.h>

#include <set>
#include <future>
#include <utility>
#include <vector>

#include "config_file.h"
#include "program_options.h"

class SLA;

struct ado {
  unsigned int shard_id;
  std::string  cpus;
  // can be container id
  std::string id;
};

struct compare {
  bool operator()(const std::pair<unsigned int, unsigned int> &lhs,
                  const std::pair<unsigned int, unsigned int> &rhs) const
  {
    if (lhs.second == rhs.second) return (lhs.first < rhs.first);
    return (lhs.second < rhs.second);
  }
};

class ADO_manager : private common::log_source {
private:
  mcas::Config_file         _config;

public:
  ADO_manager(Program_options &options)
    : log_source(0),
      _config(options.debug_level, options.config),
      _ados{},
      _ado_cpu_pool{},
      _manager_cpu_pool{},
      _thread(std::async(std::launch::async, &ADO_manager::init, this))
  {
    while (!_running) usleep(1000);
    _sla = NULL;
  }
  ADO_manager(const ADO_manager &) = delete;
  ADO_manager &operator=(const ADO_manager &) = delete;
  ~ADO_manager() { exit(); }
  void setSLA();

private:
  static constexpr const char *_cname = "ADO_manager";
  SLA *                   _sla = NULL;
  std::vector<struct ado> _ados;
  // std::set<std::pair<unsigned int, unsigned int>, compare> _ado_cpu_pool;
  std::set<unsigned int> _ado_cpu_pool;
  std::set<unsigned int> _manager_cpu_pool;
  bool                   _running = false;
  bool                   _exit    = false;
  std::future<void>      _thread;

  void init();
  void main_loop();

  void exit();

  void resource_check()
  {
    // TODO
  }
  void        register_ado(unsigned int, std::string &, std::string &);
  void        kill_ado(const struct ado &ado);
  void        schedule(unsigned int shard_core, std::string cpus, float cpu_num, numa_node_t numa_zone);
  std::string reschedule(const struct ado &ado)
  {
    // TODO
    return ado.cpus;
  }
};

#endif
