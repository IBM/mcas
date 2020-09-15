#ifndef __EXP_IOPS_H__
#define __EXP_IOPS_H__

#include "task.h"

#include <pthread.h>
#include "statistics.h"

#include <chrono>
#include <cstdlib>

class Experiment_IOPS : public core::Tasklet
{
public:

  Experiment_IOPS(ProgramOptions options)
  {
    assert(options.factory);
    _store.reset(options.factory->create(options.debug_level, options.owner, options.server_address, options.device_name));
  }

  void initialize(unsigned core) override
  {
  }

  void do_work(unsigned core) override
  {
    static bool first_iter = true;
    // handle first time setup
    if(first_iter) {
      PLOG("[%u] Starting IOPS experiment...", core);
      first_iter = false;
      _start_time = std::chrono::high_resolution_clock::now();
    }
    pthread_exit(nullptr);

    // // end experiment if we've reached the total number of components
    // if (_i == _pool_num_components)
    //   {
    //     timer.stop();
    //     PINF("[%u] put: reached total number of components. Exiting.", core);
    //     throw std::exception();
    //   }

    // // check time it takes to complete a single put operation
    // unsigned int cycles, start, end;
    // int rc;

    // timer.start();
    // start = rdtsc();
    // try
    //   {
    //     rc = _store->put(_pool, _data->key(_i), _data->value(_i), _data->value_len());
    //   }
    // catch(...)
    //   {
    //     PERR("put call failed! Returned %d. Ending experiment.", rc);
    //     throw std::exception();
    //   }
    // end = rdtsc();
    // timer.stop();

    // cycles = end - start;
    // double time = (cycles / _cycles_per_second);
    // //printf("start: %u  end: %u  cycles: %u seconds: %f\n", start, end, cycles, time);

    // std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    // auto time_since_start = std::chrono::duration<double>(end_time - _exp_start_time).count();

    // // store the information for later use
    // _start_time.push_back(time_since_start);
    // _latencies.push_back(time);

    // _latency_stats.update(time);

    // _i++;  // increment after running so all elements get used

    // _enforce_maximum_pool_size(core, _i);

    // if (rc != S_OK)
    //   {
    //     timer.stop();
    //     perror("put returned !S_OK value");
    //     throw std::exception();
    //   }
  }

  void cleanup(unsigned core)
  {
    _end_time = std::chrono::high_resolution_clock::now();
    _store.reset(nullptr);
  }

private:
  component::Itf_ref<component::IKVStore> _store;
  std::chrono::high_resolution_clock::time_point _start_time,_end_time;
};


#endif //  __EXP_IOPS_H__
