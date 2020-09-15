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

#include "fabric_transport.h"

#include <common/cycles.h>

namespace mcas
{
namespace client
{
Fabric_transport::Fabric_transport(unsigned                   debug_level_,
                                   component::IFabric_client *fabric_connection,
                                   unsigned                   patience_)
    : common::log_source(debug_level_),
      cycles_per_second(common::get_rdtsc_frequency_mhz() * 1000000.0),
      _transport(fabric_connection),
      _max_inject_size(_transport->max_inject_size()),
      _bm(debug_level(), fabric_connection, NUM_BUFFERS),
      _patience(patience_)
{
}

component::IFabric_op_completer::cb_acceptance Fabric_transport::completion_callback(void *        context,
                                                                                     status_t      st,
                                                                                     std::uint64_t completion_flags,
                                                                                     std::size_t,  // len
                                                                                     void *,       // error_data
                                                                                     void *param)
{
  if (UNLIKELY(st != S_OK)) {
    // throw General_exception("poll_completions failed unexpectedly (st=%d)
    // (cf=%lx)", st, completion_flags);
    PWRN("poll_completions failed unexpectedly (context/got=%p) (param/wanted=%p) (st=%d) (cf=%lx)", context, param, st, completion_flags);
    return component::IFabric_op_completer::cb_acceptance::ACCEPT;
  }

  if (*(static_cast<void **>(param)) == context) {
    if (1 < mcas::global::debug_level) {
      PLOG("COMPletion %p(wanted/got) accept", context);
    }
    *static_cast<void **>(param) = nullptr; /* signals completion */
    return component::IFabric_op_completer::cb_acceptance::ACCEPT;
  }
  else {
    if (1 < mcas::global::debug_level) {
      PLOG("COMPletion got %p(want) != %p(got)  defer", *static_cast<void **>(param), context);
    }
    return component::IFabric_op_completer::cb_acceptance::DEFER;
  }
}

/**
 * Wait for completion of a IO buffer posting
 *
 * @param iob IO buffer to wait for completion of
 */
void Fabric_transport::wait_for_completion(void *wr)
{
  CPLOG(1, "%s %p:%p", __func__, wr, *static_cast<void **>(wr));

  auto start_time = rdtsc();
  // currently setting time out to 2 min...
  while (wr && static_cast<double>(rdtsc() - start_time) / cycles_per_second <= _patience) {
    _transport->poll_completions_tentative(completion_callback, &wr);
  }

  if (wr)
    throw Program_exception("time out: start_time %" PRIu64 ", now %" PRIu64 " waited %lu seconds for completion",
      start_time, rdtsc(), _patience);
}

}  // namespace client
}  // namespace mcas
