/*
   Copyright [2020] [IBM Corporation]

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.

   As a special exception, if you link the code in this file with
   files compiled with a GNU compiler to produce an executable,
   that does not cause the resulting executable to be covered by
   the GNU Lesser General Public License.  This exception does not
   however invalidate any other reasons why the executable file
   might be covered by the GNU Lesser General Public License.
   This exception applies to code released by its copyright holders
   in files containing the exception.
*/

/*
  Author(s):
  Copyright (C) 2020, Daniel G. Waddington <daniel.waddington@ibm.com>
*/

#include <common/cycles.h>
#include <common/time.h>
#include <boost/numeric/conversion/cast.hpp>

using namespace std::chrono;

const common::Timespec Epoch_nil = common::Timespec();

namespace common
{

/**
 * Subtract timespec X from Y
 *
 * @param x
 * @param y
 *
 * @return Result
 */
struct timespec timespec_subtract(const struct timespec& x,
				  const struct timespec& y)
{
  struct timespec result;
  if (y.tv_nsec < x.tv_nsec) {
    result.tv_sec  = y.tv_sec - x.tv_sec - 1;
    result.tv_nsec =
      (1000000000L - x.tv_nsec) + y.tv_nsec;
  }
  else {
    result.tv_sec  = y.tv_sec - x.tv_sec;
    result.tv_nsec = y.tv_nsec - x.tv_nsec;
  }
  return result;
}

/**
 * Add timespec X and Y
 *
 * @param time
 * @param add
 *
 * @return
 */
struct timespec timespec_add(const struct timespec& x,
			     const struct timespec& y)
{
  struct timespec result = x;

  result.tv_sec += y.tv_sec;
  result.tv_nsec += y.tv_nsec;

  while ( result.tv_nsec >= 1000000000L ) {
    result.tv_nsec -= 1000000000L;
    result.tv_sec++;
  }

  return result;
}

static common::epoch_time_t Tsc_initialize();

/**
 * Static declaration
 *
 */
namespace tsc_static
{
static uint64_t             _ticks_per_second;
static double               _ticks_per_nanosecond;
static common::epoch_time_t _epoch_at_power_on = Tsc_initialize();
}


Tsc::Tsc() : _tsc(rdtsc()) {
}

Tsc::Tsc(const common::epoch_time_t& epoch) : _tsc(0) {

  if(epoch.tv_sec == 0 && epoch.tv_nsec == 0) {
    _tsc = 0;
    return;
  }

  assert(tsc_static::_epoch_at_power_on.tv_sec > 0);
  assert(tsc_static::_epoch_at_power_on <= epoch);

  auto delta = timespec_subtract(tsc_static::_epoch_at_power_on, epoch);

  _tsc = (boost::numeric_cast<uint64_t>(delta.tv_sec) * tsc_static::_ticks_per_second) +
    boost::numeric_cast<uint64_t>(boost::numeric_cast<double>(delta.tv_nsec) *
				  tsc_static::_ticks_per_nanosecond);
}

unsigned long Tsc::raw() const {
  return _tsc;
}

void Tsc::update() {
  _tsc = rdtsc();
}


static common::epoch_time_t Tsc_initialize()
{
  /* establish time datum */
  float freq_MHZ = common::get_rdtsc_frequency_mhz();
  PLOG("Tsc: clock frequency %.2f mhz", freq_MHZ);

  tsc_static::_ticks_per_second = static_cast<uint64_t>(freq_MHZ * 1000000.0f);
  tsc_static::_ticks_per_nanosecond = static_cast<double>(freq_MHZ) / 1000.0f;

  auto tscA = rdtsc();
  struct timespec ts;
  if(clock_gettime(CLOCK_REALTIME, &ts) != 0)
    throw General_exception("clock_gettime failed");

  /* Note: an arbitrary period of time may elapse between time() and rdtsc().
   * Therefore users should not rely on epoch_now() in common/utils for a current
   * time consistent with timestamps used in mapstore. Rather, they could write
   * a kv pair and then retrieve that KV pair's timestamp.
   */
  auto tscB = rdtsc();
  auto nano_seconds_since_power_on = static_cast<double>((tscA + tscB) / 2) / tsc_static::_ticks_per_nanosecond;
  common::epoch_time_t tsc_ts(boost::numeric_cast<time_t>(nano_seconds_since_power_on / 1000000000.0),
			      boost::numeric_cast<long>(remainder(nano_seconds_since_power_on, 1000000000.0)));
  tsc_static::_epoch_at_power_on = timespec_subtract(tsc_ts, ts);
  assert(tsc_static::_epoch_at_power_on.seconds() > 0);

  return tsc_static::_epoch_at_power_on;
}

common::epoch_time_t Tsc::to_epoch() const {
  auto nano_seconds_since_power_on =
    static_cast<double>(_tsc) / tsc_static::_ticks_per_nanosecond;

  common::epoch_time_t tsc_ts
    (boost::numeric_cast<time_t>(nano_seconds_since_power_on / 1000000000.0),
     boost::numeric_cast<long>(remainder(nano_seconds_since_power_on, 1000000000.0)));

  assert(tsc_ts.tv_nsec < 1000000000L);

  return timespec_add(tsc_static::_epoch_at_power_on, tsc_ts);
}


constexpr nanoseconds timespec_to_duration(timespec ts)
{
    auto duration = seconds{ts.tv_sec} + nanoseconds{ts.tv_nsec};
    return duration_cast<nanoseconds>(duration);
}

/* Timepoint class */

Timepoint::Timepoint()
  : time_point_t(std::chrono::system_clock::now())
{
}

Timepoint::Timepoint(const common::epoch_time_t& epoch)
  : time_point_t(time_point<system_clock, nanoseconds>
   		 {duration_cast<system_clock::duration>(timespec_to_duration(epoch))})
{
}

unsigned long Timepoint::raw() const
{
  system_clock::duration dtn = this->time_since_epoch();
  return boost::numeric_cast<unsigned long>(dtn.count());
}

void Timepoint::update()
{
  time_point_t& tp = *this;
  tp = std::chrono::system_clock::now();
}

common::epoch_time_t Timepoint::to_epoch() const
{
  auto& tp = *this;
  auto secs = time_point_cast<seconds>(tp);
  auto ns = time_point_cast<nanoseconds>(tp) -
    time_point_cast<nanoseconds>(secs);

  return timespec{secs.time_since_epoch().count(), ns.count()};
}


/* Timespec class */
Timepoint Timespec::to_timepoint() const {
  Timepoint tp(*this);
  return tp;
}




} // common
