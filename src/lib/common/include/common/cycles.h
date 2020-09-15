/*
   eXokernel Development Kit (XDK)

   Samsung Research America Copyright (C) 2013

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
  Authors:
  Copyright (C) 2016, Daniel G. Waddington <daniel.waddington@ibm.com>
  Copyright (C) 2013, Daniel G. Waddington <d.waddington@samsung.com>
  Copyright (C) 2015, Juan A. Colmenares <juan.col@samsung.com>
*/

#ifndef __CYCLES_H__
#define __CYCLES_H__

#include <common/common.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "types.h"

#define INLINE inline __attribute__((always_inline))

#if defined(__i386__) || defined(__amd64__)

#if defined(__amd64__)

/**
 * Reads the timestamp counter.
 * @return the read value.
 */
INLINE cpu_time_t rdtsc() {
  unsigned a, d;
  asm volatile("lfence");  // should be mfence for AMD
  asm volatile("rdtsc" : "=a"(a), "=d"(d));
  return (static_cast<unsigned long long>(a)) | ((static_cast<unsigned long long>(d)) << 32);
}

/**
 * Reads low 32 bits of the timestamp counter.
 * @return the read value.
 */
INLINE uint32_t rdtsc_low() {
  uint32_t a;
  asm volatile("lfence");           // should be mfence for AMD
  asm volatile("rdtsc" : "=a"(a));  //, "=d" (d));
  return a;
}

#if defined(__cplusplus)
/**
 * Returns the content of the timestamp counter (TSC) as well as the content of
 * the 32-bit Machine-Specific-Register MSR_TSC_AUX, which stores a value that
 * is unique to each logical processor.
 *
 * Linux kernels (since about 2.6.34) use the MSR_TSC_AUX register to store the
 * logical processor number and the socket number where that logical processor
 * is located (in multi-socket systems).
 *
 * @param[out] aux the content of MSR_TSC_AUX register.
 * @return the timestamp counter's value.
 */
INLINE
cpu_time_t rdtscp(uint32_t &aux) {
  unsigned a, d;
  asm volatile("lfence");  // should be mfence for AMD
  asm volatile("rdtscp" : "=a"(a), "=d"(d), "=c"(aux));
  return (static_cast<unsigned long long>(a)) | (static_cast<unsigned long long>(d) << 32);
  ;
}

/**
 * Returns a timestamp counter (TSC) read, the number of the socket where the
 * current processor is located, and the current processor's logical number.
 *
 * Linux kernels (since about 2.6.34) use the 32-bit MSR_TSC_AUX register to
 * store the logical processor number and (in multi-socket systems) the socket
 * number where that logical processor is located.
 * The value stored in MSR_TSC_AUX is unique to each logical processor.
 *
 * @param[out] socket_id the number of the socket for the current processor.
 * @param[out] cpu_id the current processor's logical number.
 * @return the timestamp counter's value.
 */
INLINE
cpu_time_t rdtscp(uint32_t &socket_id, uint32_t &cpu_id) {
  unsigned a, d, c;
  asm volatile("lfence");  // should be mfence for AMD
  asm volatile("rdtscp" : "=a"(a), "=d"(d), "=c"(c));
  socket_id = (c & 0xFFF000) >> 12;
  cpu_id = c & 0xFFF;
  return static_cast<unsigned long long>(a) | ((static_cast<unsigned long long>(d)) << 32);
  ;
}

#endif

#elif defined(__i386)

/**
 * Reads complete 40 bit counter into 64 bit value.
 * @return the read value.
 */
INLINE cpu_time_t rdtsc() {
  unsigned long long ret;
  asm volatile("lfence");  // should be mfence for AMD
  asm volatile("rdtsc" : "=A"(ret));
  return ret;
}

#endif

#elif defined(__powerpc64__)

#include <sys/platform/ppc.h>

INLINE cpu_time_t rdtsc() {
  return __ppc_get_timebase();
}

#else
#error Platform not supported.
#endif

namespace common
{
/**
 * Get RDTSC frequency in MHz.
 *
 *
 * @return Clock frequency in MHz
 */
float get_rdtsc_frequency_mhz();

/**
 * Convert cycles to usecs
 *
 * @param time Duration in usecs
 *
 * @return
 */
float cycles_to_usec(cpu_time_t time);

}  // namespace common

#undef INLINE

#endif
