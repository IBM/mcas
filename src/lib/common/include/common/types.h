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
  Copyright (C) 2014, Daniel G. Waddington <daniel.waddington@acm.org>
*/

#ifndef __COMMON_TYPES_H__
#define __COMMON_TYPES_H__

#include <common/common.h>
#include <stdint.h>
#ifdef __cplusplus
#include <cstddef>
#endif
#include <sys/uio.h>
#include <time.h>

typedef int status_t;

typedef unsigned long addr_t;   // physical or virtual address
typedef unsigned long saddr_t;  // storage address
typedef unsigned long umword_t;
typedef signed long mword_t;

typedef uint32_t addr32_t;
typedef unsigned long long int addr64_t;

/* Note: C++17 provides std::byte */
typedef unsigned char byte;
typedef uint16_t word;
typedef uint32_t dword;
typedef void *handle_t;
typedef uint32_t core_id_t;
typedef uint64_t atomic_t;
typedef uint64_t cpu_time_t; /* timestamp counter time */
typedef int numa_node_t;
typedef uint64_t offset_t;

#ifdef __cplusplus
#if __x86_64__
namespace common /* avoid conflict with EASTL definition of uint128_t */
{
typedef __uint128_t uint128_t;
}
#endif
#endif

#endif
