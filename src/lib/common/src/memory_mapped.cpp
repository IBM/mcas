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
 */

#include <common/memory_mapped.h>

#include <common/logging.h>
#include <sys/mman.h>
#include <cstring>

common::memory_mapped::memory_mapped(void *vaddr, std::size_t size, int prot, int flags, int fd) noexcept
  : memory_mapped(
    ::mmap(vaddr, size, prot, flags, fd, 0), size
  )
{
}

common::memory_mapped::memory_mapped(void *vaddr, std::size_t size) noexcept
  : moveable_struct<::iovec, iovec_moveable_traits>(::iovec{vaddr, size})
{
  if (iov_base == MAP_FAILED) {
    iov_len = errno;
  }
}

common::memory_mapped::~memory_mapped()
{
  if ( iov_base != MAP_FAILED )
  {
    if ( ::munmap(iov_base, iov_len) != 0 )
    {
      auto e = errno;
      PLOG("%s: munmap(%p, %zu) failed: %s", __func__, iov_base, iov_len, ::strerror(e));
    }
  }
}
