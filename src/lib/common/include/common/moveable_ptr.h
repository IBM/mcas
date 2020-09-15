/*
   Copyright [2020] [IBM Corporation]
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

#ifndef _MCAS_COMMON_MOVEABLE_PTR_
#define _MCAS_COMMON_MOVEABLE_PTR_

#include <common/moveable_value.h>

namespace common
{
  /* A non-owning pointer which zeroes the source when moved.
   * helpful for classes which use a null pointer to denote
   * "moved from" state.
   */
  template <typename T>
    struct moveable_ptr
      : public moveable_value<T *>
    {
      moveable_ptr(T *p_)
        : moveable_value<T *>(p_)
      {}

      moveable_ptr(moveable_ptr &&o_) noexcept = default;
      moveable_ptr &operator=(moveable_ptr &&o_) noexcept = default;

      operator T*() const { return this->v; }
      T &operator*() const { return *this->v; }
      T *operator->() const { return this->v; }
    };

  template<>
    struct moveable_ptr<void>
      : public moveable_value<void *>
    {
      moveable_ptr(void *p_)
        : moveable_value<void *>(p_)
      {}

      moveable_ptr(moveable_ptr &&o_) noexcept = default;
      moveable_ptr &operator=(moveable_ptr &&o_) noexcept = default;

      operator void*() const { return this->v; }
    };
}
#endif
