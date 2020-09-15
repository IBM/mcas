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

#include <algorithm> /* swap */

#ifndef _MCAS_COMMON_MOVEABLE_VALUE_
#define _MCAS_COMMON_MOVEABLE_VALUE_

namespace common
{
  /* A value which zero-initializes the source when moved.
   * helpful for classes which use a pointer or bool to denote
   * "moved from" state.
   */
  template <typename T, T None = T()>
    struct moveable_value
    {
      T v;
      moveable_value(T v_)
        : v(v_)
      {}
      moveable_value(moveable_value &&o_) noexcept
        : v(None)
      {
        using std::swap;
        swap(v, o_.v);
      }

      moveable_value &operator=(moveable_value &&o_) noexcept
      {
        using std::swap;
        swap(v, o_.v);
        return *this;
      }

      operator T() const { return v; }
    };

}
#endif
