/*
   Copyright [2019-2020] [IBM Corporation]
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
#ifndef MCAS_REGISTERED_MEMORY_H
#define MCAS_REGISTERED_MEMORY_H

#include <common/logging.h>
#include <common/moveable_ptr.h>

#include "mcas_config.h"
#include <cstdint> /* uint64_t, size_t */
#include <utility> /* swap */

namespace mcas
{
template <typename T>
struct memory_registered : private common::log_source {
private:
  common::moveable_ptr<T>     _t;
  typename T::memory_region_t _r;

 public:
  explicit memory_registered(unsigned      debug_level_,
                             T *           transport_,
                             const void *  base_,
                             std::size_t   len_,
                             std::uint64_t key_,
                             std::uint64_t flags_)
      : common::log_source(debug_level_),
        _t(transport_),
        _r(_t->register_memory(base_, len_, key_, flags_))
  {
    CPLOG(2, "%s %p (%p:0x%zx)", __func__, static_cast<const void *>(_r), base_, len_);
  }

  memory_registered(const memory_registered &) = delete;
  memory_registered(memory_registered &&other_) noexcept = default;
  memory_registered &operator=(const memory_registered) = delete;

  ~memory_registered()
  {
    if (_t) {
      CPLOG(2, "%s %p", __func__, static_cast<const void *>(_r));
      _t->deregister_memory(_r);
    }
  }
  auto mr() const { return _r; }
#if 0
  auto transport() const { return static_cast<void *>(_t); }
#else
  T *transport() const { return _t; }
#endif
  auto desc() const { return _t->get_memory_descriptor(_r); }
  auto key() const { return _t->get_memory_remote_key(_r); }
  auto get_memory_descriptor() const { return desc(); }
};

template <typename T>
  bool operator<(const memory_registered<T> &a, const memory_registered<T> &b)
  {
    return a.transport() < b.transport() || ( a.transport() == b.transport() && a.mr() < b.mr() );
  }

template <typename T>
memory_registered<T> make_memory_registered(unsigned      debug_level_,
                                            T *           transport_,
                                            void *        base_,
                                            std::size_t   len_,
                                            std::uint64_t key_,
                                            std::uint64_t flags_)
{
  return memory_registered<T>(debug_level_, transport_, base_, len_, key_, flags_);
}
}  // namespace mcas

#endif
