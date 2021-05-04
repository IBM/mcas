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
  Copyright (C) 2017, Daniel G. Waddington <daniel.waddington@ibm.com>
*/

#ifndef __COMMON_STACK_H__
#define __COMMON_STACK_H__

#include <common/common.h>
#include <common/exceptions.h>
#include <stdlib.h>

#include "mm_wrapper.h"

namespace real
{
extern malloc_function_t        malloc;
extern free_function_t          free;
}

namespace common
{
template <typename T, size_t STACK_SIZE = 10000000>
class Fixed_stack {
 private:
  T *_base;
  T *_top;
  T *_max;

 public:
  Fixed_stack()
    : _base(static_cast<T *>(real::malloc(sizeof(T) * STACK_SIZE)))
    , _top(_base)
    , _max(_base + STACK_SIZE)
  {
    assert(_base);
  }

  Fixed_stack(const Fixed_stack &) = delete;
  Fixed_stack &operator=(const Fixed_stack &) = delete;

  ~Fixed_stack() { real::free(_base); }

  void push(T &val) {
    if (_top == _max) {
      throw General_exception("stack overflow!");
    }

    *_top = val;
    _top++;
  }

  T pop() {
    if (_top == _base) {
      throw General_exception("stack empty!");
    }
    _top--;
    T rval = *_top;
    *_top = 0;
    return rval;
  }

  bool empty() const { return _base == _top; }
};
}  // namespace common

#endif  // _COMMON_STACK_H__
