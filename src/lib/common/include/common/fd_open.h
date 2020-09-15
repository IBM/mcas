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

#ifndef _MCAS_FD_OPEN_H_
#define _MCAS_FD_OPEN_H_

#include <common/common.h>
#include <common/moveable_value.h>

namespace common
{
  class Fd_open
  {
    moveable_value<int, -1> _fd;
    void close() noexcept;
  public:
    Fd_open();
    /*
     * @throw std::logic_error : initialized with a negative value
     */
    explicit Fd_open(int fd);

    ~Fd_open();
    Fd_open(Fd_open &&) noexcept = default;
    Fd_open &operator=(Fd_open &&) noexcept = default;
    int fd() const { return _fd; }
    bool good() const { return _fd != -1; }
    operator bool() const { return good(); }
  };
}

#endif
