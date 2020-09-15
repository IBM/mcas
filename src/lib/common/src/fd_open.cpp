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
 *
 */

#include <common/fd_open.h>

#include <unistd.h> /* close */

#include <stdexcept>

common::Fd_open::Fd_open()
  : _fd(-1)
{}

common::Fd_open::Fd_open(int fd_)
  : _fd(fd_)
{
  if ( _fd < 0 )
  {
    throw std::logic_error(std::string(__func__) + ": negative fd");
  }
}

common::Fd_open::~Fd_open()
{
  close();
}

void common::Fd_open::close() noexcept
{
  if ( _fd != -1 )
  {
    ::close(_fd);
  }
}
