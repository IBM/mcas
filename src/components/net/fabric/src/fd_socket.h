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


#ifndef _FD_SOCKET_H_
#define _FD_SOCKET_H_

#include <common/fd_open.h>

#include <cstddef> /* size_t */

class Fd_socket
  : public common::Fd_open
{
public:
  Fd_socket();
  /*
   * @throw std::logic_error : initialized with a negative value
   */
  explicit Fd_socket(int fd_);
  Fd_socket(Fd_socket &&) noexcept = default;
  Fd_socket &operator=(Fd_socket &&) noexcept = default;
  /*
   * @throw std::system_error - sending data on socket
   */
  void send(const void *buf, std::size_t size) const;
  /*
   * @throw std::system_error - receiving data on socket
   */
  void recv(void *buf, std::size_t size) const;
};

#endif
