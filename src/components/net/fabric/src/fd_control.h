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


#ifndef _FD_CONTROL_H_
#define _FD_CONTROL_H_

#include "fd_socket.h"

#include "fabric_types.h" /* addr_ep_t */

#include <cstdint>
#include <string>

class Fd_control
  : public Fd_socket
{
public:
  Fd_control();
  /*
   * @throw std::logic_error : socket initialized with a negative value
   */
  explicit Fd_control(int fd_);
  /*
   * @throw std::logic_error : socket initialized with a negative value (from ::socket)
   * @throw std::system_error : resolving address
   */
  explicit Fd_control(std::string dst_addr, std::uint16_t port);
  Fd_control(Fd_control &&) noexcept = default;
  Fd_control &operator=(Fd_control &&) noexcept = default;
  /*
   * @throw std::system_error - sending data on socket
   */
  void send_name(const fabric_types::addr_ep_t &name) const;
  /*
   * @throw std::system_error - receiving data on socket
   */
  fabric_types::addr_ep_t recv_name() const;
};

#endif
