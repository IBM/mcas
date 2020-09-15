/*
   Copyright [2017-2020] [IBM Corporation]
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

#ifndef _EVENT_REGISTRATION_H_
#define _EVENT_REGISTRATION_H_

#include <common/delete_copy.h>

#include "rdma-fabric.h" /* fid_t */

struct event_consumer;
struct event_producer;
struct fid_ep;
struct fid_pep;

struct event_registration
{
private:
  event_producer &_ev;
  ::fid_t _ep;
public:
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_ep_bind fail
   */
  explicit event_registration(event_producer &ev, event_consumer &ec, ::fid_ep &ep);
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_pep_bind fail
   */
  explicit event_registration(event_producer &ev, event_consumer &ec, ::fid_pep &pep);
  DELETE_COPY(event_registration); /* _ep */
  event_registration(event_registration &&) noexcept;
  ~event_registration();
};

#endif
