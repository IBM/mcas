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
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */
#ifndef __API_SAMPLE_ITF__
#define __API_SAMPLE_ITF__

#include <api/components.h>
#include <common/types.h>

namespace component
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

/**
 * ISample - demo interface
 */
class ISample : public component::IBase {
 public:
  /* generate from uuidgen command line tool - do not reuse this one */
  DECLARE_INTERFACE_UUID(0xbeb21d37, 0x503c, 0x4643, 0xafa5, 0x40, 0x5f, 0xf6, 0x4c, 0x90, 0xc1);

 public:
  /* example method */
  virtual void say_hello() = 0;
};

class ISample_factory : public component::IBase {
 public:
  /* generate from uuidgen command line tool or derive from above - do not reuse
   * this one */
  DECLARE_INTERFACE_UUID(0xfac21d37, 0x503c, 0x4643, 0xafa5, 0x40, 0x5f, 0xf6, 0x4c, 0x90, 0xc1);

  /* simple factory instance creation point */
  virtual ISample* open(std::string name) = 0;
};
#pragma GCC diagnostic pop

}  // namespace component

#endif
