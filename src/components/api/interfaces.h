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
#ifndef __COMANCHE_INTERFACES_H__
#define __COMANCHE_INTERFACES_H__

#include <component/base.h>
/* Note: these uuid decls are so we don't have to have access to the component
 * source code */

#include "itf_ref.h"
#include "types.h"

namespace interface
{
/* static interfaces, used for factory-less instantiation */
DECLARE_STATIC_INTERFACE_UUID(ado_plugin, 0x59564581, 0x9e1b, 0x4811, 0xbdb2, 0x19, 0x57, 0xa0, 0xa6, 0x84, 0x57);

}  // namespace interface

namespace Interface = ::interface;

#endif  // __COMPONENTS_H__
