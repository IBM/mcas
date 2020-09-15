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

#include "lazy_region.h"

#if (__SIZEOF_POINTER__ == 8)
static constexpr auto PREFERRED_VADDR = 0xBB00000000ULL;
#else
static constexpr auto PREFERRED_VADDR = 0xBB000000UL;
#endif

addr_t core::slab::Lazily_extending_region::addr_hint = PREFERRED_VADDR;
