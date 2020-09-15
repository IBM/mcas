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
#include "mr_lock.h"

#include "eyecatcher.h"
#include <sys/mman.h> /* mlock, munlock */
#include <exception>
#include <iostream>
#include <string>
#include <system_error>

mr_lock::mr_lock(const void *addr_, std::size_t len_)
  : _addr(addr_)
  , _len(len_)
{
  if ( 0 != ::mlock(_addr, _len) )
  {
    auto e = errno;
    auto context = std::string("in ") + __func__;
    throw std::system_error{std::error_code{e, std::system_category()}, context};
  }
}

mr_lock::~mr_lock()
try
{
  if ( _addr )
  {
    ::munlock(_addr, _len);
  }
}
catch ( std::exception &e )
{
  std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
}
