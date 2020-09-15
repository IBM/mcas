/*
   Copyright [2019-2020] [IBM Corporation]
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

#include "as_extend.h"

#include "hstore_config.h"
#include "as_emplace.h"
#include "logging.h"

/*
 * ===== allocation_state_extend =====
 */

impl::allocation_state_extend::allocation_state_extend()
	: _armed(false)
	, _ptr(nullptr)
	, _psegment_count(nullptr)
	, _segment_count_updated_value(0)
{}

impl::allocation_state_extend::allocation_state_extend(allocation_state_extend &&m_) noexcept
	: _armed(std::move(m_._armed))
	, _ptr(m_._ptr.load())
	, _psegment_count(std::move(m_._psegment_count))
	, _segment_count_updated_value(std::move(m_._segment_count_updated_value))
{}

bool impl::allocation_state_extend::is_in_use(const void *const ptr_)
{
	auto in_use =
		ptr_ != nullptr
#if USE_CC_HEAP == 4
		&&
		ptr_ == _ptr
		&&
		_psegment_count != nullptr && ( _psegment_count->actual().is_stable() && _psegment_count->actual().value()  == _segment_count_updated_value )
#endif
		;
	PLOG(PREFIX "ptr %p, extend in use %s", LOCATION, ptr_, in_use ? "true" : "false");
	return in_use;
}
