/*
   Copyright [2019] [IBM Corporation]
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

#ifndef _MCAS_HSTORE_ALLOCATION_STATE_EXTEND_H
#define _MCAS_HSTORE_ALLOCATION_STATE_EXTEND_H

#include "hstore_config.h"
#include "segment_count.h" /* segment_count_actual_t */
#include <array>
#include <atomic>

namespace impl
{
	/* Client-side persistent allocation state when using the crash-consistent
	 * allocator in an extend operation */
	struct allocation_state_extend
	{
	private:
		/* Allocation pointers stored here are tentative, and will be
		 * disclaimed upon restart *unless* (*pmask & mask) != 0, which
		 * indicates that the map ownership mask at pmask acknowledges
		 * ownership of the pointers.
		 * An emplace operation allocate up to two values: key string
		 * and value string.
		 *
		 * The intent behind atomic is that stores are seen in the order
		 * in which they where coded.
		 */

		/* expect at most one allocation */
		bool _armed;
		std::atomic<void *const *> _ptr;
		segment_count *_psegment_count;
		segment_layout::six_t _segment_count_updated_value;
	public:
		allocation_state_extend();
		allocation_state_extend(allocation_state_extend &&) noexcept;

		bool is_armed() const { return _armed; }
		bool is_in_use(const void *ptr);

		/* ERROR: There are now two mechanisms which invalidate the allocation_state data:
		 * clear (here) and the allocation_state_combined enum. This is probably one too
		 * many. Can we use the enum alone, and move the remainder of the clear to an
		 * "initialize" function?
		 */
		template <typename Persister>
			void clear(Persister p_)
			{
				/* important that allocation pointer is zeroed first */
				_ptr = nullptr;
				p_.persist(&_ptr, sizeof _ptr);
				_psegment_count = nullptr;
				_segment_count_updated_value = 0;
				p_.persist(this, sizeof *this);
			}

		template <typename Persister>
			void record_allocation(void *const *ptr_, Persister p_)
			{
				_ptr = ptr_;
				p_.persist(&_ptr, sizeof _ptr);
			}

		void reset()
		{
			_armed = false;
			_ptr = nullptr;
			_psegment_count = nullptr;
			_segment_count_updated_value = 0;
		}

		/* The extend allocation is known when it appears when the segment count
		 * increases to include it, probably.
		 */
		template <typename Persister>
			void record_segment_count_addr_and_target_value(
				segment_count *psegment_count_
				, segment_layout::six_t segment_count_updated_value_
				, Persister p_
		)
		{
			_psegment_count = psegment_count_;
			_segment_count_updated_value = segment_count_updated_value_;
			p_.persist(this, sizeof *this);
		}

		/* ERROR: should be combined with same function in allocation_sstate_emplace */
		template <typename Persister>
			void arm(Persister p_)
			{
				reset();
				_armed = true;
				p_.persist(this, sizeof *this);
			}

		template <typename Persister>
			void disarm(Persister p_)
			{
				_armed = false;
				reset();
				p_.persist(this, sizeof *this);
			}
	};
}

#endif
