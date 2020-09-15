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


#ifndef MCAS_HSTORE_PERSIST_MAP_H
#define MCAS_HSTORE_PERSIST_MAP_H

#include "bucket_aligned.h"
#include "hash_bucket.h"
#include "persist_fixed_string.h"
#include "persistent.h"
#include "persist_atomic.h"
#include "segment_count.h"
#include "segment_layout.h"
#include "size_control.h"

#include <cstddef> /* size_t */

/* Persistent data for hstore.
 */

namespace impl
{
	struct allocation_state_emplace;
	struct allocation_state_pin;
	struct allocation_state_extend;
	using segment_count_actual_t = value_unstable<segment_layout::six_t, 1>;

	template <typename Allocator>
		struct persist_controller;

	template <typename Allocator>
		struct persist_map
		{
		private:
			using value_type = typename Allocator::value_type;
			static constexpr std::size_t segment_align = 64U;
		public:
			using bucket_aligned_t = bucket_aligned<hash_bucket<value_type>>;
		private:
			using allocator_traits_type = std::allocator_traits<Allocator>;
			using bucket_allocator_t =
				typename allocator_traits_type::template rebind_alloc<bucket_aligned_t>;
			using bucket_ptr = typename std::allocator_traits<bucket_allocator_t>::pointer;

			/* bucket indexes */
			using bix_t = segment_layout::bix_t;
			/* segment indexes */
			using six_t = segment_layout::six_t;

			struct segment_control
			{
				persistent_t<bucket_ptr> bp;
				segment_control()
					: bp()
				{
				}
			};

			static constexpr six_t _segment_capacity = 32U;
			static constexpr unsigned log2_base_segment_size =
				segment_layout::log2_base_segment_size;
			static constexpr bix_t base_segment_size =
				segment_layout::base_segment_size;

			size_control _size_control;

			segment_count _segment_count;

			segment_control _sc[_segment_capacity];

			/* Three types of allocation states at the moment. At most one at a time is "active" */
			allocation_state_emplace *_ase;
			allocation_state_pin *_aspd;
			allocation_state_pin *_aspk;
			allocation_state_extend *_asx;
		public:
			persist_map(
				AK_ACTUAL
				std::size_t n
				, Allocator av
				, allocation_state_emplace *ase_
				, allocation_state_pin *aspd_
				, allocation_state_pin *aspk_
				, allocation_state_extend *asx_
			);
			persist_map(persist_map &&) noexcept(!perishable_testing) = default;
			void do_initial_allocation(
				AK_ACTUAL
				persist_controller<Allocator> *pc
			);
			void reconstitute(Allocator av);
			allocation_state_emplace &ase() { return *_ase; }
			allocation_state_extend &asx() { return *_asx; }
			friend struct persist_controller<Allocator>;
		};
}

#include "persist_map.tcc"

#endif
