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


#ifndef _MCAS_HSTORE_PERSIST_ATOMIC_H
#define _MCAS_HSTORE_PERSIST_ATOMIC_H

#include "as_emplace.h"
#include "mod_control.h"
#include "persist_fixed_string.h"
#include "persistent.h"

#include <array>
#include <cstddef> /* size_t */

/* Persistent data for hstore.
 */

namespace impl
{
	/* Until we get friendship sorted out.
	 * The atomic_controller needs a struct specialized by allocator only
	 * to be friends with persist_atomic
	 */
	template <typename Table>
		struct atomic_controller;

	template <typename Value>
		struct persist_atomic
		{
		private:
			using allocator_type = typename Value::first_type::allocator_type;
			using allocator_traits_type = std::allocator_traits<allocator_type>;
			using mod_ctl_allocator_type = typename allocator_traits_type::template rebind_alloc<mod_control>;
			using mod_ctl_ptr_t = typename std::allocator_traits<mod_ctl_allocator_type>::pointer;

			/* "owner" of the mod_key and mod_mapped. 1 when in use, 0 when not in use */
			persistent_atomic_t<std::uint64_t> mod_owner;
			/* key to destination of modification data */
			using mod_key_t = typename Value::first_type::template rebind<char>;
			mod_key_t mod_key;
			/* source of modification data */
			using mod_mapped_t = typename std::tuple_element<0, typename Value::second_type>::type::template rebind<char>;
			mod_mapped_t mod_mapped;
			/* control of modification data */
			persistent_t<mod_ctl_ptr_t> mod_ctl;
			/* size of control located by mod_ctl (0 if no outstanding modification, negative if the modfication is a replace by erase/emplace */
			persistent_atomic_t<std::ptrdiff_t> mod_size;
			/* One type of allocation state at the moment. */
			allocation_state_emplace *_ase;
			/* persist data for "swap keys" function */
			struct swap
			{
				using mapped_type = typename Value::second_type;
				std::array<char, sizeof(mapped_type)> temp;
				mapped_type *pd0;
				mapped_type *pd1;
				swap()
					: temp{}
					, pd0()
					, pd1()
				{}
				swap(const swap &) = delete;
				swap(swap &&) noexcept = default;
				swap& operator=(const swap &) = delete;
				swap& operator=(swap &&) noexcept = default;
			} _swap;

		public:
			persist_atomic(allocation_state_emplace *ase_)
				: mod_owner()
				, mod_key()
				, mod_mapped()
				, mod_ctl()
				, mod_size(0U)
				, _ase(ase_)
				, _swap{}
			{
			}
			persist_atomic(const persist_atomic &) = delete;
			persist_atomic(persist_atomic &&) noexcept(!perishable_testing) = default;
			persist_atomic& operator=(const persist_atomic &) = delete;
			allocation_state_emplace &ase() { return *_ase; }
			template <typename Table>
				friend struct impl::atomic_controller;
		};
}

#endif
