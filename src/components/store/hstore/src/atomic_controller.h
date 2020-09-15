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


#ifndef MCAS_HSTORE_ATOMIC_CTL_H_
#define MCAS_HSTORE_ATOMIC_CTL_H_

#include "construction_mode.h"
#include "mod_control.h"
#if 0
#include "persist_fixed_string.h"
#endif
#include <api/kvstore_itf.h> /* component */

#include <tuple> /* tuple_element */
#include <type_traits> /* is_base_of */
#include <vector>

namespace impl
{
	template <typename Value>
		struct persist_atomic;

	template <typename Table>
		struct atomic_controller
			: private std::allocator_traits<typename Table::allocator_type>::template rebind_alloc<mod_control>
		{
		private:
			using table_t = Table;
			using allocator_type =
				typename std::allocator_traits<typename table_t::allocator_type>::template rebind_alloc<mod_control>;

			using persist_t = persist_atomic<typename table_t::value_type>;
			persist_t *_persist; /* persist_atomic is a bad name. Should be a noun. */
			table_t *_map;
#if 0
			bool _tick_expired;
#endif
			struct update_finisher
			{
			private:
				impl::atomic_controller<table_t> &_ctlr;
			public:
				update_finisher(impl::atomic_controller<table_t> &ctlr_);
				~update_finisher() noexcept(! TEST_HSTORE_PERISHABLE);
			};
			void redo_update();
			void update_finish();
			void redo_replace();
			void redo_swap();
			void redo_finish();
#if 0
			/* Helpers for the perishable test, to avoid an exception in the finish_update destructor */
			void tick_expired() { _tick_expired = true; }
			bool is_tick_expired() { auto r = _tick_expired; _tick_expired = false; return r; }
#endif
			void persist_range(const void *first_, const void *last_, const char *what_);

			void emm_record_owner_addr_and_bitmask(
				persistent_atomic_t<std::uint64_t> *pmask_
				, std::uint64_t mask_
			)
			{
				auto pe = static_cast<allocator_type *>(this);
				_persist->ase()
					.em_record_owner_addr_and_bitmask(
						pmask_
						, mask_
						, *pe
					);
			}
		public:
			atomic_controller(
				persist_atomic<typename table_t::value_type> &persist_
				, table_t &map_
				, construction_mode mode_
			);
			atomic_controller(const atomic_controller &) = delete;
			atomic_controller& operator=(const atomic_controller &) = delete;

			void redo();

			void enter_update(
				AK_FORMAL
				typename table_t::allocator_type al_
				, const std::string &key
				, std::vector<component::IKVStore::Operation *>::const_iterator first
				, std::vector<component::IKVStore::Operation *>::const_iterator last
			);
			void enter_replace(
				AK_FORMAL
				typename table_t::allocator_type al
				, const std::string &key
				, const char *data
				, std::size_t data_len
				, std::size_t zeros_extend
				, std::size_t alignment
			);
			using mt = typename table_t::mapped_type;
			void enter_swap(
				mt &d0
				, mt &d1
			);
			friend struct atomic_controller<table_t>::update_finisher;
	};
}

#include "atomic_controller.tcc"

#endif
