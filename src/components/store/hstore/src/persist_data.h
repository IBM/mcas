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


#ifndef MCAS_HSTORE_PERSIST_DATA_H
#define MCAS_HSTORE_PERSIST_DATA_H

#include "as_emplace.h"
#include "as_pin.h"
#include "as_extend.h"
#include "persist_atomic.h"
#include "persist_map.h"

/* Persistent data for hstore.
 *  - persist_map: anchors for the unordered map
 *  - persist_atomic: currently in-progress atomic operation, if any
 */

namespace impl
{
	template <typename AllocatorSegment, typename TypeAtomic>
		struct persist_data
		{
		private:
			/* Three types of allocation states at the moment. At most one at a time is "active" */
			allocation_state_emplace _ase;
			allocation_state_pin _aspd;
			allocation_state_pin _aspk;
			allocation_state_extend _asx;
		public:
			persist_map<AllocatorSegment> _persist_map;
			persist_atomic<TypeAtomic> _persist_atomic;
		public:
			using allocator_type = AllocatorSegment;
			using pm_type = persist_map<AllocatorSegment>;
			using pa_type = persist_atomic<TypeAtomic>;
			persist_data(
				AK_ACTUAL
				std::size_t n
				, const AllocatorSegment &av
			)
				: _ase{}
				, _aspd{}
				, _aspk{}
				, _asx{}
				, _persist_map(AK_REF n, av, &_ase, &_aspd, &_aspk, &_asx)
				, _persist_atomic(&_ase)
			{
			}
            persist_data( persist_data && ) noexcept(!perishable_testing) = default;

			allocation_state_emplace &ase() { return _ase; }
			allocation_state_pin &aspd() { return _aspd; }
			allocation_state_pin &aspk() { return _aspk; }
			allocation_state_extend &asx() { return _asx; }
		};
}

#endif
