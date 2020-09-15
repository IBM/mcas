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

#include "as_emplace.h"

/*
 * ===== emplace_allocation_state =====
 */

impl::allocation_state_emplace::allocation_state_emplace()
	: _em()
	, _er()
{
}

/* ERROR: this is a constructor for use in restart. Would be better not to run a constructor
 */
impl::allocation_state_emplace::allocation_state_emplace(allocation_state_emplace &&) noexcept(!perishable_testing) = default;

/*
 * Instant ownership: client recognizes that it owns a value as soon as the allocator writes
 * the allocated value to the client-supplied pointer location.
 *
 * Delayed ownership: client recognizes that it owns a value only after it atomically sets some
 * condition.
 */

/*
 * (If the allocation combined mode is emplace, okay to call this.)
 * (Sub-steps labeled a and b may occur in any order.)
 * (The pronoun "someone" means that a step is not under the direct control of the  indicates
 * Emplace protocol:
 *
 * 1. set _em_pmask to null
 *
 * 2a. set the allocation status enum to "armed", to indicate that
 *    status of an allocation in doubt should be resolved by a call to
 *    allocation_state_emplace::is_in_use.
 * 2b. set _em_ptr0 and _em_ptr1 to null.
 *
 * 3a1. set _em_ptr0 to point to <ptr>, which is initialized to null.
 * 3a2. someone (e.g. persist_fixed_string constructor) calls allocate(<ptr>), changing *_em_ptr from null to a valid pointer.
 * 3b. possible repeat the steps in 3a for a second allocation using _em_ptr1
 * 3c. someone (e.g. hop_hash emplace) sets _em_mask to a bitmask (1 bit on), then _em_pmask to a valid bitmask address where ( _em_mask & *_em_pmask ) == 0
 *
 * 4. someone (e.g. hop_hash emplace) atomically sets the bit mask _em_mask at location at _em_pmask to 1.
 *
 * 5. set the "combined allocation" enum to "idle" to indicate that
 *    the status of an allocation in doubt is "allocated."
 *
 * emplace_arm does 1 and 2.
 * emplace_disarm does 5.
 */

/*
 * (If the allocation combined mode is erase, okay to call this.)
 * (Sub-steps labeled a and b may occur in any order.)
 * Erase protocol:
 * 1. set _er_pmask to null
 *
 * 2a. set the "emplace allocation" enum to "armed", to indicate that
 *    the status of a dellocation in doubt should be resolved by a call to
 *    allocation_state_emplace::is_in_disuse.
 * 2b. set _er_ptr0 and _er_ptr1 to null.
 *
 * 3a. set _er_ptr0 to point to <ptr0>, which is initialized to the adderss of a pointer to client memmory
 * 3b. possible repeat the steps in 3a for a second deallocation using _er_ptr1
 * 3c. someone sets _er_mask to a bitmask (1 bit on), then _er_pmask to a valid bitmask address where ( _er_mask & *_er_pmask ) != 0
 *
 * 4. someone atomically clears a bit at *_er_pmask to 0.
 *
 * 5a. someone calls deallocate(<ptr0>), changing *_er_ptr to null.
 * 5b. possibly repeat the step5a for a second deallocation using _er_ptr1.
 *
 * 6. set the "combined allocation" enum to "idle" to indicate that
 *    no deallocations are in doubt.
 *
 * After step 4 the pointers at *_er_ptr0 and *_er_ptr1 need to be deallocated, as
 * the client side no longer holds them. The recovery code looks like
 *   if ( _er_pmask && ( *_er_pmask & _er_mask ) -- 0 )
 *   (
 *     if ( _er_ptr0 && *_er_ptr0 ) { deallocate(*_er_ptr0); }
 *     if ( _er_ptr1 && *_er_ptr1 ) { deallocate(*_er_ptr1); }
 *   }
 */
