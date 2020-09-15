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

#ifndef _MCAS_HSTORE_ALLOCATION_STATE_EMPLACE_H
#define _MCAS_HSTORE_ALLOCATION_STATE_EMPLACE_H

#include "hstore_config.h"
#include "hop_hash_log.h"
#include "owner.h"
#include "logging.h"
#include <algorithm>
#include <array>
#include <atomic>

namespace impl
{
	/* Client-side persistent allocation state when using the crash-consistent
	 * allocator in an emplace operation */
	struct allocation_state_emplace
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

		/* expect at most key, value allocations */
		static constexpr unsigned max_expected_allocations = 2;
		struct controls
		{
		private:
			persistent_t<bool> _armed;
			persistent_t<unsigned> _index;
			std::array<persistent_t<void *const *>, 4> _ptr;
			persistent_atomic_t<owner::value_type> * _pmask;
			persistent_t<owner::value_type> _mask;
		public:
			controls()
				: _armed(false)
				, _index(0)
				, _ptr()
				, _pmask(nullptr)
				, _mask()
			{}

			controls(controls &&m_) = default;

			void arm()
			{
				/* One of
				 *  (1) creation,
				 *  (2) recovery, or
				 *  (3) disarm (end of the previous operation)
				 * should have left the controls in reset state, but a crash during
				 * recovery might have left _pmask or _index not cleared.
				 */
				assert( ! _armed );
				if ( _pmask )
				{
					hop_hash_log<true>::write(LOG_LOCATION, "non-null initial _pmask ", static_cast<const void *>(_pmask));
				}
				if ( _index )
				{
					hop_hash_log<true>::write(LOG_LOCATION, "non-zero initial _index ", _index);
				}
				_index = 0;
				_pmask = nullptr;
				/* 2. state set to "active" (caller must persist) */
				_armed = true;
			}

			void disarm()
			{
				reset();
			}

			bool is_armed() const { return _armed; }

			void reset()
			{
				_armed = false;
				_index = 0;
				_pmask = nullptr;
			}

			template <typename Persister>
				void record(
					void *const *ptr_
					, Persister p_
				)
				{
					hop_hash_log<false>::write(LOG_LOCATION, _armed ? "armed" : "not armed", static_cast<const void *>(ptr_));
					if ( _armed )
					{
						/* Possible that a deallocate is being redone.
						 * If this is the case, do nothing
						 * Else add the ptr
						 */
						if ( std::find( &_ptr[0], &_ptr[_index], ptr_) == &_ptr[_index] )
						{
							assert( _index != 4 );
							_ptr[_index] = ptr_;
							++_index;
							p_.persist(this, sizeof *this);
						}
					}
					else
					{
						hop_hash_log<false>::write(LOG_LOCATION, "cannot record (armed is ", "false");
					}
				}

			template <typename Persister>
				void record_owner_addr_and_bitmask(
					persistent_atomic_t<owner::value_type> *pmask_
					, owner::value_type mask_
					, Persister p_
			)
			{
				hop_hash_log<false>::write(LOG_LOCATION, _armed ? "armed" : "not armed", static_cast<void *>(pmask_));
				_pmask = pmask_;
				_mask = mask_;
				p_.persist(this, sizeof *this);
			}

			bool has_disused_ptrs()
			{
				/* might have disused ptrs if
				 * (1) the allocation state was armed
				 * (2) the owner mask pointer was valid
				 * (3) the item(s) controlled by *_pmask were atomically disclaimed by the client
				 */
				return
					_armed
					&& _pmask
					&& (*_pmask & _mask) == 0
					;
			}

			bool has_used_ptrs() const
			{
				return
					_armed
					&& _pmask != nullptr
					&& (*_pmask & _mask) != 0
				;
			}

			bool is_in_use(void *ptr_) const
			{
				auto in_use =
					has_used_ptrs()
					&&
					std::find(&_ptr[0], &_ptr[_index], ptr_) != &_ptr[_index]
				;
				hop_hash_log<false>::write(LOG_LOCATION, "ptr ", ptr_, " in use ", in_use ? "true" : "false");
				return in_use;
			}

			void *disused_ptr(unsigned i)
			{
				return
					has_disused_ptrs() && _index > i
					? *_ptr[i]
					: nullptr
					;
			}
		};
		controls _em; // control for allocation
		controls _er; // control for deallocation

/* unused */
#if 0
		template <typename Persister>
			void clear(Persister p_)
			{
				_em.clear(p_);
				_er.clear(p_);
			}
#endif
		bool has_disused_ptrs()
		{
			/* might have disused ptrs if
			 * (1) the allocation state was armed
			 * (2) the owner mask pointer was valid
			 * (3) the item(s) controlled by *_er._pmask were indeed atomically disclaimed by the client
			 */
			return _er.has_disused_ptrs();
		}

		void *er_disused_ptr(unsigned i) { return _er.disused_ptr(i); }

	public:
		allocation_state_emplace();
		allocation_state_emplace(allocation_state_emplace &&) noexcept(!perishable_testing);

		void reset()
		{
			_er.disarm();
			_em.disarm();
		}

		bool is_armed() const { return _em.is_armed() || _er.is_armed(); }

		bool is_in_use(void *ptr_) const { return _em.is_in_use(ptr_); }

		template <typename Persister>
			void arm(Persister p_)
			{
				_em.arm();
				_er.arm();
				p_.persist(this, sizeof *this);
				// static_assert(sizeof *this <= 64, "allocation_state_emplace exceeds one cache line");
			}

		/* persister parameter is not used at the moment, but is named so as to match the "arm" call. */
		template <typename Persister>
			void disarm(
				Persister // p_
			)
			{
				_er.disarm();
				_em.disarm();
			}

		template <typename Persister>
			void record_allocation(void *const *ptr_, Persister p_)
			{
				_em.record(ptr_, p_);
			}

		template <typename Persister>
			void record_deallocation(void *const *ptr_, Persister p_)
			{
				_er.record(ptr_, p_);
			}

		template <typename Persister>
			void em_record_owner_addr_and_bitmask(
				persistent_atomic_t<owner::value_type> *pmask_
				, owner::value_type mask_
				, Persister p_
		)
		{
			_em.record_owner_addr_and_bitmask(pmask_, mask_, p_);
		}

		template <typename Persister>
			void er_record_owner_addr_and_bitmask(
				persistent_atomic_t<owner::value_type> *pmask_
				, owner::value_type mask_
				, Persister p_
		)
		{
			_er.record_owner_addr_and_bitmask(pmask_, mask_, p_);
		}
	};
}

#endif
