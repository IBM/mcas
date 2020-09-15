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


#ifndef _MCAS_HSTORE_HOP_HASH_H
#define _MCAS_HSTORE_HOP_HASH_H

#include "segment_layout.h"

#include "bucket_control_unlocked.h"
#include "construction_mode.h"
#include "hash_bucket.h"
#include "hop_hash_log.h"
#include "trace_flags.h"
#include "persist_controller.h"
#include "segment_and_bucket.h"

#include <boost/iterator/transform_iterator.hpp>

#include <atomic>
#include <cstddef> /* size_t */
#include <cstdint> /* uint32_t */
#include <functional> /* equal_to */
#include <limits>
#include <new> /* allocator */
#include <shared_mutex> /* shared_timed_mutex */
#include <stdexcept>
#include <string>
#include <utility> /* hash, pair */

/* Inteded to implement Hopscotch hashing
 * http://mcg.cs.tau.ac.il/papers/disc2008-hopscotch.pdf
 */

#include "hop_hash_debug.h"

#if TRACED_TABLE
template <
	typename Key
	, typename T
	, typename Hash
	, typename Pred
	, typename Allocator
	, typename SharedMutex
>
	struct hop_hash;
#endif

namespace impl
{
	struct no_near_empty_bucket
		: public std::range_error
	{
		using bix_t = std::size_t;
	private:
		bix_t _bi;
	public:
		no_near_empty_bucket(bix_t bi, std::size_t size, const std::string &why);
		bix_t bi() const { return _bi; }
	};

	struct move_stuck
		: public no_near_empty_bucket
	{
		move_stuck(bix_t bi, std::size_t size);
	};

	struct hop_hash_full
		: public no_near_empty_bucket
	{
		hop_hash_full(bix_t bi, std::size_t size);
	};

	template <typename HopHash>
		struct hop_hash_iterator;

	template <typename HopHash>
		struct hop_hash_const_iterator;

	template <typename HopHash>
		struct hop_hash_local_iterator;

	template <typename HopHash>
		struct hop_hash_const_local_iterator;

	template <typename Bucket, typename Referent>
		struct bucket_ref
		{
			using segment_and_bucket_t = segment_and_bucket<Bucket>;
		private:
			Referent *_ref;
			segment_and_bucket_t _sb;
		public:
			bucket_ref(Referent *ref_, const segment_and_bucket_t &sb_)
				: _ref(ref_)
				, _sb(sb_)
			{
			}
			std::size_t index() const { return sb().index(); }
			const segment_and_bucket_t &sb() const { return _sb; }
			Referent &ref() const { return *_ref; }
			/* A "content" bucket_ref also owns the "in-use bit, which, for space reasons,
			 * is currently stored in the "owner" object.
			 * The holder of a content lock can get the owner ref for the sole purpose of
			 * accessing the in-use bit.
			 */
			owner &owner_ref() const { return *static_cast<hash_bucket<typename Referent::value_type> *>(_ref); }
		};

	template <typename Bucket, typename Referent, typename SharedMutex>
		struct bucket_unique_lock
			: public bucket_ref<Bucket, Referent>
			, public std::unique_lock<SharedMutex>
		{
			using base_ref = bucket_ref<Bucket, Referent>;
			using typename base_ref::segment_and_bucket_t;
			bucket_unique_lock(
				Referent &b_
				, const segment_and_bucket_t &i_
				, SharedMutex &m_
			)
				: bucket_ref<Bucket, Referent>(&b_, i_)
				, std::unique_lock<SharedMutex>((hop_hash_log<HSTORE_TRACE_LOCK>::write(LOG_LOCATION, this->index(), Referent::lock_id), m_))
			{
			}
			bucket_unique_lock(bucket_unique_lock &&) noexcept = default;
			bucket_unique_lock &operator=(bucket_unique_lock &&other_) noexcept
			{
				hop_hash_log<HSTORE_TRACE_LOCK>::write(LOG_LOCATION, this->index(), Referent::lock_id
					, "->", other_.index(), Referent::lock_id);
				std::unique_lock<SharedMutex>::operator=(std::move(other_));
				base_ref::operator=(std::move(other_));
				return *this;
			}
			~bucket_unique_lock()
			{
				if ( this->owns_lock() )
				{
					hop_hash_log<HSTORE_TRACE_LOCK>::write(LOG_LOCATION, this->index(), Referent::lock_id);
				}
			}
			template <typename HopHash>
				void assert_clear(bool b, HopHash &t)
				{
					this->ref().assert_clear(b, *this, t);
				}
		};

	template <typename Bucket, typename Referent, typename SharedMutex>
		struct bucket_shared_lock
			: public bucket_ref<Bucket, Referent>
			, public std::shared_lock<SharedMutex>
		{
			using base_ref = bucket_ref<Bucket, Referent>;
			using base_lock = std::shared_lock<SharedMutex>;
			using typename base_ref::segment_and_bucket_t;
			bucket_shared_lock(
				Bucket &b_
				, const segment_and_bucket_t &i_
				, SharedMutex &m_
			)
				: base_ref(&b_, i_)
				, base_lock(m_)
			{
				hop_hash_log<HSTORE_TRACE_LOCK>::write(LOG_LOCATION, this->index());
			}
			bucket_shared_lock(bucket_shared_lock &&) noexcept = default;
			auto operator=(bucket_shared_lock &&other_) noexcept -> bucket_shared_lock &
			{
				hop_hash_log<HSTORE_TRACE_LOCK>::write(LOG_LOCATION, this->index(), "->", other_.index());
				base_lock::operator=(std::move(other_));
				base_ref::operator=(std::move(other_));
				return *this;
			}
			~bucket_shared_lock()
			{
				if ( this->owns_lock() )
				{
					hop_hash_log<HSTORE_TRACE_LOCK>::write(LOG_LOCATION, this->index());
				}
			}
		};

	template <typename Mutex>
		struct bucket_mutexes
		{
			Mutex _m_owner;
			Mutex _m_content;
		public:
			bucket_mutexes()
				: _m_owner{}
				, _m_content{}
			{}
		};

	template <typename Bucket, typename Mutex>
		struct bucket_control
			: public bucket_control_unlocked<Bucket>
		{
			std::unique_ptr<bucket_mutexes<Mutex>[]> _bucket_mutexes;
		public:
			using base = bucket_control_unlocked<Bucket>;
			using typename base::bucket_aligned_t;
			using typename base::six_t;
			explicit bucket_control(
				six_t index_
				, bucket_aligned_t *buckets_
			)
				: bucket_control_unlocked<Bucket>(index_, buckets_)
				, _bucket_mutexes(nullptr)
			{
			}
			explicit bucket_control()
				: bucket_control(0U, nullptr)
			{
			}
			~bucket_control()
			{
			}
		};

	template <typename Allocator>
		struct hop_hash_allocator
			: public Allocator
		{
			explicit hop_hash_allocator(const Allocator &av_)
				: Allocator(av_)
			{}
		};

	template <typename HopHash>
		struct hop_hash_local_iterator_impl;

	template <typename HopHash>
		struct hop_hash_iterator_impl;

	template <
		typename Key
		, typename T
		, typename Hash
		, typename Pred
		, typename Allocator
		, typename SharedMutex
	>
		struct hop_hash_base
			: private hop_hash_allocator<Allocator>
			, private segment_layout
			, private
				persist_controller<
					typename std::allocator_traits<Allocator>::template rebind_alloc<
						std::pair<const Key, T>
					>
				>
		{
		private:
			using allocator_traits_type = std::allocator_traits<Allocator>;
		public:
#if HSTORE_TRACE_RESIZE
			using persist_controller<
				typename std::allocator_traits<Allocator>::template rebind_alloc
				<
					std::pair<const Key, T>
				>
			>::segment_count_actual;
#endif
			using key_type        = Key;
			using mapped_type     = T;
			using value_type      = std::pair<const key_type, mapped_type>;
			using persist_controller_t =
				persist_controller<
					typename allocator_traits_type::template rebind_alloc<value_type>
				>;
			using size_type       = typename persist_controller_t::size_type;
			using hasher          = Hash;
			using key_equal       = Pred;
			using allocator_type  = Allocator;
			using pointer         = typename allocator_traits_type::pointer;
			using const_pointer   = typename allocator_traits_type::const_pointer;
			using reference       = value_type &;
			using const_reference = const value_type &;
			using iterator        = hop_hash_iterator<hop_hash_base>;
			using const_iterator  = hop_hash_const_iterator<hop_hash_base>;
			using local_iterator  = hop_hash_local_iterator<hop_hash_base>;
			using const_local_iterator = hop_hash_const_local_iterator<hop_hash_base>;
			using persist_data_t =
				persist_map<typename allocator_traits_type::template rebind_alloc<value_type>>;
#if ! HSTORE_TRACE_RESIZE
		private:
#endif
			using bix_t = size_type; /* sufficient for all bucket indexes */
		private:
			using hash_result_t = typename hasher::result_type;
#if HSTORE_TRACE_RESIZE
		public:
#endif
			using bucket_t = hash_bucket<value_type>;
			using content_t = content<value_type>;
		private:
			using bucket_mutexes_t = bucket_mutexes<SharedMutex>;
			using bucket_control_t = bucket_control<bucket_t, SharedMutex>;
			using bucket_aligned_t = typename bucket_control_t::bucket_aligned_t;
#if ! TRACED_OWNER && ! TRACED_CONTENT
			static_assert(sizeof(bucket_aligned_t) <= 64, "Bucket size exceeds presumed cache line size (64)");
#endif
			using bucket_allocator_t =
				typename allocator_traits_type::template rebind_alloc<bucket_aligned_t>;
			using owner_unique_lock_t = bucket_unique_lock<bucket_t, owner, SharedMutex>;
			using owner_shared_lock_t = bucket_shared_lock<bucket_t, owner, SharedMutex>;
			using content_unique_lock_t = bucket_unique_lock<bucket_t, content_t, SharedMutex>;
			using content_shared_lock_t = bucket_shared_lock<bucket_t, content_t, SharedMutex>;
#if HSTORE_TRACE_RESIZE
		public:
#endif
			using segment_and_bucket_t = segment_and_bucket<bucket_t>;
			void ownership_check() const;
		public:
			static constexpr auto _segment_capacity =
				persist_controller_t::_segment_capacity;
			/* Need to adjust hash and bucket_ix interpretations in more places
			* before this can becom non-zero
			*/
			static constexpr unsigned log2_base_segment_size =
				persist_controller_t::log2_base_segment_size;
			static constexpr bix_t base_segment_size =
				persist_controller_t::base_segment_size;
			static_assert(
				owner::size < base_segment_size
				, "Base segment size must exceeed owner size"
			);
			using six_t = std::size_t; /* segment indexes (but uint8_t would do) */

			/* The number of non-null elements of _b may exceed _count
			* only if a crash occurred between the assignemnt of a non-null
			* value to _b[_count] and the increment of _count.
			* In that case the non-null value at _count points to lost memory.
			* That memory can be recovered the next time _b is extended.
			*/
			hasher _hasher;

			bool _auto_resize;

			bucket_control_t _bc[_segment_capacity];

			six_t segment_count() const override
			{
				return persist_controller_t::segment_count_actual().value();
			}

			six_t segment_count_not_stable() const
			{
				return persist_controller_t::segment_count_actual().value_not_stable();
			}

			auto bucket_ix(const hash_result_t h) const -> bix_t;
			auto bucket_expanded_ix(const hash_result_t h) const -> bix_t;

			auto nearest_free_bucket(segment_and_bucket_t bi) -> content_unique_lock_t;

			auto make_space_for_insert(
				bix_t bi
				, content_unique_lock_t bf
			) -> content_unique_lock_t;

			template <typename Lock, typename K>
				auto content_index_of_key(
					Lock &bi
					, const K &k
				) const -> owner::index_type;

			template <typename Lock, typename K>
				auto locate_key(
					Lock &bi
					, const K &k
				) const -> std::tuple<bucket_t *, segment_and_bucket_t>;

			void resize(AK_FORMAL0);
			void resize_pass1();
			void resize_pass2();
			bool resize_pass2_adjust_owner(
				bix_t ix_senior
				, bucket_control_t &junior_bucket_control
				, content_unique_lock_t &populated_content_lk
			);
			auto locate_bucket_mutexes(
				const segment_and_bucket_t &
			) const -> bucket_mutexes_t &;

			auto make_owner_unique_lock(
				const segment_and_bucket_t &a
			) const -> owner_unique_lock_t;

			/* lock an owner which precedes content */
			auto make_owner_unique_lock(
				const content_unique_lock_t &
				, unsigned bkwd
			) const -> owner_unique_lock_t;

			template <typename K>
				auto make_owner_shared_lock(const K &k) const -> owner_shared_lock_t;
			auto make_owner_shared_lock(
				const segment_and_bucket_t &
			) const -> owner_shared_lock_t;

			auto make_content_unique_lock(
				const segment_and_bucket_t &
			) const -> content_unique_lock_t;
			/* lock content which follows an owner */
			auto make_content_unique_lock(
				const owner_unique_lock_t &
				, unsigned fwd
			) const -> content_unique_lock_t;

			using persist_controller_t::mask;

			auto owner_value_at(owner_unique_lock_t &bi) const -> owner::value_type;
			auto owner_value_at(owner_shared_lock_t &bi) const -> owner::value_type;

			auto make_segment_and_bucket(bix_t ix) const -> segment_and_bucket_t;
			auto make_segment_and_bucket_unsafe(bix_t ix) const -> segment_and_bucket_t;
			auto make_segment_and_bucket_for_iterator(
				bix_t ix
			) const -> segment_and_bucket_t;
			auto make_segment_and_bucket_at_begin() const -> segment_and_bucket_t;
			auto make_segment_and_bucket_at_end() const -> segment_and_bucket_t;
			auto make_segment_and_bucket_prev(
				segment_and_bucket_t a
				, unsigned bkwd
			) const -> segment_and_bucket_t;

			static auto locate_owner(const segment_and_bucket_t &a) -> const owner &;

			template <typename K>
				auto bucket(const K &) const -> size_type;
			auto bucket_size(const size_type n) const -> size_type;

			auto owned_by_owner_mask(const segment_and_bucket_t &a) const -> owner::value_type;
			bool is_free_by_owner(const segment_and_bucket_t &a) const;
			bool is_Free(const segment_and_bucket_t &a);
			bool is_free(const segment_and_bucket_t &a) const;

			/* computed distance from first to last, accounting for the possibility that
			 * last is smaller than first due to wrapping.
			 */
			auto distance_wrapped(bix_t first, bix_t last) -> unsigned;

			unsigned _locate_key_call;
			unsigned _locate_key_owned;
			unsigned _locate_key_unowned;
			unsigned _locate_key_match;
			unsigned _locate_key_mismatch;

		public:
			explicit hop_hash_base(
				AK_FORMAL
				persist_data_t *pc
				, construction_mode mode
				, const Allocator &av = Allocator()
			);
		protected:
			virtual ~hop_hash_base();
		public:
			hop_hash_base(const hop_hash_base &) = delete;
			hop_hash_base &operator=(const hop_hash_base &) = delete;
			allocator_type get_allocator() const noexcept
			{
				return static_cast<const hop_hash_allocator<Allocator> &>(*this);
			}

			template <typename ... Args>
				auto emplace(
					AK_FORMAL
					Args && ... args
				) -> std::pair<iterator, bool>;
			auto insert(const value_type &value) -> std::pair<iterator, bool>;

			template <typename K>
				auto erase(const K &key) -> size_type;

			auto erase(iterator it) -> iterator;

			template <typename K>
				auto find(const K &key) -> iterator;
			template <typename K>
				auto find(const K &key) const -> const_iterator;

			template <typename K>
				auto at(const K &key) -> mapped_type &;
			template <typename K>
				auto at(const K &key) const -> const mapped_type &;

			template <typename K>
				auto count(const K &k) const -> size_type;
			auto begin() -> iterator
			{
				return iterator(make_segment_and_bucket_at_begin(), 0U);
			}
			auto end() -> iterator
			{
				return iterator(make_segment_and_bucket_at_end(), 0U);
			}
			auto begin() const -> const_iterator
			{
				return cbegin();
			}
			auto end() const -> const_iterator
			{
				return cend();
			}
			auto cbegin() const -> const_iterator
			{
				return const_iterator(make_segment_and_bucket_at_begin(), 0U);
			}
			auto cend() const -> const_iterator
			{
				return const_iterator(make_segment_and_bucket_at_end(), 0U);
			}

			using persist_controller_t::bucket_count;
			using persist_controller_t::max_bucket_count;

			auto begin(size_type n) -> local_iterator
			{
				auto sb = make_segment_and_bucket_for_iterator(n);
				auto owner_lk = make_owner_shared_lock(sb);
				return local_iterator(*this, sb, locate_owner(sb).value(owner_lk));
			}
			auto end(size_type n) -> local_iterator
			{
				auto sb = make_segment_and_bucket_for_iterator(n);
				return local_iterator(*this, sb, owner::value_type(0));
			}
			auto begin(size_type n) const -> const_local_iterator
			{
				return cbegin(n);
			}
			auto end(size_type n) const -> const_local_iterator
			{
				return cend(n);
			}
			auto cbegin(size_type n) const -> const_local_iterator
			{
				auto sb = make_segment_and_bucket_for_iterator(n);
				auto owner_lk = make_owner_shared_lock(sb);
				return const_local_iterator(sb, locate_owner(sb).value(owner_lk));
			}
			auto cend(size_type n) const -> const_local_iterator
			{
				auto sb = make_segment_and_bucket_for_iterator(n);
				return const_local_iterator(sb, owner::value_type(0));
			}
			auto size() const -> size_type;

			bool set_auto_resize(bool v1) { auto v0 = _auto_resize; _auto_resize = v1; return v0; }
			bool get_auto_resize() const { return _auto_resize; }

#if TRACED_TABLE
			friend
				auto operator<< <>(
					std::ostream &
					, const impl::hop_hash_print<
						hop_hash<Key, T, Hash, Pred, Allocator, SharedMutex>
					> &
				) -> std::ostream &;

			friend
				auto operator<< <>(
					std::ostream &
					, const impl::dump<true>::hop_hash_dump<hop_hash_base> &
				) -> std::ostream &;
#endif
#if TRACED_OWNER
			template <typename Lock>
				friend
					auto operator<<(
						std::ostream &o_
						, const impl::owner_print<Lock> &
					) -> std::ostream &;

			template <typename HopHashBase>
				friend
					auto operator<<(
						std::ostream &o_
						, const owner_print<
							bypass_lock<typename HopHashBase::bucket_t, const owner>
						> &
					) -> std::ostream &;
#endif
#if TRACEE_BUCKET
			template<
				typename HopHashBase
				, typename LockOwner
				, typename LockContent
			>
				friend
					auto operator<<(
						std::ostream &o_
						, const impl::bucket_print<
							LockOwner
							, LockContent
						> &
					) -> std::ostream &;

			template <typename HopHash>
				friend
				auto operator<<(
					std::ostream &o_
					, const bucket_print
					<
						bypass_lock<typename HopHash::bucket_t, const owner>
						, bypass_lock<
							typename HopHash::bucket_t
							, const content<typename HopHash::value_type>
						>
					> &
				) -> std::ostream &;
#endif
			template <typename HopHash>
				friend struct impl::hop_hash_local_iterator_impl;
			template <typename HopHash>
				friend struct impl::hop_hash_iterator_impl;
			template <typename HopHash>
				friend struct impl::hop_hash_local_iterator;
			template <typename HopHash>
				friend struct impl::hop_hash_const_local_iterator;
			template <typename HopHash>
				friend struct impl::hop_hash_iterator;
			template <typename HopHash>
				friend struct impl::hop_hash_const_iterator;
		};
}

template <
	typename Key
	, typename T
	, typename Hash = std::hash<Key>
	, typename Pred = std::equal_to<Key>
	, typename Allocator = std::allocator<std::pair<const Key, T>>
	, typename SharedMutex = std::shared_timed_mutex
>
	struct hop_hash
		: private impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>
	{
		using base = impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>;
		using size_type      = std::size_t;
		using key_type       = Key;
		using mapped_type    = T;
		using value_type     = std::pair<const key_type, mapped_type>;
		/* base is private. Can we use it to specify public types? Yes. */
		using iterator       = impl::hop_hash_iterator<base>;
		using const_iterator = impl::hop_hash_const_iterator<base>;
		using persist_data_t = typename base::persist_data_t;
		using allocator_type = typename base::allocator_type;

		/* contruct/destroy/copy */
		explicit hop_hash(
			AK_ACTUAL
			persist_data_t *pc_
			, construction_mode mode_
			, const Allocator &av_ = Allocator()
		)
			: base(AK_REF pc_, mode_, av_)
		{}

		using base::get_allocator;

		/* size and capacity */
		auto empty() const noexcept -> bool
		{
			return size() == 0;
		}
		using base::size;
		using base::get_auto_resize;
		using base::set_auto_resize;
		using base::bucket_count;
		auto max_size() const noexcept -> size_type
		{
			return (size_type(1U) << (base::_segment_capacity-1U));
		}
		/* iterators */

		using base::begin;
		using base::end;
		using base::cbegin;
		using base::cend;

		/* modifiers */

		using base::emplace;
		using base::insert;
		using base::erase;

		/* counting */

		using base::count;

		/* lookup */
		using base::find;
		using base::at;

		template <typename HopHash>
			friend struct impl::hop_hash_local_iterator_impl;

		template <typename HopHash>
			friend struct impl::hop_hash_iterator_impl;
	};

namespace impl
{
	template <typename HopHash>
		bool operator!=(
			const hop_hash_local_iterator_impl<HopHash> a
			, const hop_hash_local_iterator_impl<HopHash> b
		);

	template <typename HopHash>
		bool operator==(
			const hop_hash_local_iterator_impl<HopHash> a
			, const hop_hash_local_iterator_impl<HopHash> b
		);

	template <typename HopHash>
		bool operator!=(
			const hop_hash_iterator_impl<HopHash> a
			, const hop_hash_iterator_impl<HopHash> b
		);

	template <typename HopHash>
		bool operator==(
			const hop_hash_iterator_impl<HopHash> a
			, const hop_hash_iterator_impl<HopHash> b
		);

	template <typename HopHash>
		struct hop_hash_local_iterator_impl
			: public std::iterator
				<
					std::forward_iterator_tag
					, typename HopHash::value_type
				>
		{
		private:
			using segment_and_bucket_t = typename HopHash::segment_and_bucket_t;
			segment_and_bucket_t _sb_owner;
			owner::index_type _content_index;
			void advance_to_in_use()
			{
				while ( _content_index != owner::size && ! _sb_owner.deref().is_in_use(_content_index) )
				{
					++_content_index;
				}
				if ( _content_index == owner::size )
				{
					_content_index = 0;
					_sb_owner.incr_with_wrap();
				}
			}
			segment_and_bucket_t sb_content() const
			{
				segment_and_bucket_t sbc(_sb_owner);
				sbc.add_small(_content_index);
				return sbc;
			}

		protected:
			using base =
				std::iterator<std::forward_iterator_tag, typename HopHash::value_type>;

			/* HopHash 106 (Iterator) */
			auto deref() const -> typename base::reference
			{
				return sb_content().deref().content<typename HopHash::value_type>::value();
			}
			void incr()
			{
				++_content_index;
				advance_to_in_use();
			}
		public:
			/* HopHash 17 (EqualityComparable) */
			friend
				bool operator== <>(
					hop_hash_local_iterator_impl<HopHash> a
					, hop_hash_local_iterator_impl<HopHash> b
				);
			/* HopHash 107 (InputIterator) */
			friend
				bool operator!= <>(
					hop_hash_local_iterator_impl<HopHash> a
					, hop_hash_local_iterator_impl<HopHash> b
				);
			/* HopHash 109 (ForwardIterator) - handled by 107 */
		public:
			hop_hash_local_iterator_impl(const segment_and_bucket_t &sb_owner_)
				: _sb_owner(sb_owner_)
				, _content_index(0)
			{
				advance_to_in_use();
			}
		};

	template <typename HopHash>
		struct hop_hash_iterator_impl
			: public std::iterator
				<
					std::forward_iterator_tag
					, typename HopHash::value_type
				>
		{
		private:
			using segment_and_bucket_t = typename HopHash::segment_and_bucket_t;
			/* ERROR: iterator runs through non-empty content.
			 * It should run through owners, indexing the content within each owner
			 * that is necessary to allow erase(), which needs both to locate both owner bucket
			 * and content bucket, to take an iterator.
			 */
			segment_and_bucket_t _sb_owner;
			unsigned _content_index;
			/* The "end" value is the past the last _sb_owner, that is (last segment, last bucket+1) */
			void advance_to_in_use()
			{
				while ( ! _sb_owner.at_end() && ! _sb_owner.deref().is_in_use(_content_index) )
				{
					++_content_index;
					if ( _content_index == owner::size )
					{
						_content_index = 0;
						_sb_owner.incr_without_wrap();
					}
				}
			}

		protected:
			using base =
				std::iterator<std::forward_iterator_tag, typename HopHash::value_type>;

			segment_and_bucket_t sb_content() const
			{
				segment_and_bucket_t sbc(_sb_owner);
				sbc.add_small(_content_index);
				return sbc;
			}

			/* HopHash 106 (Iterator) */
			auto deref() const -> typename base::reference
			{
				return sb_content().deref().content<typename HopHash::value_type>::value();
			}
			void incr()
			{
				++_content_index;
				if ( _content_index == owner::size )
				{
					_content_index = 0;
					_sb_owner.incr_without_wrap();
				}
				advance_to_in_use();
			}
		public:
			/* HopHash 17 (EqualityComparable) */
			friend
				bool operator== <>(
					hop_hash_iterator_impl<HopHash> a
					, hop_hash_iterator_impl<HopHash> b
				);
			/* HopHash 107 (InputIterator) */
			friend
				bool operator!= <>(
					hop_hash_iterator_impl<HopHash> a
					, hop_hash_iterator_impl<HopHash> b
				);
			/* HopHash 109 (ForwardIterator) - handled by 107 */
		public:
			hop_hash_iterator_impl(const segment_and_bucket_t &sb_owner_, const owner::index_type ix_)
				: _sb_owner(sb_owner_)
				, _content_index(ix_)
			{
				advance_to_in_use();
			}
			friend struct impl::hop_hash_iterator<HopHash>;
		};

	template <typename HopHash>
		struct hop_hash_local_iterator
			: public hop_hash_local_iterator_impl<HopHash>
		{
		private:
			using segment_and_bucket_t = typename HopHash::segment_and_bucket_t;
			using typename hop_hash_local_iterator_impl<HopHash>::base;
		public:
			hop_hash_local_iterator(
				const HopHash & // t_
				, const segment_and_bucket_t & sb_
				, owner::value_type mask_
			)
				: hop_hash_local_iterator_impl<HopHash>(sb_, mask_)
			{}
			/* HopHash 106 (Iterator) */
			auto operator*() const -> typename base::reference
			{
				return this->deref();
			}
			auto operator++() -> hop_hash_local_iterator &
			{
				this->incr();
				return *this;
			}
			/* HopHash 17 (EqualityComparable) */
			/* HopHash 107 (InputIterator) */
			auto operator->() const -> typename base::pointer
			{
				return &this->deref();
			}
			auto operator++(int) -> hop_hash_local_iterator
			{
				auto ti = *this;
				this->incr();
				return ti;
			}
			/* HopHash 109 (ForwardIterator) - handled by 107 */
		};

	template <typename HopHash>
		struct hop_hash_const_local_iterator
			: public hop_hash_local_iterator_impl<HopHash>
		{
		private:
			using segment_and_bucket_t = typename HopHash::segment_and_bucket_t;
			using typename hop_hash_local_iterator_impl<HopHash>::base;
		public:
			hop_hash_const_local_iterator(const segment_and_bucket_t & sb_, owner::value_type mask_)
				: hop_hash_local_iterator_impl<HopHash>(sb_, mask_)
			{}
			/* HopHash 106 (Iterator) */
			auto operator*() const -> const typename base::reference
			{
				return this->deref();
			}
			auto operator++() -> hop_hash_const_local_iterator &
			{
				this->incr();
				return *this;
			}
			/* HopHash 17 (EqualityComparable) */
			/* HopHash 107 (InputIterator) */
			auto operator->() const -> const typename base::pointer
			{
				return &this->deref();
			}
			auto operator++(int) -> hop_hash_const_local_iterator
			{
				auto ti = *this;
				this->incr();
				return ti;
			}
			/* HopHash 109 (ForwardIterator) - handled by 107 */
		};

	template <typename HopHash>
		struct hop_hash_iterator
			: public hop_hash_iterator_impl<HopHash>
		{
		private:
			using segment_and_bucket_t = typename HopHash::segment_and_bucket_t;
			using typename hop_hash_iterator_impl<HopHash>::base;
			const auto &sb_owner() const { return this->_sb_owner; }
			using hop_hash_iterator_impl<HopHash>::sb_content;
			auto content_index() const { return this->_content_index; }
		public:
			hop_hash_iterator(const segment_and_bucket_t & sb_, owner::index_type ix_)
				: hop_hash_iterator_impl<HopHash>(sb_, ix_)
			{}
			/* HopHash 106 (Iterator) */
			auto operator*() const -> typename base::reference
			{
				return this->deref();
			}
			auto operator++() -> hop_hash_iterator &
			{
				this->incr();
				return *this;
			}
			/* HopHash 17 (EqualityComparable) */
			/* HopHash 107 (InputIterator) */
			auto operator->() const -> typename base::pointer
			{
				return &this->deref();
			}
			hop_hash_iterator operator++(int)
			{
				auto ti = *this;
				this->incr();
				return ti;
			}
			/* HopHash 109 (ForwardIterator) - handled by 107 */
			template <
				typename Key
				, typename T
				, typename Hash
				, typename Pred
				, typename Allocator
				, typename SharedMutex
			>
				friend struct hop_hash_base;
		};

	template <typename HopHash>
		struct hop_hash_const_iterator
			: public hop_hash_iterator_impl<HopHash>
		{
		private:
			using segment_and_bucket_t = typename HopHash::segment_and_bucket_t;
			using typename hop_hash_iterator_impl<HopHash>::base;
		public:
			hop_hash_const_iterator(const segment_and_bucket_t & sb_, owner::index_type ix_)
				: hop_hash_iterator_impl<HopHash>(sb_, ix_)
			{}
			hop_hash_const_iterator(typename HopHash::size_type i)
				: hop_hash_iterator_impl<HopHash>(i)
			{}
			/* HopHash 106 (Iterator) */
			auto operator*() const -> const typename base::reference
			{
				return this->deref();
			}
			auto operator++() -> hop_hash_const_iterator &
			{
				this->incr();
				return *this;
			}
			/* HopHash 17 (EqualityComparable) */
			/* HopHash 107 (InputIterator) */
			auto operator->() const -> const typename base::pointer
			{
				return &this->deref();
			}
			hop_hash_const_iterator operator++(int)
			{
				auto ti = *this;
				this->incr();
				return ti;
			}
			/* HopHash 109 (ForwardIterator) - handled by 107 */
		};

	template <typename HopHash>
		bool operator==(
			const hop_hash_local_iterator_impl<HopHash> a
			, const hop_hash_local_iterator_impl<HopHash> b
		)
		{
			return a._sb_owner == b._sb_owner && a._content_index == b._content_index;
		}

	template <typename HopHash>
		bool operator!=(
			const hop_hash_local_iterator_impl<HopHash> a
			, const hop_hash_local_iterator_impl<HopHash> b
		)
		{
			return !(a==b);
		}

	template <typename HopHash>
		bool operator==(
			const hop_hash_iterator_impl<HopHash> a
			, const hop_hash_iterator_impl<HopHash> b
		)
		{
			return a._sb_owner == b._sb_owner && a._content_index == b._content_index;
		}

	template <typename HopHash>
		bool operator!=(
			const hop_hash_iterator_impl<HopHash> a
			, const hop_hash_iterator_impl<HopHash> b
		)
		{
			return !(a==b);
		}
}

#if TRACK_OWNER
#include "owner.tcc"
#endif
#include "persist_controller.tcc"
#include "hop_hash.tcc"

#if TRACED_CONTENT || TRACED_OWNER || TRACED_BUCKET
#include "hop_hash_debug.tcc"
#endif

#endif
