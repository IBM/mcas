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


#ifndef MCAS_HSTORE_HEAP_CC_H
#define MCAS_HSTORE_HEAP_CC_H

#include "as_emplace.h"
#include "cptr.h"
#include "histogram_log2.h"
#include "hop_hash_log.h"
#include "persistent.h"
#include "persister_nupm.h"
#include "trace_flags.h"

#include <boost/icl/interval_set.hpp>
#if 0
#include <valgrind/memcheck.h>
#else
#define VALGRIND_CREATE_MEMPOOL(pool, x, y) do {} while(0)
#define VALGRIND_DESTROY_MEMPOOL(pool) do {} while(0)
#define VALGRIND_MAKE_MEM_DEFINED(pool, size) do {} while(0)
#define VALGRIND_MAKE_MEM_UNDEFINED(pool, size) do {} while(0)
#define VALGRIND_MEMPOOL_ALLOC(pool, addr, size) do {} while(0)
#define VALGRIND_MEMPOOL_FREE(pool, size) do {} while(0)
#endif
#include <ccpm/interfaces.h>
#include <common/exceptions.h> /* General_exception */
#include <common/logging.h> /* log_source */

#include <sys/uio.h> /* iovec */

#include <algorithm>
#include <array>
#include <cstddef> /* size_t, ptrdiff_t */
#include <memory>
#include <vector>

class Devdax_manager;

namespace impl
{
	struct allocation_state_pin;
	struct allocation_state_extend;
}

namespace ccpm
{
	class IHeap_expandable;
	struct region_vector_t;
}

template <typename Foo, typename Persister>
	struct deallocator_cc;

template <typename T, std::size_t SmallLimit, typename Allocator>
	union persist_fixed_string;

struct heap_cc_shared_ephemeral
  : private common::log_source
{
private:
	std::unique_ptr<ccpm::IHeap_expandable> _heap;
	std::vector<::iovec> _managed_regions;
	std::size_t _capacity;
	std::size_t _allocated;
	impl::allocation_state_emplace *_ase;
	impl::allocation_state_pin *_aspd;
	impl::allocation_state_pin *_aspk;
	impl::allocation_state_extend *_asx;

	using hist_type = util::histogram_log2<std::size_t>;
	hist_type _hist_alloc;
	hist_type _hist_inject;
	hist_type _hist_free;

	static constexpr unsigned log_min_alignment = 3U; /* log (sizeof(void *)) */
	static_assert(sizeof(void *) == 1U << log_min_alignment, "log_min_alignment does not match sizeof(void *)");
	/* Rca_LB seems not to allocate at or above about 2GiB. Limit reporting to 16 GiB. */
	static constexpr unsigned hist_report_upper_bound = 34U;

	void add_managed_region(const ::iovec &r);
	explicit heap_cc_shared_ephemeral(
		unsigned debug_level_
		, impl::allocation_state_emplace *ase
		, impl::allocation_state_pin *aspd
		, impl::allocation_state_pin *aspk
		, impl::allocation_state_extend *asx
		, std::unique_ptr<ccpm::IHeap_expandable> p
		, const ccpm::region_vector_t &rv
	);
	std::vector<::iovec> get_managed_regions() const { return _managed_regions; }

	template <bool B>
		void write_hist(const ::iovec & pool_) const
		{
			static bool suppress = false;
			if ( ! suppress )
			{
				hop_hash_log<B>::write(LOG_LOCATION, "pool ", pool_.iov_base);
				std::size_t lower_bound = 0;
				auto limit = std::min(std::size_t(hist_report_upper_bound), _hist_alloc.data().size());
				for ( unsigned i = log_min_alignment; i != limit; ++i )
				{
					const std::size_t upper_bound = 1ULL << i;
					hop_hash_log<B>::write(LOG_LOCATION
						, "[", lower_bound, "..", upper_bound, "): "
						, _hist_alloc.data()[i], " ", _hist_inject.data()[i], " ", _hist_free.data()[i]
						, " "
					);
					lower_bound = upper_bound;
				}
				suppress = true;
			}
		}
public:
	friend struct heap_cc_shared;

	using common::log_source::debug_level;
	explicit heap_cc_shared_ephemeral(
		unsigned debug_level
		, impl::allocation_state_emplace *ase
		, impl::allocation_state_pin *aspd
		, impl::allocation_state_pin *aspk
		, impl::allocation_state_extend *asx
		, const ccpm::region_vector_t &rv_
	);
	explicit heap_cc_shared_ephemeral(
		unsigned debug_level
		, impl::allocation_state_emplace *ase
		, impl::allocation_state_pin *aspd
		, impl::allocation_state_pin *aspk
		, impl::allocation_state_extend *asx
		, const ccpm::region_vector_t &rv_
		, ccpm::ownership_callback_t f
	);
	std::size_t free(persistent_t<void *> *p_, std::size_t sz_);
	heap_cc_shared_ephemeral(const heap_cc_shared_ephemeral &) = delete;
	heap_cc_shared_ephemeral& operator=(const heap_cc_shared_ephemeral &) = delete;
};

struct heap_cc_shared
{
private:
	::iovec _pool0;
	unsigned _numa_node;
	std::size_t _more_region_uuids_size;
	std::array<std::uint64_t, 1024U> _more_region_uuids;
	std::unique_ptr<heap_cc_shared_ephemeral> _eph;

public:
	explicit heap_cc_shared(
		unsigned debug_level
		, uint64_t pool0_uuid
		, const std::unique_ptr<Devdax_manager> &devdax_manager_
	);
	explicit heap_cc_shared(
		unsigned debug_level
		, impl::allocation_state_emplace *ase
		, impl::allocation_state_pin *aspd
		, impl::allocation_state_pin *aspk
		, impl::allocation_state_extend *asx
		, void *p
		, std::size_t sz
		, unsigned numa_node
	);
	explicit heap_cc_shared(
		unsigned debug_level
		, const std::unique_ptr<Devdax_manager> &devdax_manager
		, impl::allocation_state_emplace *ase
		, impl::allocation_state_pin *aspd
		, impl::allocation_state_pin *aspk
		, impl::allocation_state_extend *asx
	);
#if 0
	explicit heap_cc_shared(const ccpm::region_vector_t &rv_);
#endif
	heap_cc_shared(const heap_cc_shared &) = delete;
	heap_cc_shared &operator=(const heap_cc_shared &) = delete;

	~heap_cc_shared();

	static void *iov_limit(const ::iovec &r);

	auto grow(
		const std::unique_ptr<Devdax_manager> & devdax_manager_
		, std::uint64_t uuid_
		, std::size_t increment_
	) -> std::size_t;

	void quiesce();

	void alloc(persistent_t<void *> *p, std::size_t sz, std::size_t alignment);
	void free(persistent_t<void *> *p, std::size_t sz);

	void emplace_arm() const { _eph->_ase->arm(persister_nupm()); }
	void emplace_disarm() const { _eph->_ase->disarm(persister_nupm()); }

	auto &aspd() const { return *_eph->_aspd; }
	auto &aspk() const { return *_eph->_aspk; }
	void pin_data_arm(cptr &cptr) const;
	void pin_key_arm(cptr &cptr) const;

	char *pin_data_get_cptr() const;
	char *pin_key_get_cptr() const;
	void pin_data_disarm() const;
	void pin_key_disarm() const;
	void extend_arm() const;
	void extend_disarm() const;

	unsigned percent_used() const
	{
		return
			unsigned(
				_eph->_capacity
				? _eph->_allocated * 100U / _eph->_capacity
				: 100U
			);
	}

	std::vector<::iovec> regions() const;
};

struct heap_cc
{
private:
	heap_cc_shared *_heap;

public:
	explicit heap_cc(heap_cc_shared *area)
		: _heap(area)
	{
	}

	~heap_cc()
	{
	}

	heap_cc(const heap_cc &) noexcept = default;

	heap_cc & operator=(const heap_cc &) = default;

	static constexpr std::uint64_t magic_value = 0x7c84297de2de94a3;

	heap_cc_shared *operator->() const
	{
		return _heap;
	}
};

#endif
