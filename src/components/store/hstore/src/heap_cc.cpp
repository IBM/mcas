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

#include "hstore_config.h"
#include "heap_cc.h"

#include "as_pin.h"
#include "as_emplace.h"
#include "as_extend.h"
#include "devdax_manager.h"
#include <ccpm/cca.h>
#include <common/utils.h> /* round_up */
#include <algorithm>
#include <cassert>
#include <cstdlib> /* getenv */
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

constexpr unsigned heap_cc_shared_ephemeral::log_min_alignment;
constexpr unsigned heap_cc_shared_ephemeral::hist_report_upper_bound;

namespace
{
#if USE_CC_HEAP == 4
	auto leak_check_str = std::getenv("LEAK_CHECK");
	bool leak_check = bool(leak_check_str);
#endif
}

heap_cc_shared_ephemeral::heap_cc_shared_ephemeral(
	unsigned debug_level_
	, impl::allocation_state_emplace *ase_
	, impl::allocation_state_pin *aspd_
	, impl::allocation_state_pin *aspk_
	, impl::allocation_state_extend *asx_
	, std::unique_ptr<ccpm::IHeap_expandable> p
	, const ccpm::region_vector_t &rv_
)
	: common::log_source(debug_level_)
	, _heap(std::move(p))
	, _managed_regions(rv_.begin(), rv_.end())
	, _capacity(std::accumulate(rv_.begin(), rv_.end(), ::iovec{nullptr, 0}, [] (const auto &a, const auto &b) -> ::iovec { return {nullptr, a.iov_len + b.iov_len}; }).iov_len)
	, _allocated(
		[this] ()
		{
			std::size_t r;
			auto rc = _heap->remaining(r);
			return _capacity - (rc == S_OK ? r : 0);
		} ()
	)
	, _ase(ase_)
	, _aspd(aspd_)
	, _aspk(aspk_)
	, _asx(asx_)
	, _hist_alloc()
	, _hist_inject()
	, _hist_free()
{}

heap_cc_shared_ephemeral::heap_cc_shared_ephemeral(
	unsigned debug_level_
	, impl::allocation_state_emplace *ase_
	, impl::allocation_state_pin *aspd_
	, impl::allocation_state_pin *aspk_
	, impl::allocation_state_extend *asx_
	, const ccpm::region_vector_t &rv_
	)
	: heap_cc_shared_ephemeral(debug_level_, ase_, aspd_, aspk_, asx_, std::make_unique<ccpm::cca>(rv_), rv_)
{}

heap_cc_shared_ephemeral::heap_cc_shared_ephemeral(
	unsigned debug_level_
	, impl::allocation_state_emplace *ase_
	, impl::allocation_state_pin *aspd_
	, impl::allocation_state_pin *aspk_
	, impl::allocation_state_extend *asx_
	, const ccpm::region_vector_t &rv_
	, ccpm::ownership_callback_t f
)
	: heap_cc_shared_ephemeral(debug_level_, ase_, aspd_, aspk_, asx_, std::make_unique<ccpm::cca>(rv_, f), rv_)
{}

void heap_cc_shared_ephemeral::add_managed_region(const ::iovec &r_)
{
	ccpm::region_vector_t rv(r_);
	_heap->add_regions(rv);
	for ( const auto &r : rv )
	{
		_managed_regions.push_back(r);
		_capacity += r.iov_len;
	}
}

std::size_t heap_cc_shared_ephemeral::free(persistent_t<void *> *p_, std::size_t sz_)
{
	/* Our free does not know the true size, because alignment is not known.
	 * But the pool free will know, as it can see how much has been allocated.
	 *
	 * The free, however, does not return a size. Pretend that it does.
	 */
#if USE_CC_HEAP == 4
	/* Note: order of testing is important. An extend arm+allocate) can occur while
	 * emplace is armed, but not vice-versa
	 */
	if ( _asx->is_armed() )
	{
		CPLOG(1, PREFIX "unexpected segment deallocation of %p of %zu", LOCATION, persistent_ref(*p_), sz_);
		abort();
#if 0
		_asx->record_deallocation(&persistent_ref(*p_), persister_nupm());
#endif
	}
	else if ( _ase->is_armed() )
	{
		_ase->record_deallocation(&persistent_ref(*p_), persister_nupm());
	}
	else
	{
		CPLOG(1, PREFIX "leaky deallocation of %p of %zu", LOCATION, persistent_ref(*p_), sz_);
	}
#endif
	/* IHeap interface does not support abstract pointers. Cast to regular pointer */
	auto sz = (_heap->free(*reinterpret_cast<void **>(p_), sz_), sz_);
	/* We would like to carry the persistent_t through to the crash-conssitent allocator,
	 * but for now just assume that the allocator has modifed p_, and call tick to indicate that.
	 */
	perishable::tick();
	assert(sz <= _allocated);
	_allocated -= sz;
	_hist_free.enter(sz);
	return sz;
}

namespace
{
	::iovec align(void *first_, std::size_t sz_, std::size_t alignment_)
	{
		auto first = reinterpret_cast<uintptr_t>(first_);
		auto last = first + sz_;
		/* It may not even be the case that Rca_LB does not need managed regions
		 * aligned at all,
		 * as the allocated slabs are aligned even if the region is not.
		 * Some part of ado processing, though, map try mmap the area, which means
		 * that the are must be aligned and sized to page multiples (4K or maybe 2M).
		 */
		last = round_down(last, alignment_);
		auto c = round_up_t(first, alignment_);
		/* It may not even be the case that managed regions need to be aligned at all,
		 * as the allocated slabs are aligned even if the region is not.
		 */
		if ( last <= c )
		{
			throw std::runtime_error("Insufficent size for managed region");
		}
		return ::iovec{reinterpret_cast<void *>(c), last - c};
	}

	::iovec open_region(const std::unique_ptr<Devdax_manager> &devdax_manager_, std::uint64_t uuid_, unsigned numa_node_)
	{
		::iovec iov;
		iov.iov_base = devdax_manager_->open_region(uuid_, numa_node_, &iov.iov_len);
		if ( iov.iov_base == 0 )
		{
			throw std::range_error("failed to re-open region " + std::to_string(uuid_));
		}
		return iov;
	}

	const ccpm::region_vector_t add_regions(ccpm::region_vector_t &&rv_, const std::unique_ptr<Devdax_manager> &devdax_manager_, unsigned numa_node_, std::uint64_t *first_, std::uint64_t *last_)
	{
		auto v = std::move(rv_);
		for ( auto it = first_; it != last_; ++it )
		{
			auto r = open_region(devdax_manager_, *it, numa_node_);
			VALGRIND_MAKE_MEM_DEFINED(r.iov_base, r.iov_len);
			VALGRIND_CREATE_MEMPOOL(r.iov_base, 0, true);
			v.push_back(r);
		}
		return v;
	}
}

/* When used with ADO, this space apparently needs a 2MiB alignment.
 * 4 KiB produces sometimes produces a disagreement between server and AOo mappings
 * which manifest as incorrect key and data values as seen on the ADO side.
 */
heap_cc_shared::heap_cc_shared(
	unsigned debug_level_
	, impl::allocation_state_emplace *ase_
	, impl::allocation_state_pin *aspd_
	, impl::allocation_state_pin *aspk_
	, impl::allocation_state_extend *asx_
	, void *p
	, std::size_t sz
	, unsigned numa_node_
)
	: _pool0(align(p, sz, MiB(2)))
	, _numa_node(numa_node_)
	, _more_region_uuids_size(0)
	, _more_region_uuids()
	, _eph(
		std::make_unique<heap_cc_shared_ephemeral>(
			debug_level_
			, ase_
			, aspd_
			, aspk_
			, asx_
			, ccpm::region_vector_t(_pool0.iov_base, _pool0.iov_len)
		)
	)
{
	/* cursor now locates the best-aligned region */
	hop_hash_log<trace_heap_summary>::write(
		LOG_LOCATION
		, " pool ", _pool0.iov_base, " .. ", iov_limit(_pool0)
		, " size ", _pool0.iov_len
		, " new"
	);
	VALGRIND_CREATE_MEMPOOL(_pool0.iov_base, 0, false);
}

#if 0
heap_cc_shared::heap_cc_shared(unsigned debug_level_, uint64_t pool0_uuid_, const std::unique_ptr<Devdax_manager> &devdax_manager_, unsigned numa_node_)
	: _pool0(align(open_region(devdax_manager_, pool0_uuid_, numa_node_)))
	, _more_region_uuids_size(0)
	, _more_region_uuids()
	, _eph(
		std::make_unique<heap_cc_shared_ephemeral>(
			debug_level_, add_regions(ccpm::region_vector_t(_pool0.iov_base, _pool0.iov_len), devdax_manager_, numa_node_, nullptr, nullptr)
		)
	)
{
	/* cursor now locates the best-aligned region */
	hop_hash_log<trace_heap_summary>::write(
		LOG_LOCATION
		, " pool ", _pool0.iov_base, " .. ", iov_limit(_pool0)
		, " size ", _pool0.iov_len
		, " new"
	);
	VALGRIND_CREATE_MEMPOOL(_pool0.iov_base, 0, false);
}
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winit-self"
#pragma GCC diagnostic ignored "-Wuninitialized"
heap_cc_shared::heap_cc_shared(
	unsigned debug_level_
	, const std::unique_ptr<Devdax_manager> &devdax_manager_
	, impl::allocation_state_emplace *ase_
	, impl::allocation_state_pin *aspd_
	, impl::allocation_state_pin *aspk_
	, impl::allocation_state_extend *asx_
)
	: _pool0(this->_pool0)
	, _numa_node(this->_numa_node)
	, _more_region_uuids_size(this->_more_region_uuids_size)
	, _more_region_uuids(this->_more_region_uuids)
	, _eph(
		std::make_unique<heap_cc_shared_ephemeral>(
			debug_level_
			, ase_
			, aspd_
			, aspk_
			, asx_
			, add_regions(
				ccpm::region_vector_t(
					_pool0.iov_base, _pool0.iov_len
				)
				, devdax_manager_
				, _numa_node
				, &_more_region_uuids[0]
				, &_more_region_uuids[_more_region_uuids_size]
			)
			, [ase_, aspd_, aspk_, asx_] (const void *p) -> bool {
				/* To answer whether the map or the allocator owns pointer p?
				 * Guessing that true means that the map owns p
				 */
				auto cp = const_cast<void *>(p);
				return ase_->is_in_use(cp) || aspd_->is_in_use(p) || aspk_->is_in_use(p) || asx_->is_in_use(p);
			}
		)
	)
{
	hop_hash_log<trace_heap_summary>::write(
		LOG_LOCATION
		, " pool ", _pool0.iov_base, " .. ", iov_limit(_pool0)
		, " size ", _pool0.iov_len
		, " reconstituting"
	);
	VALGRIND_MAKE_MEM_DEFINED(_pool0.iov_base, _pool0.iov_len);
	VALGRIND_CREATE_MEMPOOL(_pool0.iov_base, 0, true);
}
#pragma GCC diagnostic pop

heap_cc_shared::~heap_cc_shared()
{
	quiesce();
}

std::vector<::iovec> heap_cc_shared::regions() const
{
	return _eph->get_managed_regions();
}

void *heap_cc_shared::iov_limit(const ::iovec &r)
{
	return static_cast<char *>(r.iov_base) + r.iov_len;
}

auto heap_cc_shared::grow(
	const std::unique_ptr<Devdax_manager> & devdax_manager_
	, std::uint64_t uuid_
	, std::size_t increment_
) -> std::size_t
{
	if ( 0 < increment_ )
	{
		if ( _more_region_uuids_size == _more_region_uuids.size() )
		{
			throw std::bad_alloc(); /* max # of regions used */
		}
		const auto hstore_grain_size = std::size_t(1) << (HSTORE_LOG_GRAIN_SIZE);
		auto size = ( (increment_ - 1) / hstore_grain_size + 1 ) * hstore_grain_size;
		auto uuid = _more_region_uuids_size == 0 ? uuid_ : _more_region_uuids[_more_region_uuids_size-1];
		auto uuid_next = uuid + 1;
		for ( ; uuid_next != uuid; ++uuid_next )
		{
			if ( uuid_next != 0 )
			{
				try
				{
					/* Note: crash between here and "Slot persist done" may cause devdax_manager_
					 * to leak the region.
					 */
					::iovec r { devdax_manager_->create_region(uuid_next, _numa_node, size), size };
					{
						auto &slot = _more_region_uuids[_more_region_uuids_size];
						slot = uuid_next;
						persister_nupm::persist(&slot, sizeof slot);
						/* Slot persist done */
					}
					{
						++_more_region_uuids_size;
						persister_nupm::persist(&_more_region_uuids_size, _more_region_uuids_size);
					}
					_eph->add_managed_region(r);
					hop_hash_log<trace_heap_summary>::write(
						LOG_LOCATION
						, " pool ", r.iov_base, " .. ", iov_limit(r)
						, " size ", r.iov_len
						, " grow"
					);
					break;
				}
				catch ( const std::bad_alloc & )
				{
					/* probably means that the uuid is in use */
				}
				catch ( const General_exception & )
				{
					/* probably means that the space cannot be allocated */
					throw std::bad_alloc();
				}
			}
		}
		if ( uuid_next == uuid )
		{
			throw std::bad_alloc(); /* no more UUIDs */
		}
	}
	return _eph->_capacity;
}

void heap_cc_shared::quiesce()
{
	hop_hash_log<trace_heap_summary>::write(LOG_LOCATION, " size ", _pool0.iov_len, " allocated ", _eph->_allocated);
	VALGRIND_DESTROY_MEMPOOL(_pool0.iov_base);
	VALGRIND_MAKE_MEM_UNDEFINED(_pool0.iov_base, _pool0.iov_len);
	_eph->write_hist<trace_heap_summary>(_pool0);
	_eph.reset(nullptr);
}

void heap_cc_shared::alloc(persistent_t<void *> *p_, std::size_t sz_, std::size_t alignment_)
{
	alignment_ = std::max(alignment_, sizeof(void *));

	if ( (alignment_ & (alignment_ - 1U)) != 0 )
	{
		throw std::invalid_argument("alignment is not a power of 2");
	}

	/* allocation must be multiple of alignment */
	auto sz = (sz_ + alignment_ - 1U)/alignment_ * alignment_;

	try {
#if USE_CC_HEAP == 4
		if ( _eph->_aspd->is_armed() )
		{
		}
		else if ( _eph->_aspk->is_armed() )
		{
		}
		/* Note: order of testing is important. An extend arm+allocate) can occur while
		 * emplace is armed, but not vice-versa
		 */
		else if ( _eph->_asx->is_armed() )
		{
			_eph->_asx->record_allocation(&persistent_ref(*p_), persister_nupm());
		}
		else if ( _eph->_ase->is_armed() )
		{
			_eph->_ase->record_allocation(&persistent_ref(*p_), persister_nupm());
		}
		else
		{
			if ( leak_check )
			{
				PLOG(PREFIX "leaky allocation, size %zu", LOCATION, sz_);
			}
		}
#endif
		/* IHeap interface does not support abstract pointers. Cast to regular pointer */
		_eph->_heap->allocate(*reinterpret_cast<void **>(p_), sz, alignment_);
		/* We would like to carry the persistent_t through to the crash-conssitent allocator,
		 * but for now just assume that the allocator has modifed p_, and call tick ti indicate that.
		 */
		perishable::tick();

		VALGRIND_MEMPOOL_ALLOC(_pool0.iov_base, p_, sz);
		/* size grows twice: once for aligment, and possibly once more in allocation */
		hop_hash_log<trace_heap>::write(LOG_LOCATION, "pool ", _pool0.iov_base, " addr ", p_, " size ", sz_, "->", sz);
		_eph->_allocated += sz;
		_eph->_hist_alloc.enter(sz);
	}
	catch ( const std::bad_alloc & )
	{
		_eph->write_hist<true>(_pool0);
		/* Sometimes lack of space will cause heap to throw a bad_alloc. */
		throw;
	}
}

void heap_cc_shared::free(persistent_t<void *> *p_, std::size_t sz_)
{
	VALGRIND_MEMPOOL_FREE(_pool0.iov_base, p_);
	auto sz = _eph->free(p_, sz_);
	hop_hash_log<trace_heap>::write(LOG_LOCATION, "pool ", _pool0.iov_base, " addr ", p_, " size ", sz_, "->", sz);
}

void heap_cc_shared::extend_arm() const
{
	_eph->_asx->arm(persister_nupm());
}

void heap_cc_shared::extend_disarm() const
{
	_eph->_asx->disarm(persister_nupm());
}

void heap_cc_shared::pin_data_arm(
	cptr &cptr_
) const
{
#if USE_CC_HEAP == 4
	_eph->_aspd->arm(cptr_, persister_nupm());
#else
	(void)cptr_;
#endif
}

void heap_cc_shared::pin_key_arm(
	cptr &cptr_
) const
{
#if USE_CC_HEAP == 4
	_eph->_aspk->arm(cptr_, persister_nupm());
#else
	(void)cptr_;
#endif
}

char *heap_cc_shared::pin_data_get_cptr() const
{
#if USE_CC_HEAP == 4
	assert(_eph->_aspd->is_armed());
	return _eph->_aspd->get_cptr();
#else
	return nullptr;
#endif
}
char *heap_cc_shared::pin_key_get_cptr() const
{
#if USE_CC_HEAP == 4
	assert(_eph->_aspk->is_armed());
	return _eph->_aspk->get_cptr();
#else
	return nullptr;
#endif
}

void heap_cc_shared::pin_data_disarm() const
{
	_eph->_aspd->disarm(persister_nupm());
}

void heap_cc_shared::pin_key_disarm() const
{
	_eph->_aspk->disarm(persister_nupm());
}
