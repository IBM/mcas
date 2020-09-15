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

#include "heap_rc.h"
#include "devdax_manager.h"
#include "hstore_config.h"
#include <common/utils.h>
#include <cinttypes>

constexpr unsigned heap_rc_shared_ephemeral::log_min_alignment;
constexpr unsigned heap_rc_shared_ephemeral::hist_report_upper_bound;

heap_rc_shared_ephemeral::heap_rc_shared_ephemeral(
	unsigned // debug_level_
)
	: _heap()
	, _managed_regions()
	, _allocated(0)
	, _capacity(0)
	, _reconstituted()
	, _hist_alloc()
	, _hist_inject()
	, _hist_free()
{}

void heap_rc_shared_ephemeral::add_managed_region(const ::iovec &r, const unsigned numa_node)
{
	_heap.add_managed_region(r.iov_base, r.iov_len, int(numa_node));
	_managed_regions.push_back(r);
	_capacity += r.iov_len;
}

void heap_rc_shared_ephemeral::inject_allocation(void *p_, std::size_t sz_, unsigned numa_node_)
{
	_heap.inject_allocation(p_, sz_, int(numa_node_));
	{
		auto pc = static_cast<alloc_set_t::element_type>(p_);
		_reconstituted.add(alloc_set_t::segment_type(pc, pc + sz_));
	}
	_allocated += sz_;
	_hist_alloc.enter(sz_);
}

void *heap_rc_shared_ephemeral::allocate(std::size_t sz_, unsigned _numa_node_, std::size_t alignment_)
{
	auto p = _heap.alloc(sz_, int(_numa_node_), alignment_);
	_allocated += sz_;
	_hist_alloc.enter(sz_);
	return p;
}

void heap_rc_shared_ephemeral::free(void *p_, std::size_t sz_, unsigned numa_node_)
{
	_heap.free(p_, int(numa_node_), sz_);
	_allocated -= sz_;
	_hist_free.enter(sz_);
}

bool heap_rc_shared_ephemeral::is_reconstituted(const void * p_) const
{
	return contains(_reconstituted, static_cast<alloc_set_t::element_type>(p_));
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
}

/* When used with ADO, this space apparently needs a 2MiB alignment.
 * 4 KiB produces sometimes produces a disagreement between server and AOo mappings
 * which manifest as incorrect key and data values as seen on the ADO side.
 */
heap_rc_shared::heap_rc_shared(unsigned debug_level_, void *pool_, std::size_t sz_, unsigned numa_node_)
	: _pool0{align(pool_, sz_, MiB(2))}
	, _numa_node(numa_node_)
	, _more_region_uuids_size(0)
	, _more_region_uuids()
	, _eph(std::make_unique<heap_rc_shared_ephemeral>(debug_level_))
{
	void *last = static_cast<char *>(pool_) + sz_;
	if ( 0 < debug_level_ )
	{
		PLOG("%s: split %p .. %p) into segments", __func__, pool_, last);
	}

	if ( 0 < debug_level_ )
	{
		PLOG("i%s: region %p: 0x%zx", __func__, _pool0.iov_base, _pool0.iov_len);
	}
	/* cursor now locates the best-aligned region */
	_eph->add_managed_region(_pool0, _numa_node);
	hop_hash_log<trace_heap_summary>::write(
		LOG_LOCATION
		, " pool ", _pool0.iov_base, " .. ", iov_limit(_pool0)
		, " size ", _pool0.iov_len
		, " new"
	);
	VALGRIND_CREATE_MEMPOOL(_pool0.iov_base, 0, false);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winit-self"
#pragma GCC diagnostic ignored "-Wuninitialized"
heap_rc_shared::heap_rc_shared(unsigned debug_level_, const std::unique_ptr<Devdax_manager> &devdax_manager_)
	: _pool0(this->_pool0)
	, _numa_node(this->_numa_node)
	, _more_region_uuids_size(this->_more_region_uuids_size)
	, _more_region_uuids(this->_more_region_uuids)
	, _eph(std::make_unique<heap_rc_shared_ephemeral>(debug_level_))
{
	_eph->add_managed_region(_pool0, _numa_node);
	hop_hash_log<trace_heap_summary>::write(
		LOG_LOCATION
		, " pool ", _pool0.iov_base, " .. ", iov_limit(_pool0)
		, " size ", _pool0.iov_len
		, " reconstituting"
	);

	VALGRIND_MAKE_MEM_DEFINED(_pool0.iov_base, _pool0.iov_len);
	VALGRIND_CREATE_MEMPOOL(_pool0.iov_base, 0, true);

	for ( std::size_t i = 0; i != _more_region_uuids_size; ++i )
	{
		auto r = open_region(devdax_manager_, _more_region_uuids[i], _numa_node);
		_eph->add_managed_region(r, _numa_node);
		VALGRIND_MAKE_MEM_DEFINED(r.iov_base, r.iov_len);
		VALGRIND_CREATE_MEMPOOL(r.iov_base, 0, true);
	}
}
#pragma GCC diagnostic pop

heap_rc_shared::~heap_rc_shared()
{
	quiesce();
}

::iovec heap_rc_shared::open_region(const std::unique_ptr<Devdax_manager> &devdax_manager_, std::uint64_t uuid_, unsigned numa_node_)
{
	::iovec iov;
	iov.iov_base = devdax_manager_->open_region(uuid_, numa_node_, &iov.iov_len);
	if ( iov.iov_base == 0 )
	{
		throw std::range_error("failed to re-open region " + std::to_string(uuid_));
	}
	return iov;
}

std::vector<::iovec> heap_rc_shared::regions() const
{
	return _eph->get_managed_regions();
}

void *heap_rc_shared::iov_limit(const ::iovec &r)
{
	return static_cast<char *>(r.iov_base) + r.iov_len;
}

auto heap_rc_shared::grow(
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
					_eph->add_managed_region(r, _numa_node);
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
	return _eph->capacity();
}

void heap_rc_shared::quiesce()
{
	hop_hash_log<trace_heap_summary>::write(LOG_LOCATION, " size ", _pool0.iov_len, " allocated ", _eph->allocated());
	_eph->write_hist<trace_heap_summary>(_pool0);
	VALGRIND_DESTROY_MEMPOOL(_pool0.iov_base);
	VALGRIND_MAKE_MEM_UNDEFINED(_pool0.iov_base, _pool0.iov_len);
	_eph.reset(nullptr);
}

namespace
{
	/* Round up to (ceiling) power of 2, from Hacker's Delight 3-2 */
	std::size_t clp2(std::size_t sz_)
	{
		if ( sz_ != 0 )
		{
			--sz_;
			sz_ |= sz_ >> 1;
			sz_ |= sz_ >> 2;
			sz_ |= sz_ >> 4;
			sz_ |= sz_ >> 8;
			sz_ |= sz_ >> 16;
			sz_ |= sz_ >> 32;
		}
		return sz_ + 1;
	}
}

void *heap_rc_shared::alloc(const std::size_t sz_, const std::size_t alignment_)
{
	auto alignment = std::max(alignment_, sizeof(void *));

	if ( (alignment & (alignment - 1U)) != 0 )
	{
		throw std::invalid_argument("alignment is not a power of 2");
	}

	auto sz = sz_;

	if ( sz < alignment )
	{
		/* round up only to a power of 2, so Rca_LB will find the element
		 * on free.
		 */
		sz = clp2(sz);
		assert( (sz & (sz - 1)) == 0 );
		/* Allocation must be a multiple of alignment. In the case,
		 * adjust alignment. */
		alignment = std::max(sizeof(void *), sz);
	}

	/* In any case, sz must be a multiple of alignment. */
	sz = (sz + alignment - 1U)/alignment * alignment;

	try {
		auto p = _eph->allocate(sz, _numa_node, alignment);
		/* Note: allocation exception from Rca_LB is General_exception, which does not derive
		 * from std::bad_alloc.
		 */

		VALGRIND_MEMPOOL_ALLOC(_pool0.iov_base, p, sz);
		hop_hash_log<trace_heap>::write(LOG_LOCATION, "pool ", _pool0.iov_base, " addr ", p, " align ", alignment_, " -> ", alignment, " size ", sz_, " -> ", sz);
		return p;
	}
	catch ( const std::bad_alloc & )
	{
		_eph->write_hist<true>(_pool0);
		/* Sometimes lack of space will cause heap to throw a bad_alloc. */
		throw;
	}
	catch ( const General_exception &e )
	{
		_eph->write_hist<true>(_pool0);
		/* Sometimes lack of space will cause heap to throw a General_exception with this explanation. */
		/* Convert to bad_alloc. */
		if ( e.cause() == std::string("region allocation out-of-space") )
		{
			throw std::bad_alloc();
		}
		throw;
	}
}

void heap_rc_shared::inject_allocation(const void * p, std::size_t sz_)
{
	auto alignment = sizeof(void *);
	sz_ = std::max(sz_, alignment);
	auto sz = (sz_ + alignment - 1U)/alignment * alignment;
	/* NOTE: inject_allocation should take a const void* */
	_eph->inject_allocation(const_cast<void *>(p), sz, _numa_node);
	VALGRIND_MEMPOOL_ALLOC(_pool0.iov_base, p, sz);
	hop_hash_log<trace_heap>::write(LOG_LOCATION, "pool ", _pool0.iov_base, " addr ", p, " size ", sz);
}

void heap_rc_shared::free(void *p_, std::size_t sz_, std::size_t alignment_)
{
	alignment_ = std::max(alignment_, sizeof(void *));
	sz_ = std::max(sz_, alignment_);
	auto sz = (sz_ + alignment_ - 1U)/alignment_ * alignment_;
	VALGRIND_MEMPOOL_FREE(_pool0.iov_base, p_);
	hop_hash_log<trace_heap>::write(LOG_LOCATION, "pool ", _pool0.iov_base, " addr ", p_, " size ", sz);
	return _eph->free(p_, sz, _numa_node);
}

bool heap_rc_shared::is_reconstituted(const void * p_) const
{
	return _eph->is_reconstituted(p_);
}
