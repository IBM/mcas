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

#include "store_map.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <ccpm/value_tracked.h>
#include <ccpm/container_cc.h>
#include <common/utils.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Weffc++"
#include <EASTL/bitset.h>
#include <EASTL/iterator.h>
#include <EASTL/list.h>
#include <EASTL/vector.h>
#pragma GCC diagnostic pop
#include <api/kvstore_itf.h>

#include <algorithm> // equal, reverse
#include <cstddef> // size_t
#include <cstdlib> // getenv
#include <iostream> // cerr
#include <memory> // shared_ptr
#include <string> // string

#include <ccpm/container_cc.h>

using namespace component;

using logged_int = ccpm::value_tracked<int, ccpm::tracker_log>;
using logged_ptr_to_int = ccpm::value_tracked<int *, ccpm::tracker_log>;
using logged_shared_ptr_to_int = ccpm::value_tracked<std::shared_ptr<int>, ccpm::tracker_log>;

// The fixture for testing class Foo.
class Log_test : public ::testing::Test
{
protected:

	component::IKVStore *instantiate()
	{
		/* create object instance through factory */
		auto link_library = "libcomponent-" + store_map::impl->name + ".so";
		auto comp =
			component::load_component(
				link_library,
				store_map::impl->factory_id
			);

		if ( comp )
		{
			auto fact = component::make_itf_ref(static_cast<IKVStore_factory *>(comp->query_interface(IKVStore_factory::iid())));
			/* numa node 0 */
			return fact->create(
              0
              , {
                  { +component::IKVStore_factory::k_name, "numa0"}
                  , { +component::IKVStore_factory::k_dax_config, store_map::location }
                }
            );
		}
		else
		{
			return nullptr;
		}
	}

	component::IKVStore::pool_t create_pool(
		component::IKVStore *kvstore
		, const std::string &name
		, std::size_t size
	)
	{
		/* remove any old pool */
		try
		{
			kvstore->delete_pool(name);
		}
		catch ( const Exception & ) {}

		auto pool = kvstore->create_pool(name, size, 0, 0);
		if ( 0 == int64_t(pool) )
		{
			std::cerr
				<< "Pool not created\n";
		}
		return pool;
	}

	void close_pool(
		component::IKVStore *kvstore
		, component::IKVStore::pool_t pool
	)
	{
		if ( pmem_effective )
		{
				kvstore->close_pool(pool);
		}
	}

	/* persistent memory if enabled at all, is simulated and not real */
	static bool pmem_simulated;
	/* persistent memory is effective (either real, indicated by no PMEM_IS_PMEM_FORCE or simulated by PMEM_IS_PMEM_FORCE 0 not 1 */
	static bool pmem_effective;

	static std::string pool_name()
	{
		return "/mnt/pmem0/pool/0/test-" + store_map::impl->name + store_map::numa_zone() + ".pool";
	}

};

bool Log_test::pmem_simulated = std::getenv("PMEM_IS_PMEM_FORCE");
bool Log_test::pmem_effective = ! std::getenv("PMEM_IS_PMEM_FORCE") || std::getenv("PMEM_IS_PMEM_FORCE") == std::string("0");

TEST_F(Log_test, CCBitset)
{
/* operations to test
 *   emplace()
 *   insert() - 5 versions
 *   erase()
 *   clear()
 *   assign() - 3 versions
 */
	auto kvstore = std::unique_ptr<component::IKVStore>(instantiate());
	ASSERT_TRUE(kvstore.get());
	auto pool = create_pool(kvstore.get(), pool_name(), MiB(10));
	ASSERT_LT(0, int64_t(pool));

	void *heap_area;
	std::size_t heap_size = MiB(5);
	component::IKVStore::key_t heap_lock{};
	{
    size_t alignment = 0;
		/* An odd "insert or locate" interface. */
		auto r = kvstore->lock(pool, "heap", component::IKVStore::STORE_LOCK_WRITE, heap_area, heap_size, alignment, heap_lock);
		ASSERT_EQ(S_OK_CREATED, r);
	}

	void *bitset_area;
	using cc_bitset = ccpm::container_cc<eastl::bitset<52, std::uint64_t, ccpm::allocator_tl::tracker_type>>;

	std::size_t bitset_size = sizeof(cc_bitset);
	component::IKVStore::key_t container_lock{};
	{
    size_t alignment = 0;
		/* An odd "insert or locate" interface. */
		auto r = kvstore->lock(pool, "vector", component::IKVStore::STORE_LOCK_WRITE, bitset_area, bitset_size, alignment, container_lock);
		ASSERT_EQ(S_OK_CREATED, r);
	}

	ccpm::cca mr(ccpm::region_vector_t(heap_area, heap_size));
	auto ccv = new (bitset_area) cc_bitset(mr);

	(*ccv->container)[3] = true;
	(*ccv->container)[4] = true;
	(*ccv->container)[5] = true;

	/* now 3 4 5 */
    ASSERT_EQ(3, ccv->container->count());
    ASSERT_EQ(false, (*ccv->container)[2]);
    ASSERT_EQ(true, (*ccv->container)[3]);
    ASSERT_EQ(true, (*ccv->container)[5]);
    ASSERT_EQ(false, (*ccv->container)[6]);
	ccv->commit();
	(*ccv->container)[7] = true;
	ASSERT_EQ(true, (*ccv->container)[7]);
    ASSERT_EQ(4, ccv->container->count());
	/* 3 4 5 7 */
	{
		ccv->container->flip(4);
		ASSERT_EQ(3, ccv->container->count());
		ccv->container->set(6);
	}
	ASSERT_EQ(4, ccv->container->count());
	ASSERT_EQ(true, ccv->container->test(3));
	ASSERT_EQ(true, ccv->container->test(5));
	ASSERT_EQ(true, ccv->container->test(6));
	ASSERT_EQ(true, ccv->container->test(7));
	/* 3 5 6 7 */
	ccv->container->set(2);
	/* 2 3 5 6 7 */
	/* Is the container as expected? */
	{
		ASSERT_EQ(5, ccv->container->count());
		std::vector<int> v{2,3,5,6,7};
		for ( auto i : v )
		{
			ASSERT_EQ(true, ccv->container->test(i));
			ASSERT_EQ(true, (*ccv->container)[i]);
		}
	}

	ccv->rollback();

	{
		/* Is the rolled back container as expected? */
		ASSERT_EQ(3, ccv->container->count());
		ASSERT_EQ(false, (*ccv->container)[2]);
		ASSERT_EQ(true, (*ccv->container)[3]);
		ASSERT_EQ(true, (*ccv->container)[5]);
		ASSERT_EQ(false, (*ccv->container)[6]);
	}

	/* placement delete */
	ccv->~cc_bitset();

	{
		auto r = kvstore->unlock(pool, container_lock);
		ASSERT_EQ(S_OK, r);
	}

	{
		auto r = kvstore->unlock(pool, heap_lock);
		ASSERT_EQ(S_OK, r);
	}
	close_pool(kvstore.get(), pool);
}

TEST_F(Log_test, CCVectorOfComposite)
{
/* operations to test
 *   emplace()
 *   insert() - 5 versions
 *   erase()
 *   clear()
 *   assign() - 3 versions
 */
	auto kvstore = std::unique_ptr<component::IKVStore>(instantiate());
	ASSERT_TRUE(kvstore.get());
	auto pool = create_pool(kvstore.get(), pool_name(), MiB(10));
	ASSERT_LT(0, int64_t(pool));

	void *heap_area;
	std::size_t heap_size = MiB(5);
	component::IKVStore::key_t heap_lock{};
	{
    size_t alignment = 0;
		/* An odd "insert or locate" interface. */
		auto r = kvstore->lock(pool, "heap", component::IKVStore::STORE_LOCK_WRITE, heap_area, heap_size, alignment, heap_lock);
		ASSERT_EQ(S_OK_CREATED, r);
	}

	void *vector_area;

	using vt = std::shared_ptr<int>; /* the composite value */
	using logged_vt = ccpm::value_tracked<vt, ccpm::tracker_log>;

	using cc_vector = ccpm::container_cc<eastl::vector<logged_vt, ccpm::allocator_tl>>;

	std::size_t vector_size = sizeof(cc_vector);
	component::IKVStore::key_t container_lock{};
	{
    size_t alignment = 0;
		/* An odd "insert or locate" interface. */
		auto r = kvstore->lock(pool, "vector", component::IKVStore::STORE_LOCK_WRITE, vector_area, vector_size, alignment, container_lock);
		ASSERT_EQ(S_OK_CREATED, r);
	}

	ccpm::cca mr(
		ccpm::region_vector_t(
			ccpm::region_vector_t::value_type(
				common::make_byte_span(heap_area, heap_size)
			)
		)
	);
	auto ccv = new (vector_area) cc_vector(mr);

	const auto c2 = std::make_shared<int>(2);
	const auto c3 = std::make_shared<int>(3);
	const auto c4 = std::make_shared<int>(4);
	const auto c5 = std::make_shared<int>(5);
	const auto c6 = std::make_shared<int>(6);
	const auto c7 = std::make_shared<int>(7);
	std::vector<vt> original{c3, c4, c5};

	std::copy(original.begin(), original.end(), eastl::back_inserter(*ccv->container));

	ASSERT_EQ(3, ccv->container->size());
	ASSERT_TRUE(std::equal(ccv->container->begin(), ccv->container->end(), original.begin()));
	/* now 3 4 5 */
	ccv->commit();
	ccv->container->push_back(c7);
	ASSERT_EQ(4, ccv->container->size());
	ASSERT_TRUE(std::equal(ccv->container->begin(), ccv->container->end(), std::vector<vt>{c3,c4,c5,c7}.begin()));
	/* 3 4 5 7 */
	{
		auto it = ccv->container->begin(); // -> 3
		++it; // -> 4
		auto jt = ccv->container->erase(it); // -> 5
		ASSERT_EQ(c5, *jt);
		ASSERT_EQ(3, ccv->container->size());
		ASSERT_TRUE(std::equal(ccv->container->begin(), ccv->container->end(), std::vector<vt>{c3,c5,c7}.begin()));
		/* 3 5 7 */
		++jt;
		ccv->container->insert(jt, c6);
	}
	ASSERT_EQ(4, ccv->container->size());
	ASSERT_TRUE(std::equal(ccv->container->begin(), ccv->container->end(), std::vector<vt>{c3,c5,c6,c7}.begin()));
	/* 3 5 6 7 */
	ccv->container->insert(ccv->container->begin(), c2);
	/* 2 3 5 6 7 */
	/* Is the container as expected? */
	{
		std::vector<vt> mod_expected{c2,c3,c5,c6,c7};
		ASSERT_EQ(mod_expected.size(), ccv->container->size());
		ASSERT_TRUE(std::equal(ccv->container->begin(), ccv->container->end(), mod_expected.begin()));
	}

	ccv->rollback();

	{
		/* Is the rolled back container as expected? */
		ASSERT_EQ(original.size(), ccv->container->size());
		bool is_same = std::equal(ccv->container->begin(), ccv->container->end(), original.begin());

		if ( ! is_same )
		{
			for ( auto it = ccv->container->begin(); it != ccv->container->end(); ++it )
			{
				auto i = *it;
				std::cerr << "value: " << i << "\n";
			}
		}

		ASSERT_TRUE(is_same);
	}

	/* Reverse the elements, to exercise backward iterator (probably). */
	ccv->commit();

/* can use std::reverse only if EASTL uses standard iterator categories */
#if EASTL_STD_ITERATOR_CATEGORY_ENABLED
	std::reverse(ccv->container->begin(), ccv->container->end());
	{
		/* Is the reversal also as expected? */
		ASSERT_EQ(original.size(), ccv->container->size());
		ASSERT_TRUE(std::equal(ccv->container->rbegin(), ccv->container->rend(), original.begin()));
	}
#endif

	/* Undo the reversal */
	ccv->rollback();

	{
		/* Is the rolled back container as expected? */
		ASSERT_EQ(original.size(), ccv->container->size());
		auto is_same = std::equal(ccv->container->begin(), ccv->container->end(), original.begin());
		if ( ! is_same )
		{
			for ( auto it = ccv->container->begin(); it != ccv->container->end(); ++it )
			{
				auto i = *it;
				std::cerr << "value: " << i << "\n";
			}
		}
		ASSERT_TRUE(is_same);
	}

	/* placement delete */
	ccv->~cc_vector();

	{
		auto r = kvstore->unlock(pool, container_lock);
		ASSERT_EQ(S_OK, r);
	}

	{
		auto r = kvstore->unlock(pool, heap_lock);
		ASSERT_EQ(S_OK, r);
	}
	close_pool(kvstore.get(), pool);
}

TEST_F(Log_test, CCVector)
{
/* operations to test
 *   emplace()
 *   insert() - 5 versions
 *   erase()
 *   clear()
 *   assign() - 3 versions
 */
	auto kvstore = std::unique_ptr<component::IKVStore>(instantiate());
	ASSERT_TRUE(kvstore.get());
	auto pool = create_pool(kvstore.get(), pool_name(), MiB(10));
	ASSERT_LT(0, int64_t(pool));

	void *heap_area;
	std::size_t heap_size = MiB(5);
	component::IKVStore::key_t heap_lock{};
	{
    size_t alignment = 0;
		/* An odd "insert or locate" interface. */
		auto r = kvstore->lock(pool, "heap", component::IKVStore::STORE_LOCK_WRITE, heap_area, heap_size, alignment, heap_lock);
		ASSERT_EQ(S_OK_CREATED, r);
	}

	void *vector_area;

	using cc_vector = ccpm::container_cc<eastl::vector<logged_int, ccpm::allocator_tl>>;

	std::size_t vector_size = sizeof(cc_vector);
	component::IKVStore::key_t container_lock{};
	{
    size_t alignment = 0;
		/* An odd "insert or locate" interface. */
		auto r = kvstore->lock(pool, "vector", component::IKVStore::STORE_LOCK_WRITE, vector_area, vector_size, alignment, container_lock);
		ASSERT_EQ(S_OK_CREATED, r);
	}

	ccpm::cca mr(
		ccpm::region_vector_t(
			ccpm::region_vector_t::value_type(
				common::make_byte_span(heap_area, heap_size)
			)
		)
	);
	auto ccv = new (vector_area) cc_vector(mr);

	std::vector<int> original{3, 4, 5};

	std::copy(original.begin(), original.end(), eastl::back_inserter(*ccv->container));

	ASSERT_EQ(3, ccv->container->size());
	ASSERT_TRUE(std::equal(ccv->container->begin(), ccv->container->end(), original.begin()));
	/* now 3 4 5 */
	ccv->commit();
	ccv->container->push_back(7);
	ASSERT_EQ(4, ccv->container->size());
	ASSERT_TRUE(std::equal(ccv->container->begin(), ccv->container->end(), std::vector<int>{3,4,5,7}.begin()));
	/* 3 4 5 7 */
	{
		auto it = ccv->container->begin(); // -> 3
		++it; // -> 4
		auto jt = ccv->container->erase(it); // -> 5
		ASSERT_EQ(5, *jt);
		ASSERT_EQ(3, ccv->container->size());
		ASSERT_TRUE(std::equal(ccv->container->begin(), ccv->container->end(), std::vector<int>{3,5,7}.begin()));
		/* 3 5 7 */
		++jt;
		ccv->container->insert(jt, 6);
	}
	ASSERT_EQ(4, ccv->container->size());
	ASSERT_TRUE(std::equal(ccv->container->begin(), ccv->container->end(), std::vector<int>{3,5,6,7}.begin()));
	/* 3 5 6 7 */
	ccv->container->insert(ccv->container->begin(), 2);
	/* 2 3 5 6 7 */
	/* Is the container as expected? */
	{
		std::vector<int> mod_expected{2,3,5,6,7};
		ASSERT_EQ(mod_expected.size(), ccv->container->size());
		ASSERT_TRUE(std::equal(ccv->container->begin(), ccv->container->end(), mod_expected.begin()));
	}

	ccv->rollback();

	{
		/* Is the rolled back container as expected? */
		ASSERT_EQ(original.size(), ccv->container->size());
		bool is_same = std::equal(ccv->container->begin(), ccv->container->end(), original.begin());

		if ( ! is_same )
		{
			for ( auto it = ccv->container->begin(); it != ccv->container->end(); ++it )
			{
				auto i = *it;
				std::cerr << "value: " << i << "\n";
			}
		}

		ASSERT_TRUE(is_same);
	}

	/* Reverse the elements, to exercise backward iterator (probably). */
	ccv->commit();

/* can use std::reverse only if EASTL uses standard iterator categories */
#if EASTL_STD_ITERATOR_CATEGORY_ENABLED
	std::reverse(ccv->container->begin(), ccv->container->end());
	{
		/* Is the reversal also as expected? */
		ASSERT_EQ(original.size(), ccv->container->size());
		ASSERT_TRUE(std::equal(ccv->container->rbegin(), ccv->container->rend(), original.begin()));
	}
#endif

	/* Undo the reversal */
	ccv->rollback();

	{
		/* Is the rolled back container as expected? */
		ASSERT_EQ(original.size(), ccv->container->size());
		auto is_same = std::equal(ccv->container->begin(), ccv->container->end(), original.begin());
		if ( ! is_same )
		{
			for ( auto it = ccv->container->begin(); it != ccv->container->end(); ++it )
			{
				auto i = *it;
				std::cerr << "value: " << i << "\n";
			}
		}
		ASSERT_TRUE(is_same);
	}

	/* placement delete */
	ccv->~cc_vector();

	{
		auto r = kvstore->unlock(pool, container_lock);
		ASSERT_EQ(S_OK, r);
	}

	{
		auto r = kvstore->unlock(pool, heap_lock);
		ASSERT_EQ(S_OK, r);
	}
	close_pool(kvstore.get(), pool);
}

TEST_F(Log_test, CCList)
{
	auto kvstore = std::unique_ptr<component::IKVStore>(instantiate());
	ASSERT_TRUE(kvstore.get());
	auto pool = create_pool(kvstore.get(), pool_name(), MiB(10));
	ASSERT_LT(0, int64_t(pool));

	void *heap_area;
	std::size_t heap_size = MiB(5);
	component::IKVStore::key_t heap_lock{};
	{
    size_t alignment = 0;
		/* An odd "insert or locate" interface. */
		auto r = kvstore->lock(pool, "heap", component::IKVStore::STORE_LOCK_WRITE, heap_area, heap_size, alignment, heap_lock);
		ASSERT_EQ(S_OK_CREATED, r);
	}

	void *list_area;

	using cc_list = ccpm::container_cc<eastl::list<logged_int, ccpm::allocator_tl>>;

	std::size_t list_size = sizeof(cc_list);
	component::IKVStore::key_t list_lock{};
	{
    size_t alignment = 0;
		/* An odd "insert or locate" interface. */
		auto r = kvstore->lock(pool, "list", component::IKVStore::STORE_LOCK_WRITE, list_area, list_size, alignment, list_lock);
		ASSERT_EQ(S_OK_CREATED, r);
	}

	ccpm::cca mr(
		ccpm::region_vector_t(
			ccpm::region_vector_t::value_type(
				common::make_byte_span(heap_area, heap_size)
			)
		)
	);
	auto ccl = new (list_area) cc_list(mr);

	std::vector<int> original{3, 4, 5};
	std::copy(original.begin(), original.end(), eastl::back_inserter(*ccl->container));

	ASSERT_EQ(3, ccl->container->size());
	ASSERT_TRUE(std::equal(ccl->container->begin(), ccl->container->end(), original.begin()));
	/* now 3 4 5 */
	ccl->commit();
	ccl->container->push_back(7);
	ASSERT_EQ(4, ccl->container->size());
	ASSERT_TRUE(std::equal(ccl->container->begin(), ccl->container->end(), std::vector<int>{3,4,5,7}.begin()));
	/* 3 4 5 7 */
	{
		auto it = ccl->container->begin(); // -> 3
		++it; // -> 4
		auto jt = ccl->container->erase(it); // -> 5
		ASSERT_EQ(5, *jt);
		ASSERT_EQ(3, ccl->container->size());
		ASSERT_TRUE(std::equal(ccl->container->begin(), ccl->container->end(), std::vector<int>{3,5,7}.begin()));
		/* 3 5 7 */
		++jt;
		ccl->container->insert(jt, 6);
	}
	ASSERT_EQ(4, ccl->container->size());
	ASSERT_TRUE(std::equal(ccl->container->begin(), ccl->container->end(), std::vector<int>{3,5,6,7}.begin()));
	/* 3 5 6 7 */
	ccl->container->push_front(2);
	/* 2 3 5 6 7 */
	/* Is the container as expected? */
	{
		std::vector<int> mod_expected{2,3,5,6,7};
		ASSERT_EQ(mod_expected.size(), ccl->container->size());
		ASSERT_TRUE(std::equal(ccl->container->begin(), ccl->container->end(), mod_expected.begin()));
	}

	ccl->rollback();

	{
		/* Is the rolled back container as expected? */
		ASSERT_EQ(original.size(), ccl->container->size());
		ASSERT_TRUE(std::equal(ccl->container->begin(), ccl->container->end(), original.begin()));
	}

	/* Reverse the elements, to exercise backward iterator (probably). */
	ccl->commit();

	/* As reversal is performed by an algorithm (not by a container method),
	 * logging does not work unless the container elements (not just the container)
	 * are tracked.
	 */

/* can use std::reverse only of EASTL uses standard iterator categories */
#if EASTL_STD_ITERATOR_CATEGORY_ENABLED
	std::reverse(ccl->container->begin(), ccl->container->end());
	{
		/* Is the reversal also as expected? */
		ASSERT_EQ(original.size(), ccl->container->size());
		ASSERT_TRUE(std::equal(ccl->container->rbegin(), ccl->container->rend(), original.begin()));
	}
#endif

	/* Undo the reversal */
	ccl->rollback();

	{
		/* Is the rolled back container as expected? */
		ASSERT_EQ(original.size(), ccl->container->size());
		bool is_same = std::equal(ccl->container->begin(), ccl->container->end(), original.begin());
		if ( ! is_same )
		{
			for ( auto it = ccl->container->begin(); it != ccl->container->end(); ++it )
			{
				auto i = *it;
				std::cerr << "value: " << i << "\n";
			}
		}
		ASSERT_TRUE(is_same);
	}

	/* placement delete */
	ccl->~cc_list();

	{
		auto r = kvstore->unlock(pool, list_lock);
		ASSERT_EQ(S_OK, r);
	}

	{
		auto r = kvstore->unlock(pool, heap_lock);
		ASSERT_EQ(S_OK, r);
	}
	close_pool(kvstore.get(), pool);
}

TEST_F(Log_test, CCListOfPointer)
{
	auto kvstore = std::unique_ptr<component::IKVStore>(instantiate());
	ASSERT_TRUE(kvstore.get());
	auto pool = create_pool(kvstore.get(), pool_name(), MiB(10));
	ASSERT_LT(0, int64_t(pool));

	void *heap_area;
	std::size_t heap_size = MiB(5);
	component::IKVStore::key_t heap_lock{};
	{
    size_t alignment = 0;
		/* An odd "insert or locate" interface. */
		auto r = kvstore->lock(pool, "heap", component::IKVStore::STORE_LOCK_WRITE, heap_area, heap_size, alignment, heap_lock);
		ASSERT_EQ(S_OK_CREATED, r);
	}

	void *list_area;

	using cc_list = ccpm::container_cc<eastl::list<logged_ptr_to_int, ccpm::allocator_tl>>;

	std::size_t list_size = sizeof(cc_list);
	component::IKVStore::key_t list_lock{};
	{
    size_t alignment = 0;
		/* An odd "insert or locate" interface. */
		auto r = kvstore->lock(pool, "list", component::IKVStore::STORE_LOCK_WRITE, list_area, list_size, alignment, list_lock);
		ASSERT_EQ(S_OK_CREATED, r);
	}

	ccpm::cca mr(
		ccpm::region_vector_t(
			ccpm::region_vector_t::value_type(
				common::make_byte_span(heap_area, heap_size)
			)
		)
	);
	auto ccl = new (list_area) cc_list(mr);

	int *p2 = new int(2);
	int *p3 = new int(3);
	int *p4 = new int(4);
	int *p5 = new int(5);
	int *p6 = new int(6);
	int *p7 = new int(7);
	std::vector<int *> original{p3, p4, p5};
	std::copy(original.begin(), original.end(), eastl::back_inserter(*ccl->container));

	ASSERT_EQ(3, ccl->container->size());
	ASSERT_TRUE(std::equal(ccl->container->begin(), ccl->container->end(), original.begin()));
	/* now 3 4 5 */
	ccl->commit();
	ccl->container->push_back(p7);
	ASSERT_EQ(4, ccl->container->size());
	ASSERT_TRUE(std::equal(ccl->container->begin(), ccl->container->end(), std::vector<int *>{p3,p4,p5,p7}.begin()));
	/* 3 4 5 7 */
	{
		auto it = ccl->container->begin(); // -> 3
		++it; // -> 4
		auto jt = ccl->container->erase(it); // -> 5
		ASSERT_EQ(p5, *jt);
		ASSERT_EQ(3, ccl->container->size());
		ASSERT_TRUE(std::equal(ccl->container->begin(), ccl->container->end(), std::vector<int *>{p3,p5,p7}.begin()));
		/* 3 5 7 */
		++jt;
		ccl->container->insert(jt, p6);
	}
	ASSERT_EQ(4, ccl->container->size());
	ASSERT_TRUE(std::equal(ccl->container->begin(), ccl->container->end(), std::vector<int *>{p3,p5,p6,p7}.begin()));
	/* 3 5 6 7 */
	ccl->container->push_front(p2);
	/* 2 3 5 6 7 */
	/* Is the container as expected? */
	{
		std::vector<int *> mod_expected{p2,p3,p5,p6,p7};
		ASSERT_EQ(mod_expected.size(), ccl->container->size());
		ASSERT_TRUE(std::equal(ccl->container->begin(), ccl->container->end(), mod_expected.begin()));
	}

	ccl->rollback();

	{
		/* Is the rolled back container as expected? */
		ASSERT_EQ(original.size(), ccl->container->size());
		ASSERT_TRUE(std::equal(ccl->container->begin(), ccl->container->end(), original.begin()));
	}

	/* Reverse the elements, to exercise backward iterator (probably). */
	ccl->commit();

	/* As reversal is performed by an algorithm (not by a container method),
	 * logging does not work unless the container elements (not just the container)
	 * are tracked.
	 */

/* can use std::reverse only of EASTL uses standard iterator categories */
#if EASTL_STD_ITERATOR_CATEGORY_ENABLED
	std::reverse(ccl->container->begin(), ccl->container->end());
	{
		/* Is the reversal also as expected? */
		ASSERT_EQ(original.size(), ccl->container->size());
		ASSERT_TRUE(std::equal(ccl->container->rbegin(), ccl->container->rend(), original.begin()));
	}
#endif

	/* Undo the reversal */
	ccl->rollback();

	{
		/* Is the rolled back container as expected? */
		ASSERT_EQ(original.size(), ccl->container->size());
		bool is_same = std::equal(ccl->container->begin(), ccl->container->end(), original.begin());
		if ( ! is_same )
		{
			for ( auto it = ccl->container->begin(); it != ccl->container->end(); ++it )
			{
				auto i = *it;
				std::cerr << "value: " << i << "\n";
			}
		}
		ASSERT_TRUE(is_same);
	}

	/* placement delete */
	ccl->~cc_list();

	{
		auto r = kvstore->unlock(pool, list_lock);
		ASSERT_EQ(S_OK, r);
	}

	{
		auto r = kvstore->unlock(pool, heap_lock);
		ASSERT_EQ(S_OK, r);
	}
	close_pool(kvstore.get(), pool);
}

TEST_F(Log_test, CCListOfSharedPointer)
{
	auto kvstore = std::unique_ptr<component::IKVStore>(instantiate());
	ASSERT_TRUE(kvstore.get());
	auto pool = create_pool(kvstore.get(), pool_name(), MiB(10));
	ASSERT_LT(0, int64_t(pool));

	void *heap_area;
	std::size_t heap_size = MiB(5);
	component::IKVStore::key_t heap_lock{};
	{
    size_t alignment = 0;
		/* An odd "insert or locate" interface. */
		auto r = kvstore->lock(pool, "heap", component::IKVStore::STORE_LOCK_WRITE, heap_area, heap_size, alignment, heap_lock);
		ASSERT_EQ(S_OK_CREATED, r);
	}

	void *list_area;

	using cc_list = ccpm::container_cc<eastl::list<logged_shared_ptr_to_int, ccpm::allocator_tl>>;

	std::size_t list_size = sizeof(cc_list);
	component::IKVStore::key_t list_lock{};
	{
    size_t alignment = 0;
		/* An odd "insert or locate" interface. */
		auto r = kvstore->lock(pool, "list", component::IKVStore::STORE_LOCK_WRITE, list_area, list_size, alignment, list_lock);
		ASSERT_EQ(S_OK_CREATED, r);
	}

	ccpm::cca mr(
		ccpm::region_vector_t(
			ccpm::region_vector_t::value_type(
				common::make_byte_span(heap_area, heap_size)
			)
		)
	);
	auto ccl = new (list_area) cc_list(mr);

	auto p2 = std::make_shared<int>(2);
	auto p3 = std::make_shared<int>(3);
	auto p4 = std::make_shared<int>(4);
	auto p5 = std::make_shared<int>(5);
	auto p6 = std::make_shared<int>(6);
	auto p7 = std::make_shared<int>(7);
	std::vector<std::shared_ptr<int>> original{p3, p4, p5};
	std::copy(original.begin(), original.end(), eastl::back_inserter(*ccl->container));

	ASSERT_EQ(3, ccl->container->size());
	ASSERT_TRUE(std::equal(ccl->container->begin(), ccl->container->end(), original.begin()));
	/* now 3 4 5 */
	ccl->commit();
	ccl->container->push_back(p7);
	ASSERT_EQ(4, ccl->container->size());
	ASSERT_TRUE(std::equal(ccl->container->begin(), ccl->container->end(), std::vector<std::shared_ptr<int>>{p3,p4,p5,p7}.begin()));
	/* 3 4 5 7 */
	{
		auto it = ccl->container->begin(); // -> 3
		++it; // -> 4
		auto jt = ccl->container->erase(it); // -> 5
		ASSERT_EQ(p5, *jt);
		ASSERT_EQ(3, ccl->container->size());
		ASSERT_TRUE(std::equal(ccl->container->begin(), ccl->container->end(), std::vector<std::shared_ptr<int>>{p3,p5,p7}.begin()));
		/* 3 5 7 */
		++jt;
		ccl->container->insert(jt, p6);
	}
	ASSERT_EQ(4, ccl->container->size());
	ASSERT_TRUE(std::equal(ccl->container->begin(), ccl->container->end(), std::vector<std::shared_ptr<int>>{p3,p5,p6,p7}.begin()));
	/* 3 5 6 7 */
	ccl->container->push_front(p2);
	/* 2 3 5 6 7 */
	/* Is the container as expected? */
	{
		std::vector<std::shared_ptr<int>> mod_expected{p2,p3,p5,p6,p7};
		ASSERT_EQ(mod_expected.size(), ccl->container->size());
		ASSERT_TRUE(std::equal(ccl->container->begin(), ccl->container->end(), mod_expected.begin()));
	}

	ccl->rollback();

	{
		/* Is the rolled back container as expected? */
		ASSERT_EQ(original.size(), ccl->container->size());
		ASSERT_TRUE(std::equal(ccl->container->begin(), ccl->container->end(), original.begin()));
	}

	/* Reverse the elements, to exercise backward iterator (probably). */
	ccl->commit();

	/* As reversal is performed by an algorithm (not by a container method),
	 * logging does not work unless the container elements (not just the container)
	 * are tracked.
	 */

/* can use std::reverse only of EASTL uses standard iterator categories */
#if EASTL_STD_ITERATOR_CATEGORY_ENABLED
	std::reverse(ccl->container->begin(), ccl->container->end());
	{
		/* Is the reversal also as expected? */
		ASSERT_EQ(original.size(), ccl->container->size());
		ASSERT_TRUE(std::equal(ccl->container->rbegin(), ccl->container->rend(), original.begin()));
	}
#endif

	/* Undo the reversal */
	ccl->rollback();

	{
		/* Is the rolled back container as expected? */
		ASSERT_EQ(original.size(), ccl->container->size());
		bool is_same = std::equal(ccl->container->begin(), ccl->container->end(), original.begin());
		if ( ! is_same )
		{
			for ( auto it = ccl->container->begin(); it != ccl->container->end(); ++it )
			{
				auto i = *it;
				std::cerr << "value: " << i << "\n";
			}
		}
		ASSERT_TRUE(is_same);
	}

	/* placement delete */
	ccl->~cc_list();

	{
		auto r = kvstore->unlock(pool, list_lock);
		ASSERT_EQ(S_OK, r);
	}

	{
		auto r = kvstore->unlock(pool, heap_lock);
		ASSERT_EQ(S_OK, r);
	}
	close_pool(kvstore.get(), pool);
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	auto r = RUN_ALL_TESTS();

	return r;
}
