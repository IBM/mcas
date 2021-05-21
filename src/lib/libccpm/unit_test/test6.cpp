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
#include <common/profiler.h>
#include <common/utils.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Weffc++"
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
              0 /* debug mask */
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
			std::cerr << "Pool not created\n";
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

TEST_F(Log_test, CCVectorOfPointer)
{
/* operations to test
 *   emplace()
 *   insert() - 5 versions
 *   erase()
 *   clear()
 *   assign() - 3 versions
 */
	std::size_t heap_size = MiB(150);
	auto kvstore = std::unique_ptr<component::IKVStore>(instantiate());
	ASSERT_TRUE(kvstore.get());
	auto pool = create_pool(kvstore.get(), pool_name(), heap_size * 3);
	ASSERT_LT(0, int64_t(pool));

	void *heap_area;
	component::IKVStore::key_t heap_lock{};
	{
		/* An odd "insert or locate" interface. */
    size_t alignment = 0;
		auto r = kvstore->lock(pool, "heap", component::IKVStore::STORE_LOCK_WRITE,
                           heap_area, heap_size, alignment, heap_lock);
		ASSERT_EQ(S_OK_CREATED, r);
	}

	void *vector_area;

	using vt = int *;
	using logged_vt = ccpm::value_tracked<vt, ccpm::tracker_log>;

	using cc_vector = ccpm::container_cc<eastl::vector<logged_vt, ccpm::allocator_tl>>;

	std::size_t vector_size = sizeof(cc_vector);
	component::IKVStore::key_t container_lock{};
	{
    size_t alignment = 0;
		/* An odd "insert or locate" interface. */
		auto r = kvstore->lock(pool, "vector", component::IKVStore::STORE_LOCK_WRITE,
                           vector_area, vector_size, alignment, container_lock);
		ASSERT_EQ(S_OK_CREATED, r);
	}

	common::profiler pr("test6-vp-cpu-" + store_map::impl->name + ".profile");
	{
		ccpm::cca mr(
          ccpm::region_vector_t(
            ccpm::region_vector_t::value_type(
              common::make_byte_span(heap_area, heap_size)
            )
          )
        );
		auto ccv = new (vector_area) cc_vector(mr);

		for ( int i = 0; i != 1000000; ++i )
		{
			ccv->container->push_back(new int(i));
		}

		/* placement delete */
		ccv->~cc_vector();
	}

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

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	auto r = RUN_ALL_TESTS();

	return r;
}
