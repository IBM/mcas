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
#include "timer.h"

#include <gtest/gtest.h>
#include <common/profiler.h>
#include <common/utils.h>
#include <api/components.h>
/* note: we do not include component source, only the API definition */
#include <api/kvstore_itf.h>

#include <algorithm>
#include <chrono>
#include <cstdlib> /* getenv */
#include <future>
#include <iostream>
#include <string>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>


/*
 * Performancte test:
 *
 * export PMEM_IS_PMEM_FORCE=1
 * LD_PRELOAD=/usr/lib/libprofiler.so CPUPROFILE=cpu.profile \
 *   ./src/components/hstore/unit_test/hstore-test3
 *
 * memory requirement:
 *   memory (MB)  entries   key_sz  value_sz
 *    MB(128)       67484     8        16
 *    MB(512)      459112     8        16
 *    MB(2048UL)  2028888     8        16
 *    MB(2048UL)   918925    16        32
 *    MB(4096UL) >2000000    16        32
 *
 * (this table obsoleted, or at least made harder to generate, by use of
 * multiple key and data sizes)
 */

using namespace component;

namespace {

// The fixture for testing class Foo.
class KVStore_test
  : public ::testing::Test
{

  static const std::size_t many_count_target_large;
  static const std::size_t estimated_object_count_large;
  /* Shorter test: use when PMEM_IS_PMEM_FORCE=0 */
  static const std::size_t many_count_target_small;
  /* More testing of table splits, at a performance cost */
  static constexpr std::size_t estimated_object_count_small = 1;

 protected:

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  virtual void SetUp() {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  virtual void TearDown() {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

  static constexpr std::size_t threads = 4;
  using poolv_t = std::array<component::IKVStore::pool_t, threads>;

  // Objects declared here can be used by all tests in the test case

  /* persistent memory if enabled at all, is simulated and not real */
  static bool pmem_simulated;
  static component::IKVStore * _kvstore;

  static constexpr unsigned many_key_length_short = 16;
  static constexpr unsigned many_key_length_long = 32;
  static constexpr unsigned many_value_length_short = 16;
  static constexpr unsigned many_value_length_long = 128;
  using kv_t = std::tuple<std::string, std::string>;
  using kvv_t = std::vector<std::tuple<std::string, std::string>>;
  /* data (puts in particular) are tested in 3 combinations:
   * short short (key and data both fit in the hash table space)
   * short long (key fits in hash table space, data needs a memory allocation)
   * long long (key and data both need memory allocations)
   *
   * The current limit for fit in hash table is 23 bytes, for both key and data.
   */
  static kvv_t kvv_short_short;
  static kvv_t kvv_short_long;
  static kvv_t kvv_long_long;

  static bool short_short_put;
  static bool short_long_put;
  static bool long_long_put;

  static std::size_t multi_count_actual;
  static constexpr unsigned get_expand = 2;
  static std::size_t estimated_object_count;
  static std::size_t many_count_target;
  static auto populate_many(
    char tag
    , std::size_t key_length
    , std::size_t value_length
  ) -> kvv_t;
  static long unsigned put_many(
    component::IKVStore::pool_t pool
    , const kvv_t &kvv
    , const std::string &descr
  );
  static long unsigned put_many_threaded(const kvv_t &kvv, const std::string &descr);
  static void get_many(
    component::IKVStore::pool_t pool
    , const kvv_t &kv
    , const std::string &descr
  );
  static void get_many_threaded(const kvv_t &kvv, const std::string &descr);

  std::string pool_name(std::size_t i) const
  {
    return
      "test-" + store_map::impl->name + store_map::numa_zone()
      + "-" + std::to_string(i) + ".pool";
  }
  static poolv_t pool;
  static std::size_t string_length_and_overhead(std::size_t length)
  {
    return round_up_to_pow2(length + 16);
  }
  static std::size_t round_up_to_pow2(std::size_t n)
  {
    n = (n - 1) | 1;
    while ( n & (n-1) )
    {
      n &= (n - 1);
    }
    return n << 1;
  }
};

constexpr std::size_t KVStore_test::estimated_object_count_small;
const std::size_t KVStore_test::many_count_target_large =
  std::getenv("COUNT_TARGET") ? std::stoul(std::getenv("COUNT_TARGET")) : 500000;

/* there are three test, each of which uses many_count_target objects */
const std::size_t KVStore_test::estimated_object_count_large =
  many_count_target_large * 3;

const std::size_t KVStore_test::many_count_target_small =
  std::getenv("COUNT_TARGET") ? std::stoul(std::getenv("COUNT_TARGET")) : 400;

bool KVStore_test::pmem_simulated = getenv("PMEM_IS_PMEM_FORCE");
component::IKVStore *KVStore_test::_kvstore;
KVStore_test::poolv_t KVStore_test::pool;

constexpr unsigned KVStore_test::many_key_length_short;
constexpr unsigned KVStore_test::many_key_length_long;
constexpr unsigned KVStore_test::many_value_length_short;
constexpr unsigned KVStore_test::many_value_length_long;
KVStore_test::kvv_t KVStore_test::kvv_short_short;
KVStore_test::kvv_t KVStore_test::kvv_short_long;
KVStore_test::kvv_t KVStore_test::kvv_long_long;
bool KVStore_test::short_short_put = false;
bool KVStore_test::short_long_put = false;
bool KVStore_test::long_long_put = false;

std::size_t KVStore_test::multi_count_actual = 0;
std::size_t KVStore_test::estimated_object_count =
  KVStore_test::pmem_simulated
  ? estimated_object_count_small
  : estimated_object_count_large;
std::size_t KVStore_test::many_count_target =
  KVStore_test::pmem_simulated
  ? many_count_target_small
  : many_count_target_large;

TEST_F(KVStore_test, Instantiate)
{
  /* create object instance through factory */
  auto link_library = "libcomponent-" + store_map::impl->name + ".so";
  component::IBase * comp = component::load_component(link_library,
                                                      store_map::impl->factory_id);

  ASSERT_TRUE(comp);
  auto fact =
    component::make_itf_ref(
      static_cast<IKVStore_factory *>(
        comp->query_interface(IKVStore_factory::iid())
      )
    );

  _kvstore =
    fact->create(
      0
      , {
          { +component::IKVStore_factory::k_dax_config, store_map::location }
        }
    );
}

TEST_F(KVStore_test, RemoveOldPool)
{
  if ( _kvstore )
  {
    for ( auto i = std::size_t(); i != pool.size(); ++i )
    {
      try
      {
        _kvstore->delete_pool(pool_name(i));
      }
      catch ( Exception & )
      {
      }
    }
  }
}

TEST_F(KVStore_test, CreatePools)
{
  /* time to create pools, in estimate objects / second */
  std::cerr << "many_count_target " << many_count_target << "\n";
  std::cerr << "estimated_object_count " << estimated_object_count << "\n";
  /* the three values are for small/small, samm/large, and large/large */
  std::size_t large_keys_size =
    0
    + 0
    + string_length_and_overhead(many_key_length_long);
  std::size_t large_values_size =
    0
    + string_length_and_overhead(many_value_length_long)
    + string_length_and_overhead(many_value_length_long);
  /* index requirement is
   *    estimated_object_count * size of index object / index efficiency
   *   rounded up to power of 2.
   */
  auto index_alloc = round_up_to_pow2(( estimated_object_count * 64U* 5U) / 5U);
  /* The largest segment, which consumes half the space, will be require
   * that double its size be availabe during allocation. (nupm::Region_map
   * requires size bytes aligned by alignment, where alignment == size.
   * The strategy for doing that is to ask for size+alignment bytes and
   * from that carve out size bytes properly aligned.
   */
  index_alloc = index_alloc * 3U / 2U;
  auto key_alloc = large_keys_size * many_count_target;
  auto value_alloc = large_values_size * many_count_target;
  std::cerr << "index alloc " << index_alloc << "\n";
  std::cerr << "key alloc " << key_alloc << "\n";
  std::cerr << "value alloc " << value_alloc << "\n";
  auto pool_alloc_needed = index_alloc + key_alloc + value_alloc;
  std::size_t pool_header_size = 0x4d0;
  auto pool_alloc = pool_header_size + pool_alloc_needed * 4U/ 3U;
  if ( std::getenv("POOL_ALLOCATE_FACTOR") )
  {
    double af = std::stod(getenv("POOL_ALLOCATE_FACTOR"));
    pool_alloc = std::size_t(double(pool_alloc) * af);
    std::cerr << "pool allocatin extimate: " << pool_alloc << "\n";
  }
  ASSERT_TRUE(_kvstore);
  timer t(
    [] (timer::duration_t d) {
      auto seconds = std::chrono::duration<double>(d).count();
      std::cerr << "create pool" << " "
        << double(estimated_object_count * pool.size()) / seconds
        << " estimated objects per second\n";
    }
  );
  for ( auto i = std::size_t(); i != pool.size(); ++i )
  {
#if 0
    pool[i] =
    _kvstore->create_pool(
      pool_name(i)
      , GB(1) + pool_alloc
      , 0
      , estimated_object_count
    );
#else
    pool[i] = _kvstore->create_pool(pool_name(i), GB(1), 0, 1);

    std::size_t old_size = 0;
    /* zero growth should return current size */
    {
      auto rc = _kvstore->grow_pool(pool[i], 0, old_size);
      EXPECT_EQ(S_OK, rc);
    }
    /* Not sure hwo much will be reported allocated, but it should be at least
     * half of what we asked for,
     */
    EXPECT_LT(GB(1)/2, old_size);

    if ( GB(1) < pool_alloc )
    {
      std::size_t sz = 0;
      auto rc = _kvstore->grow_pool(pool[i], pool_alloc - GB(1), sz);
      EXPECT_EQ(S_OK, rc);
      std::cerr << "grow pool" << " from " << old_size << " to " << sz << "\n";
      /* new size should be more than old size */
      EXPECT_LT(old_size, sz);
    }
#endif
    ASSERT_LT(0, int64_t(pool[i]));
  }
}

auto KVStore_test::populate_many(
  const char tag
  , std::size_t key_length
  , std::size_t value_length
) -> kvv_t
{
  std::mt19937_64 r0{};
  kvv_t kvv;
  for ( auto i = std::size_t(); i != many_count_target; ++i )
  {
    auto ukey = r0();
    std::ostringstream s;
    s << tag << std::hex << ukey;
    auto key = s.str();
    key.resize(key_length, '.');
    auto value = std::to_string(i);
    value.resize(value_length, '.');
    kvv.emplace_back(key, value);
  }
  return kvv;
}

TEST_F(KVStore_test, PopulateManySmallSmall)
{
  kvv_short_short =
    populate_many('A', many_key_length_short, many_value_length_short);
}

TEST_F(KVStore_test, PopulateManySmallLarge)
{
  kvv_short_long =
    populate_many('B', many_key_length_short, many_value_length_long);
}

TEST_F(KVStore_test, PopulateManyLargeLarge)
{
  kvv_long_long =
    populate_many('C', many_key_length_long, many_value_length_long);
}

long unsigned KVStore_test::put_many(
  component::IKVStore::pool_t pool_, const kvv_t &kvv, const std::string &descr
)
{
  long unsigned count = 0;
  {
    timer t(
      [&count, &descr] (timer::duration_t d) {
        auto seconds = std::chrono::duration<double>(d).count();
        std::cerr << descr << " " << double(count) / seconds << " per second\n";
      }
    );
    for ( auto &kv : kvv )
    {
      const auto &key = std::get<0>(kv);
      const auto &value = std::get<1>(kv);
      auto r = _kvstore->put(pool_, key, value.data(), value.length());
      if ( r == S_OK )
      {
          ++count;
      }
      else
      {
         std::cerr << __func__ << " FAIL " << key << "\n";
      }
    }
  }
  return count;
}

long unsigned KVStore_test::put_many_threaded(
  const kvv_t &kvv
  , const std::string &descr
)
{

  std::vector<std::future<long unsigned>> v;
  common::profiler pr("test4-put-" + descr + "-cpu-" + store_map::impl->name + ".profile");
  for ( auto p : pool )
  {
    v.emplace_back(std::async(std::launch::async, put_many, p, kvv, descr));
  }

  long unsigned count_actual = 0;
  for ( auto &e : v ) { count_actual += e.get(); }
  return count_actual;
}

TEST_F(KVStore_test, PutManyShortShort)
{
  ASSERT_NE(nullptr, _kvstore);
  for ( auto p : pool ) { ASSERT_LT(0, int64_t(p)); }

  auto count_actual = put_many_threaded(kvv_short_short, "short_short");

  EXPECT_GE(many_count_target*pool.size(), count_actual);
  EXPECT_LE(many_count_target*pool.size() * 99 / 100, count_actual);

  multi_count_actual += count_actual;
  short_short_put = true;
}

TEST_F(KVStore_test, PutManyShortLong)
{
  ASSERT_NE(nullptr, _kvstore);
  for ( auto p : pool ) { ASSERT_LT(0, int64_t(p)); }

  auto count_actual = put_many_threaded(kvv_short_long, "short_long");

  EXPECT_GE(many_count_target*pool.size(), count_actual);
  EXPECT_LE(many_count_target*pool.size() * 99 / 100, count_actual);

  multi_count_actual += count_actual;
  short_long_put = true;
}

TEST_F(KVStore_test, PutManyLongLong)
{
  ASSERT_NE(nullptr, _kvstore);
  for ( auto p : pool ) { ASSERT_LT(0, int64_t(p)); }

  auto count_actual = put_many_threaded(kvv_long_long, "long_long");

  EXPECT_GE(many_count_target*pool.size(), count_actual);
  EXPECT_LE(many_count_target*pool.size() * 99 / 100, count_actual);

  multi_count_actual += count_actual;
  long_long_put = true;
}

TEST_F(KVStore_test, Size1)
{
  ASSERT_NE(nullptr, _kvstore);
  for ( auto p : pool ) { ASSERT_LT(0, int64_t(p)); }

  unsigned long count = 0;
  for ( auto p : pool )
  {
    count += _kvstore->count(p);
  }

  /* count should reflect PutMany */
  EXPECT_EQ(multi_count_actual, count);
}

void KVStore_test::get_many(
  component::IKVStore::pool_t pool_
  , const kvv_t &kvv
  , const std::string &descr
)
{
  /* get is quick; run 10 for better profiling */
  {
  auto ct = get_expand * kvv.size();
    timer t(
      [&descr,ct] (timer::duration_t d) {
        auto seconds = std::chrono::duration<double>(d).count();
        std::cerr << descr << " " << ct
               << " in " << seconds << " => "
               << double(ct) / seconds << " per second\n";
      }
    );
    for ( auto i = 0; i != get_expand; ++i )
    {
      for ( auto &kv : kvv )
      {
        const auto &key = std::get<0>(kv);
        void * value = nullptr;
        size_t value_len = 0;
        auto r = _kvstore->get(pool_, key, value, value_len);
        EXPECT_EQ(S_OK, r);
        if ( S_OK == r )
        {
          EXPECT_EQ(std::get<1>(kv).size(), value_len);
        }
        _kvstore->free_memory(value);
      }
    }
  }
}

void KVStore_test::get_many_threaded(const kvv_t &kvv, const std::string &descr)
{
  std::vector<std::future<void>> v;
  common::profiler pr("test4-get-" + descr + "-cpu-" + store_map::impl->name + ".profile");
  for ( auto p : pool )
  {
    v.emplace_back(std::async(std::launch::async, get_many, p, kvv, descr));
  }
  for ( auto &e : v ) { e.get(); }
}

TEST_F(KVStore_test, GetManyShortShort)
{
  ASSERT_NE(nullptr, _kvstore);
  ASSERT_EQ(short_short_put, true);
  for ( auto p : pool ) { ASSERT_LT(0, int64_t(p)); }

  get_many_threaded(kvv_short_short, "short_short");
}

TEST_F(KVStore_test, GetManyShortLong)
{
  ASSERT_NE(nullptr, _kvstore);
  ASSERT_EQ(short_long_put, true);
  for ( auto p : pool ) { ASSERT_LT(0, int64_t(p)); }

  get_many_threaded(kvv_short_long, "short_long");
}

TEST_F(KVStore_test, GetManyLongLong)
{
  ASSERT_NE(nullptr, _kvstore);
  ASSERT_EQ(long_long_put, true);
  for ( auto p : pool ) { ASSERT_LT(0, int64_t(p)); }

  get_many_threaded(kvv_long_long, "long_long");
}

TEST_F(KVStore_test, ClosePool)
{
  timer t(
    [] (timer::duration_t d) {
      auto seconds = std::chrono::duration<double>(d).count();
      std::cerr << "close pool" << " " << double(multi_count_actual) / seconds
        << " objects per second\n";
    }
  );
  for ( auto p : pool )
  {
    if ( _kvstore && 0 < int64_t(p) )
    {
      _kvstore->close_pool(p);
    }
  }
}

TEST_F(KVStore_test, OpenPool2)
{
  common::profiler p("test4-open-pool-2-cpu-" + store_map::impl->name + ".profile");
  timer t(
    [] (timer::duration_t d) {
      auto seconds = std::chrono::duration<double>(d).count();
      std::cerr << "open pool" << " "
        << multi_count_actual << "/" << seconds << " = "
        << double(multi_count_actual) / seconds << " objects per second\n";
    }
  );
  ASSERT_TRUE(_kvstore);
  using fv = std::vector<std::future<component::IKVStore::pool_t>>;
  fv v;
  for ( auto i = std::size_t(); i != pool.size(); ++i )
  {
    v.emplace_back(
      std::async(
        std::launch::async
        , [&] (unsigned i_) { return _kvstore->open_pool(pool_name(i_)); }
        , i )
    );
  }
  std::transform(
    v.begin()
    , v.end()
    , pool.begin()
    , [] (fv::value_type &f) { auto r = f.get(); EXPECT_LT(0, r); return r; }
  );
}

TEST_F(KVStore_test, ClosePool2)
{
  for ( auto p : pool )
  {
    if ( _kvstore && 0 < int64_t(p) )
    {
      _kvstore->close_pool(p);
    }
  }
}

TEST_F(KVStore_test, DeletePool)
{
  for ( auto i = std::size_t(); i != pool.size(); ++i )
  {
    if ( _kvstore )
    {
      try
      {
        _kvstore->delete_pool(pool_name(i));
      }
      catch ( ... )
      {
      }
    }
  }
}

} // namespace

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
