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
#include "store_map.h"

#include <gtest/gtest.h>
#include <common/utils.h>
#include <api/components.h>
/* note: we do not include component source, only the API definition */
#include <api/kvstore_itf.h>

#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace component;

namespace {

// The fixture for testing class Foo.
class KVStore_test : public ::testing::Test {

  static constexpr std::size_t estimated_object_count_small = 20;

  static constexpr std::size_t many_count_target = 20;

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

  // Objects declared here can be used by all tests in the test case
  /* persistent memory if enabled at all, is simulated and not real */
  static bool pmem_simulated;
  /* persistent memory is effective (either real, indicated by no PMEM_IS_PMEM_FORCE or simulated by PMEM_IS_PMEM_FORCE 0 not 1 */
  static bool pmem_effective;
  static component::IKVStore * _kvstore;

  static const std::size_t estimated_object_count;

  static constexpr unsigned many_key_length = 8;
  static constexpr unsigned many_value_length = 16;
  static constexpr char long_value[24] = "........" "........" ".......";
  using kv_t = std::tuple<std::string, std::string>;
  static std::vector<kv_t> kvv;
  static const std::size_t many_count_target;
  static std::size_t many_count_actual;

  std::string pool_name() const
  {
    return "/mnt/pmem0/pool/0/test-" + store_map::impl->name + store_map::numa_zone() + ".pool";
  }
};

constexpr std::size_t KVStore_test::estimated_object_count_small;
constexpr std::size_t KVStore_test::estimated_object_count_large;
constexpr std::size_t KVStore_test::many_count_target_small;
constexpr std::size_t KVStore_test::many_count_target_large;
constexpr char KVStore_test::long_value[24];

bool KVStore_test::pmem_simulated = getenv("PMEM_IS_PMEM_FORCE");
bool KVStore_test::pmem_effective = ! getenv("PMEM_IS_PMEM_FORCE") || getenv("PMEM_IS_PMEM_FORCE") == std::string("0");
component::IKVStore * KVStore_test::_kvstore;

const std::size_t KVStore_test::estimated_object_count =
#if 0
  pmem_simulated ? estimated_object_count_small : estimated_object_count_large
#else
  0 /* test resize */
#endif
  ;

constexpr unsigned KVStore_test::many_key_length;
constexpr unsigned KVStore_test::many_value_length;

const std::size_t KVStore_test::many_count_target = pmem_simulated ? many_count_target_small : many_count_target_large;
std::size_t KVStore_test::many_count_actual;
std::vector<KVStore_test::kv_t> KVStore_test::kvv;

TEST_F(KVStore_test, Instantiate)
{
  std::cerr
    << "PMEM " << (pmem_simulated ? "simulated" : "not simluated")
    << ", " << (pmem_effective ? "effective" : "not effective")
    << "\n";
  /* create object instance through factory */
  /* This test only: use hstore-pe. the version compiled with simulated injection */
  auto link_library = "libcomponent-" + store_map::impl->name + "-pe.so";
  component::IBase * comp = component::load_component(link_library,
                                                      store_map::impl->factory_id);

  ASSERT_TRUE(comp);
  auto fact = component::make_itf_ref(static_cast<IKVStore_factory *>(comp->query_interface(IKVStore_factory::iid())));

  _kvstore = fact->create("owner", "name", store_map::location);

struct pool_open
{
private:
  component::IKVStore *_kvstore;
  component::IKVStore::pool_t _pool;
public:
  explicit pool_open(
    component::IKVStore *kvstore_
    , const std::string& name_
    , unsigned int flags = 0
  )
    : _kvstore(kvstore_)
    , _pool(_kvstore->open_pool(name_, flags))
  {
    if ( int64_t(_pool) < 0 )
    {
      throw std::runtime_error("Failed to open pool code " + std::to_string(-_pool));
    }
  }

  explicit pool_open(
    component::IKVStore *kvstore_
    , const std::string& name_
    , const size_t size
    , unsigned int flags = 0
    , uint64_t expected_obj_count = 0
  )
    : _kvstore(kvstore_)
    , _pool(_kvstore->create_pool(name_, size, flags, expected_obj_count))
  {}
  pool_open(const pool_open &) = delete;
  pool_open& operator=(const pool_open &) = delete;

  ~pool_open()
  {
    _kvstore->close_pool(_pool);
  }

  component::IKVStore::pool_t pool() const noexcept { return _pool; }
};

TEST_F(KVStore_test, RemoveOldPool)
{
  if ( _kvstore )
  {
    try
    {
      _kvstore->delete_pool(pool_name());
    }
    catch ( Exception & )
    {
    }
  }
}

TEST_F(KVStore_test, CreatePool)
{
  ASSERT_TRUE(_kvstore);
  pool_open p(_kvstore, pool_name(), GB(15UL), 0, estimated_object_count);
  ASSERT_LT(0, int64_t(p.pool()));
}

TEST_F(KVStore_test, PopulateMany)
{
  std::mt19937_64 r0{};
  for ( auto i = 0UL; i != many_count_target; ++i )
  {
    auto ukey = r0();
    std::ostringstream s;
    s << std::hex << ukey;
    auto key = s.str();
    key.resize(many_key_length, '.');
    auto value = std::to_string(i);
    value.resize(many_value_length, '.');
    kvv.emplace_back(key, value);
  }
}

TEST_F(KVStore_test, PutMany)
{
  /* We will try the inserts many times, as the perishable timer will abort all but the last attempt */
  bool finished = false;

  _kvstore->debug(0, 0 /* enable */, 0);
  /*
   * Crash on after every other perishable operation. Guarantees progress, albeit slow.
   */
  unsigned perishable_count = 1;
  for (
    ;
    ; ! finished
    )
  {
    unsigned extant_count = 0;
    unsigned fail_count = 0;
    unsigned succeed_count = 0;
    _kvstore->debug(0, 1 /* reset */, perishable_count);
    _kvstore->debug(0, 0 /* enable */, true);
    try
    {
      pool_open p(_kvstore, pool_name());

      std::mt19937_64 r0{};

      for ( auto &kv : kvv )
      {
        const auto &key = std::get<0>(kv);
        const auto &value = std::get<1>(kv);
        void * old_value = nullptr;
        size_t old_value_len = 0;
        if ( S_OK == _kvstore->get(p.pool(), key, old_value, old_value_len) )
        {
          _kvstore->free_memory(old_value);
          ++extant_count;
        }
        else
        {
          auto r = _kvstore->put(p.pool(), key, value.c_str(), value.length());
          EXPECT_EQ(S_OK, r);
          if ( r == S_OK )
          {
            ++succeed_count;
          }
          else
          {
            ++fail_count;
          }
        }
      }
      EXPECT_EQ(many_count_target, extant_count + succeed_count + fail_count);
      many_count_actual = extant_count + succeed_count;
      finished = true;
      /* Done with forcing crashes */
      _kvstore->debug(0, 0 /* enable */, false);
      std::cerr << __func__ << " Final put pass " << perishable_count << " exists " << extant_count << " inserts " << succeed_count << " total " << many_count_actual << "\n";
    }
    catch ( const std::runtime_error &e )
    {
      if ( e.what() != std::string("perishable timer expired") ) { throw; }
      std::cerr << __func__ << " Perishable pass " << perishable_count << " exists " << extant_count << " inserts " << succeed_count << " total " << many_count_actual << "\n";
    }
  }
}

TEST_F(KVStore_test, GetMany)
{
  ASSERT_TRUE(_kvstore);
  if ( pmem_effective )
  {
    pool_open p(_kvstore, pool_name());
    ASSERT_LT(0, int64_t(p.pool()));
    auto count = _kvstore->count(p.pool());
    {
      /* count should be close to PutMany many_count_actual; duplicate keys are the difference */
      EXPECT_LE(many_count_actual * 99 / 100, count);
    }
    {
      std::size_t mismatch_count = 0;
      for ( auto &kv : kvv )
      {
        const auto &key = std::get<0>(kv);
        const auto &ev = std::get<1>(kv);
        void * value = nullptr;
        size_t value_len = 0;
        auto r = _kvstore->get(p.pool(), key, value, value_len);
        EXPECT_EQ(S_OK, r);
        if ( S_OK == r )
        {
          EXPECT_EQ(ev.size(), value_len);
          mismatch_count += ( ev.size() != value_len || 0 != memcmp(ev.data(), value, ev.size()) );
          _kvstore->free_memory(value);
        }
      }
      /* We do not know exactly now many mismatches (caused by duplicates) to expect,
       * because "extant_count" counts both extant items due to duplicate keys in the
       * population arrays and extant items due to restarts.
       * But it should be a small fraction of the total number of keys
       */
      EXPECT_GT(many_count_target / 100, mismatch_count);
    }
  }
}

TEST_F(KVStore_test, UpdateMany)
{
  /* We will try the inserts many times, as the perishable timer will abort all but the last attempt */
  bool finished = false;

  _kvstore->debug(0, 0 /* enable */, 0);
  /*
   * We would like to generate "crashes" with some reasonable frequency,
   * but not at every store. (Every store would be too slow, at least
   * when using mmap to simulate persistent store). We use a Fibonacci
   * series to produce crashes at decreasingly frequent intervals.
   *
   * p0 and p1 produce a Fibonacci series in perishable_count
   */
  unsigned perishable_count = 1;
  for ( ; ! finished ; )
  {
    unsigned fail_count = 0;
    unsigned succeed_count = 0;
    _kvstore->debug(0, 1 /* reset */, perishable_count);
    _kvstore->debug(0, 0 /* enable */, true);
    try
    {
      pool_open p(_kvstore, pool_name());

      std::mt19937_64 r0{};

      for ( auto &kv : kvv )
      {
        /* Long enough to force a move to out-of-line storage */
        const auto &key = std::get<0>(kv);
        const auto &value = std::get<1>(kv);
        const auto update_value = value + ((key[0] & 1) ? "X" : "") + ((key[0] & 2) ? long_value : "");
        {
          auto r = _kvstore->put(p.pool(), key, update_value.c_str(), update_value.length());
          EXPECT_EQ(S_OK, r);
          if ( r == S_OK )
          {
            ++succeed_count;
          }
          else
          {
            ++fail_count;
          }
        }
      }
      EXPECT_EQ(many_count_target, succeed_count + fail_count);
      many_count_actual = succeed_count;
      finished = true;
      /* Done with forcing crashes */
      _kvstore->debug(0, 0 /* enable */, false);
      std::cerr << __func__ << " Final put pass " << perishable_count << " inserts " << succeed_count << " total " << many_count_actual << "\n";
    }
    catch ( const std::runtime_error &e )
    {
      if ( e.what() != std::string("perishable timer expired") ) { throw; }
      std::cerr << __func__ << " Perishable pass " << perishable_count << " inserts " << succeed_count << " total " << many_count_actual << "\n";
    }
  }
}

TEST_F(KVStore_test, GetManyUpdates)
{
  _kvstore->debug(0, 0 /* enable */, 0);
  ASSERT_TRUE(_kvstore);
  if ( pmem_effective )
  {
    pool_open p(_kvstore, pool_name());
    ASSERT_LT(0, int64_t(p.pool()));
    auto count = _kvstore->count(p.pool());
    {
      /* count should be close to PutMany many_count_actual; duplicate keys are the difference */
      EXPECT_LE(many_count_actual * 99 / 100, double(count));
    }
    {
      std::size_t mismatch_count = 0;
      for ( auto &kv : kvv )
      {
        const auto &key = std::get<0>(kv);
        const auto &ev = std::get<1>(kv);
        const auto update_ev = ev + ((key[0] & 1) ? "X" : "") + ((key[0] & 2) ? long_value : "");
        void * value = nullptr;
        size_t value_len = 0;
        auto r = _kvstore->get(p.pool(), key, value, value_len);
        EXPECT_EQ(S_OK, r);
        if ( S_OK == r )
        {
          if ( update_ev.size() != value_len )
          {
            std::cerr << "Length mismatch, key length " << key.size() << " key " << key << " expected value " << update_ev << " actual value " << std::string(static_cast<char *>(value), value_len) << " key[0] " << key[0] << " selector " << (key[0] & 1) << "\n";
          }
          EXPECT_EQ(update_ev.size(), value_len);
          mismatch_count += ( update_ev.size() != value_len || 0 != memcmp(update_ev.data(), value, update_ev.size()) );
          _kvstore->free_memory(value);
        }
      }
      /* We do not know exactly now many mismatches (caused by duplicates) to expcect,
       * because "extant_count" counts both extant items due to duplicate keys in the
       * population arrays and extant items due to restarts.
       * But it should be a small fraction of the total number of keys
       */
      EXPECT_GT(many_count_target / 100, mismatch_count);
    }
  }
}

TEST_F(KVStore_test, EraseMany)
{
  ASSERT_TRUE(_kvstore);
  bool finished = false;

  unsigned perishable_count = 1;
  for ( ; ! finished )
  {
    auto erase_count = 0;
    try
    {
      pool_open p(_kvstore, pool_name());
      ASSERT_LT(0, int64_t(p.pool()));
      {
        for ( auto &kv : kvv )
        {
          const auto &key = std::get<0>(kv);
          auto r = _kvstore->erase(p.pool(), key);
          if ( r == S_OK )
          {
            ++erase_count;
          }
        }
        auto count = _kvstore->count(p.pool());
        EXPECT_EQ(0U, count);
        finished = true;
      }
    }
    catch ( const std::runtime_error &e )
    {
      if ( e.what() != std::string("perishable timer expired") ) { throw; }
      std::cerr << __func__ << " Perishable pass " << perishable_count << " erasures " << erase_count << " total " << many_count_actual << "\n";
    }
  }
}

TEST_F(KVStore_test, DeletePool)
{
  if ( pmem_effective )
  {
    _kvstore->delete_pool(pool_name());
  }
}

} // namespace

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
