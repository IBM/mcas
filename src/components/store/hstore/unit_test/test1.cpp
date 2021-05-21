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
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <api/components.h>
/* note: we do not include component source, only the API definition */
#include <api/kvstore_itf.h>
#include <common/byte_span.h>
#include <common/str_utils.h> /* random_string */
#include <common/utils.h> /* MiB, GiB */
#include <nupm/region_descriptor.h>

#include <algorithm>
#include <random>
#include <set>
#include <sstream>
#include <string>

using namespace component;

namespace {

struct memo_lock
{
private:
  component::IKVStore * _kvstore;
  component::IKVStore::pool_t *_pool;
public:
  IKVStore::key_t k;
  memo_lock(
    component::IKVStore *kvstore_
    , component::IKVStore::pool_t *pool_
  )
    : _kvstore(kvstore_)
    , _pool(pool_)
    , k()
  {}
  memo_lock(const memo_lock &) = delete;
  memo_lock &operator=(const memo_lock &) = delete;
  ~memo_lock()
  {
    if ( IKVStore::KEY_NONE != k )
    {
      auto r = _kvstore->unlock(*_pool, k, component::IKVStore::UNLOCK_FLAGS_FLUSH);
      EXPECT_EQ(S_OK, r);
    }
  }
};

struct shared_lock
  : private memo_lock
{
  using memo_lock::k;
  shared_lock(
    component::IKVStore *kvstore_
    , component::IKVStore::pool_t *pool_
  )
    : memo_lock(kvstore_, pool_)
  {}
};

struct exclusive_lock
  : private memo_lock
{
  using memo_lock::k;
  exclusive_lock(
    component::IKVStore *kvstore_
    , component::IKVStore::pool_t *pool_
  )
    : memo_lock(kvstore_, pool_)
  {}
};

// The fixture for testing class Foo.
class KVStore_test : public ::testing::Test {

  static constexpr std::size_t many_count_target_large = 2000000;
  /* Shorter test: use when PMEM_IS_PMEM_FORCE=0 */
  static constexpr std::size_t many_count_target_small = 400;

  static constexpr std::size_t estimated_object_count_large = many_count_target_large;
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

  // Objects declared here can be used by all tests in the test case
  /* persistent memory if enabled at all, is simulated and not real */
  static bool pmem_simulated;
  /* persistent memory is effective (either real, indicated by no PMEM_IS_PMEM_FORCE or simulated by PMEM_IS_PMEM_FORCE 0 not 1 */
  static bool pmem_effective;
  static component::IKVStore * _kvstore;
  static component::IKVStore::pool_t pool;

  static const std::size_t estimated_object_count;

  static std::string single_key;
  static std::string missing_key;
  static std::string single_value;
  static std::size_t single_value_size;
  static std::string single_value_updated_same_size;
  static std::string single_value_updated_different_size;
  static std::string single_value_updated3;
  static std::size_t single_count;

  static constexpr unsigned many_key_length = 8;
  static constexpr unsigned many_value_length = 16;
  using kv_t = std::tuple<std::string, std::string>;
  static std::vector<kv_t> kvv;
  static const std::size_t many_count_target;
  static std::size_t many_count_actual;
  static constexpr unsigned get_expand = 2;

  /* NOTE: ignoring the remote possibility of a random number collision in the first lock_count entries */
  static const std::size_t lock_count;

  static std::size_t extant_count; /* Number of PutMany keys not placed because they already existed */

  std::string pool_name() const
  {
    return "pool/" + store_map::numa_zone() + "/test-" + store_map::impl->name;
  }
  static std::set<std::string> many_insert_set;
  static std::set<std::string> many_erase_okay;
  static std::set<std::string> many_erase_fail;
  static std::string debug_level()
  {
    return std::getenv("DEBUG") ? std::getenv("DEBUG") : "0";
  }
};


constexpr std::size_t KVStore_test::estimated_object_count_small;
constexpr std::size_t KVStore_test::estimated_object_count_large;
constexpr std::size_t KVStore_test::many_count_target_small;
constexpr std::size_t KVStore_test::many_count_target_large;

bool KVStore_test::pmem_simulated = getenv("PMEM_IS_PMEM_FORCE");
bool KVStore_test::pmem_effective = ! getenv("PMEM_IS_PMEM_FORCE") || getenv("PMEM_IS_PMEM_FORCE") == std::string("0");
component::IKVStore * KVStore_test::_kvstore;
component::IKVStore::pool_t KVStore_test::pool;

const std::size_t KVStore_test::estimated_object_count = pmem_simulated ? estimated_object_count_small : estimated_object_count_large;

/* Keys 23-byte or fewer are stored inline. Provide one longer to force allocation */
std::string KVStore_test::single_key = "MySingleKeyLongEnoughToForceAllocation";
std::string KVStore_test::missing_key = "KeyNeverInserted";
std::string KVStore_test::single_value         = "Hello world!";
std::size_t KVStore_test::single_value_size    = MiB(8);
std::string KVStore_test::single_value_updated_same_size = "Jello world!";
std::string KVStore_test::single_value_updated_different_size = "Hello world!";
std::string KVStore_test::single_value_updated3 = "WeXYZ world!";
std::size_t KVStore_test::single_count = 1U;

constexpr unsigned KVStore_test::many_key_length;
constexpr unsigned KVStore_test::many_value_length;
const std::size_t KVStore_test::many_count_target = pmem_simulated ? many_count_target_small : many_count_target_large;
std::size_t KVStore_test::many_count_actual;
std::size_t KVStore_test::extant_count = 0;
std::vector<KVStore_test::kv_t> KVStore_test::kvv;

const std::size_t KVStore_test::lock_count = std::min(many_count_target, std::size_t(60));
std::set<std::string> KVStore_test::many_insert_set;
std::set<std::string> KVStore_test::many_erase_okay;
std::set<std::string> KVStore_test::many_erase_fail;

TEST_F(KVStore_test, Instantiate)
{
  /* create object instance through factory */
  auto link_library = "libcomponent-" + store_map::impl->name + ".so";
  component::IBase * comp = component::load_component(link_library,
                                                      store_map::impl->factory_id);

  ASSERT_TRUE(comp);
  auto fact = component::make_itf_ref(static_cast<IKVStore_factory *>(comp->query_interface(IKVStore_factory::iid())));
  /* numa node 0 */
  _kvstore =
     fact->create(
      0
      , {
          { +component::IKVStore_factory::k_name, "numa0"}
          , { +component::IKVStore_factory::k_dax_config, store_map::location }
          , { +component::IKVStore_factory::k_debug, debug_level() }
        }
    );
}

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
  /* count of elements multiplied
   *  - by 64 for bucket size,
   *  - by 3U to account for 40% table density at expansion (typical),
   *  - by 2U to account for worst-case due to doubling strategy for increasing bucket array size
   * requires size multiplied
   *  - by 8U to account for current AVL_LB allocator alignment requirements
   */
  pool = _kvstore->create_pool(pool_name(), ( many_count_target * 64U * 3U * 2U + 4 * single_value_size ) * 8U, component::IKVStore::FLAGS_CREATE_ONLY, estimated_object_count);
  ASSERT_NE(0, int64_t(pool));
  auto pool2 = _kvstore->open_pool(pool_name());
  {
    void *v0;
    size_t s = 4096;
    auto r0 = _kvstore->allocate_pool_memory(pool, 8, s, v0);
    EXPECT_EQ(S_OK, r0);
    void *v1;
    auto r1 = _kvstore->allocate_pool_memory(pool, 8, s, v1);
    EXPECT_EQ(S_OK, r1);
    EXPECT_NE(v0, v1);
  }
  {
    auto rc = _kvstore->close_pool(pool2);
    EXPECT_EQ(S_OK, rc);
  }
  {
    auto rc = _kvstore->close_pool(pool);
    EXPECT_EQ(S_OK, rc);
  }
  /* All open handles closed */
  {
    auto rc = _kvstore->close_pool(pool2);
    EXPECT_NE(S_OK, rc);
  }
  pool = _kvstore->open_pool(pool_name());
  ASSERT_LT(0, int64_t(pool));
}

TEST_F(KVStore_test, BasicGet0)
{
  ASSERT_NE(nullptr, _kvstore);
  {
    std::vector<uint64_t> attr;
    auto r = _kvstore->get_attribute(pool, IKVStore::PERCENT_USED, attr, nullptr);
    EXPECT_EQ(S_OK, r);
    if ( S_OK == r )
    {
      EXPECT_GE(100, attr[0]);
      PINF("Percent used %zu", attr[0]);
    }
  }

  void * value = nullptr;
  size_t value_len = 0;

  auto r = _kvstore->get(pool, single_key, value, value_len);
  EXPECT_NE(S_OK, r);
  if( r == S_OK )
  {
    ASSERT_EQ("Key already exists", "Did you forget to delete the pool before running the test?");
  }
  _kvstore->free_memory(value);
}

TEST_F(KVStore_test, BasicPut)
{
  ASSERT_NE(nullptr, _kvstore);
  single_value.resize(single_value_size);

  auto r = _kvstore->put(pool, single_key, single_value.data(), single_value.length());
  EXPECT_EQ(S_OK, r);
}

TEST_F(KVStore_test, BasicGet1)
{
  ASSERT_NE(nullptr, _kvstore);
  void * value = nullptr;
  size_t value_len = 0;
  auto r = _kvstore->get(pool, single_key, value, value_len);
  EXPECT_EQ(S_OK, r);
  if ( S_OK == r )
  {
    PINF("Value=(%.*s) %zu", static_cast<int>(value_len), static_cast<char *>(value), value_len);
    EXPECT_EQ(single_value.size(), value_len);
    EXPECT_EQ(0, memcmp(single_value.data(), value, single_value.size()));
    _kvstore->free_memory(value);
  }
}

TEST_F(KVStore_test, BasicPutLocked)
{
  ASSERT_NE(nullptr, _kvstore);
  single_value.resize(single_value_size);
  void *value0 = nullptr;
  std::size_t value0_len = 0;

  {
    auto lk = IKVStore::key_t();

    {
      size_t alignment = 0;
      auto r = _kvstore->lock(pool, single_key, IKVStore::STORE_LOCK_READ, value0, value0_len, alignment, lk);
      EXPECT_EQ(S_OK, r);
      EXPECT_NE(nullptr, lk);
      r = _kvstore->put(pool, single_key, single_value.data(), single_value.length());
      EXPECT_EQ(E_LOCKED, r);
    }
    {
      auto r = _kvstore->resize_value(pool, single_key, single_value.size(), 64);
      EXPECT_EQ(E_LOCKED, r);
    }

    {
      void * value = nullptr;
      size_t value_len = 0;
      auto r = _kvstore->get(pool, single_key, value, value_len);
      EXPECT_EQ(S_OK, r);
      EXPECT_EQ(single_value.size(), value_len);
      if ( r == S_OK )
      {
        _kvstore->free_memory(value);
      }
    }

    {
      auto r = _kvstore->unlock(pool, lk);
      EXPECT_EQ(S_OK, r);
    }
  }

  std::size_t small_size = 16; /* Small, but large enough to preserve all characters in "Hello world!" */

  /* resize to an inline size. */
  {
    auto r = _kvstore->resize_value(pool, single_key, small_size, 16);
    EXPECT_EQ(S_OK, r);
  }
  {
    void *v = nullptr;
    std::size_t v_len = 0;
    auto lk = IKVStore::key_t();
    size_t alignment = 0;
    auto r = _kvstore->lock(pool, single_key, IKVStore::STORE_LOCK_READ, v, v_len, alignment, lk);
    EXPECT_EQ(S_OK, r);
    EXPECT_NE(nullptr, v);
    EXPECT_EQ(16, v_len);
#if 0
    /* alignment is just a suggestion, so we cannot insist on it, especiall for small-sized values */
    EXPECT_EQ(0, reinterpret_cast<std::size_t>(v) % 16);
#endif
    /* We cannot insist that v be unaligned to 256, but if it is aligned then the following test will verify nothing */
    EXPECT_NE(0, reinterpret_cast<std::size_t>(v) % 256);
    if ( r == S_OK )
    {
      r = _kvstore->unlock(pool, lk);
      EXPECT_EQ(S_OK, r);
    }
  }

  {
    void * value = nullptr;
    size_t value_len = 0;
    auto r = _kvstore->get(pool, single_key, value, value_len);
    EXPECT_EQ(S_OK, r);
    EXPECT_EQ(small_size, value_len);
    if ( r == S_OK )
    {
      _kvstore->free_memory(value);
    }
  }

  {
    auto r = _kvstore->resize_value(pool, single_key, single_value.size() / 2, 256);
    EXPECT_EQ(S_OK, r);
  }
  {
    void *v = nullptr;
    std::size_t v_len = 0;
    auto lk = IKVStore::key_t();
    size_t alignment = 0;
    auto r = _kvstore->lock(pool, single_key, IKVStore::STORE_LOCK_READ, v, v_len, alignment, lk);
    EXPECT_EQ(S_OK, r);
    EXPECT_NE(nullptr, v);
    EXPECT_EQ(single_value.size() / 2, v_len);
    EXPECT_EQ(0, reinterpret_cast<std::size_t>(v) % 256);
    /* We cannot insist that v be unaligned to 1024, but if it is aligned then the following 1024 test will verify nothing */
    EXPECT_NE(0, reinterpret_cast<std::size_t>(v) % 1024);
    if ( r == S_OK )
    {
      r = _kvstore->unlock(pool, lk);
      EXPECT_EQ(S_OK, r);
    }
  }

  {
    void * value = nullptr;
    size_t value_len = 0;
    auto r = _kvstore->get(pool, single_key, value, value_len);
    EXPECT_EQ(S_OK, r);
    EXPECT_EQ(single_value.size() / 2, value_len);
    if ( r == S_OK )
    {
      _kvstore->free_memory(value);
    }
  }

  {
    auto r = _kvstore->resize_value(pool, single_key, single_value.size(), 1024);
    EXPECT_EQ(S_OK, r);
  }
  {
    void *v = nullptr;
    std::size_t v_len = 0;
    auto lk = IKVStore::key_t();
    size_t alignment = 0;
    auto r = _kvstore->lock(pool, single_key, IKVStore::STORE_LOCK_READ, v, v_len, alignment, lk);
    EXPECT_EQ(S_OK, r);
    EXPECT_NE(nullptr, v);
    EXPECT_EQ(single_value.size(), v_len);
    EXPECT_EQ(0, reinterpret_cast<std::size_t>(v) % 1024);
    if ( r == S_OK )
    {
      r = _kvstore->unlock(pool, lk);
      EXPECT_EQ(S_OK, r);
    }
  }

  {
    void * value = nullptr;
    size_t value_len = 0;
    auto r = _kvstore->get(pool, single_key, value, value_len);
    EXPECT_EQ(S_OK, r);
    EXPECT_EQ(single_value.size(), value_len);
    if ( r == S_OK )
    {
      _kvstore->free_memory(value);
    }
  }

  /* Reqeusted change in behavior: length 0 is now a magic value which means
   * "do not try to create an element for the key, if not found"
   */
  {
    void *v = nullptr;
    size_t value_len = 0;
    auto lk = IKVStore::key_t();
    size_t alignment = 0;
    auto r = _kvstore->lock(pool, missing_key, IKVStore::STORE_LOCK_READ, v, value_len, alignment, lk);
    EXPECT_EQ(IKVStore::E_KEY_NOT_FOUND, r);
  }

}

TEST_F(KVStore_test, BasicGet2)
{
  ASSERT_NE(nullptr, _kvstore);
  void * value = nullptr;
  size_t value_len = 0;
  auto r = _kvstore->get(pool, single_key, value, value_len);
  EXPECT_EQ(S_OK, r);
  if ( S_OK == r )
  {
    PINF("Value=(%.*s) %zu", static_cast<int>(value_len), static_cast<char *>(value), value_len);
    EXPECT_EQ(single_value.size(), value_len);
    EXPECT_EQ(0, memcmp(single_value.data(), value, single_value.size()));
    _kvstore->free_memory(value);
  }
}

/* hstore issue 41 specifies different implementations for same-size replace vs different-size replace. */
TEST_F(KVStore_test, BasicReplaceSameSize)
{
  ASSERT_NE(nullptr, _kvstore);
  {
    single_value_updated_same_size.resize(single_value_size);
    auto r = _kvstore->put(pool, single_key, single_value_updated_same_size.data(), single_value_updated_same_size.length());
    EXPECT_EQ(S_OK, r);
  }
  void * value = nullptr;
  size_t value_len = 0;
  auto r = _kvstore->get(pool, single_key, value, value_len);
  EXPECT_EQ(S_OK, r);
  if ( S_OK == r )
  {
    PINF("Value=(%.*s) %zu", static_cast<int>(value_len), static_cast<char *>(value), value_len);
    EXPECT_EQ(0, memcmp(single_value_updated_same_size.data(), value, single_value_updated_same_size.size()));
    _kvstore->free_memory(value);
  }
}

TEST_F(KVStore_test, BasicReplaceDifferentSize)
{
  ASSERT_NE(nullptr, _kvstore);
  {
    auto r = _kvstore->put(pool, single_key, single_value_updated_different_size.data(), single_value_updated_different_size.length());
    EXPECT_EQ(S_OK, r);
  }
  void * value = nullptr;
  size_t value_len = 0;
  auto r = _kvstore->get(pool, single_key, value, value_len);
  EXPECT_EQ(S_OK, r);
  if ( S_OK == r )
  {
    PINF("Value=(%.*s) %zu", static_cast<int>(value_len), static_cast<char *>(value), value_len);
    EXPECT_EQ(0, memcmp(single_value_updated_different_size.data(), value, single_value_updated_different_size.size()));
    _kvstore->free_memory(value);
  }
}

TEST_F(KVStore_test, PopulateMany)
{
  ASSERT_NE(nullptr, _kvstore);
  ASSERT_LT(0, int64_t(pool));
  std::mt19937_64 r0{};
  for ( auto i = std::size_t(); i != many_count_target; ++i )
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
  ASSERT_NE(nullptr, _kvstore);
  ASSERT_LT(0, int64_t(pool));
  many_count_actual = 0;

  for ( auto &kv : kvv )
  {
    const auto &key = std::get<0>(kv);
    const auto &value = std::get<1>(kv);
    void * old_value = nullptr;
    size_t old_value_len = 0;
    if ( S_OK == _kvstore->get(pool, key, old_value, old_value_len) )
    {
      _kvstore->free_memory(old_value);
      ++extant_count;
    }
    else
    {
      auto r = _kvstore->put(pool, key, value.data(), value.length());
      EXPECT_EQ(S_OK, r);
      if ( r == S_OK )
      {
        many_insert_set.insert(key);
        ++many_count_actual;
      }
    }
  }
  EXPECT_LE(many_count_actual, many_count_target);
  EXPECT_LE(many_count_target * 99 / 100, many_count_actual);
}

TEST_F(KVStore_test, BasicMap)
{
  ASSERT_NE(nullptr, _kvstore);
  ASSERT_LT(0, int64_t(pool));
  auto value_len_sum = std::size_t();
  _kvstore->map(
    pool
    , [&value_len_sum] (
        const void * // key
        , const size_t // key_len
        , const void * // value
        , const size_t value_len
      ) -> int
      {
        value_len_sum += value_len;
        return 0;
      }
    );
  EXPECT_EQ(single_value_updated_different_size.length() + many_count_actual * many_value_length, value_len_sum);
}

TEST_F(KVStore_test, BasicMapKeys)
{
  ASSERT_NE(nullptr, _kvstore);
  ASSERT_LT(0, int64_t(pool));
  auto key_len_sum = std::size_t();
  _kvstore->map_keys(
    pool
    , [&key_len_sum](
        const std::string &key
      ) -> int
      {
        key_len_sum += key.size();
        return 0;
      }
  );
  EXPECT_EQ(single_key.size() + many_count_actual * many_key_length, key_len_sum);
}

TEST_F(KVStore_test, Count1)
{
  ASSERT_NE(nullptr, _kvstore);
  auto count = _kvstore->count(pool);
  /* count should reflect Put, PutMany */
  EXPECT_EQ(single_count + many_count_actual, count);
}

TEST_F(KVStore_test, CountByBucket)
{
/* buckets implemented, but not available at the kvstore interface */
#if 0
  std::uint64_t count = 0;
  _kvstore->debug(pool, 2 /* COUNT_BY_BUCKET */, reinterpret_cast<std::uint64_t>(&count));
  /* should reflect Put, PutMany */
  EXPECT_EQ(single_count + many_count_actual, count);
#endif
}

TEST_F(KVStore_test, ClosePool)
{
  ASSERT_NE(nullptr, _kvstore);
  {
    std::vector<uint64_t> attr;
    auto r = _kvstore->get_attribute(pool, IKVStore::PERCENT_USED, attr, nullptr);
    EXPECT_EQ(S_OK, r);
    if ( S_OK == r )
    {
      EXPECT_GE(100, attr[0]);
      PINF("Percent used %zu", attr[0]);
    }
  }

  if ( pmem_effective )
  {
    _kvstore->close_pool(pool);
  }
}

TEST_F(KVStore_test, OpenPool)
{
  ASSERT_TRUE(_kvstore);
  if ( pmem_effective )
  {
    pool = _kvstore->open_pool(pool_name(), 0);
  }
  ASSERT_LT(0, int64_t(pool));

  {
    std::vector<uint64_t> attr;
    auto r = _kvstore->get_attribute(pool, IKVStore::PERCENT_USED, attr, nullptr);
    EXPECT_EQ(S_OK, r);
    if ( S_OK == r )
    {
      EXPECT_GE(100, attr[0]);
      PINF("Percent used %zu", attr[0]);
    }
  }
}

TEST_F(KVStore_test, Size2a)
{
  ASSERT_NE(nullptr, _kvstore);
  auto count = _kvstore->count(pool);
  /* count should reflect Put, PutMany */
  EXPECT_EQ(single_count + many_count_actual, count);
}

TEST_F(KVStore_test, BasicGet3)
{
  ASSERT_NE(nullptr, _kvstore);
  void * value = nullptr;
  size_t value_len = 0;
  auto r = _kvstore->get(pool, single_key, value, value_len);
  EXPECT_EQ(S_OK, r);
  if ( S_OK == r )
  {
    PINF("Value=(%.*s) %zu", static_cast<int>(value_len), static_cast<char *>(value), value_len);
    _kvstore->free_memory(value);
  }
}

TEST_F(KVStore_test, BasicGetAttribute)
{
  ASSERT_NE(nullptr, _kvstore);
  {
    std::vector<uint64_t> attr;
    auto r = _kvstore->get_attribute(pool, IKVStore::VALUE_LEN, attr, &single_key);
    EXPECT_EQ(S_OK, r);
    if ( S_OK == r )
    {
      EXPECT_EQ(1, attr.size());
      if ( 1 == attr.size() )
      {
        EXPECT_EQ(attr[0], single_value_updated_different_size.length());
      }
    }
    r = _kvstore->get_attribute(pool, component::IKVStore::Attribute(0), attr, &single_key);
    EXPECT_EQ(E_NOT_SUPPORTED, r);
    r = _kvstore->get_attribute(pool, IKVStore::VALUE_LEN, attr, nullptr);
    EXPECT_EQ(E_BAD_PARAM, r);
    r = _kvstore->get_attribute(pool, IKVStore::VALUE_LEN, attr, &missing_key);
    EXPECT_EQ(IKVStore::E_KEY_NOT_FOUND, r);
  }

  /* verify that item still exists */
  { // 511
    void * value = nullptr;
    size_t value_len = 0;
    auto r = _kvstore->get(pool, single_key, value, value_len);
    EXPECT_EQ(S_OK, r);
    if ( S_OK == r )
    {
      PINF("Value=(%.*s) %zu", static_cast<int>(value_len), static_cast<char *>(value), value_len);
      _kvstore->free_memory(value);
    }
  }
}

TEST_F(KVStore_test, ResizeAttribute)
{
  ASSERT_NE(nullptr, _kvstore);
  std::vector<uint64_t> attr;

  auto r = _kvstore->get_attribute(pool, IKVStore::AUTO_HASHTABLE_EXPANSION, attr, nullptr);
  EXPECT_EQ(S_OK, r);
  ASSERT_EQ(1, attr.size());
  EXPECT_EQ(1, attr[0]);

  attr[0] = false;
  r = _kvstore->set_attribute(pool, IKVStore::AUTO_HASHTABLE_EXPANSION, attr, nullptr);
  EXPECT_EQ(S_OK, r);
  EXPECT_EQ(1, attr.size());

  attr.clear();
  r = _kvstore->get_attribute(pool, IKVStore::AUTO_HASHTABLE_EXPANSION, attr, nullptr);
  EXPECT_EQ(S_OK, r);
  ASSERT_EQ(1, attr.size());
  EXPECT_EQ(0, attr[0]);

  attr[0] = 34;
  r = _kvstore->set_attribute(pool, IKVStore::AUTO_HASHTABLE_EXPANSION, attr, nullptr);
  EXPECT_EQ(S_OK, r);
  EXPECT_EQ(1, attr.size());

  attr.clear();
  r = _kvstore->get_attribute(pool, IKVStore::AUTO_HASHTABLE_EXPANSION, attr, nullptr);
  EXPECT_EQ(S_OK, r);
  ASSERT_EQ(1, attr.size());
  EXPECT_EQ(1, attr[0]);
}

TEST_F(KVStore_test, Size2b)
{
  ASSERT_NE(nullptr, _kvstore);
  auto count = _kvstore->count(pool);
  /* count should reflect PutMany */
  EXPECT_EQ(single_count + many_count_actual, count);
}

TEST_F(KVStore_test, GetMany)
{
  ASSERT_NE(nullptr, _kvstore);
  {
    std::vector<uint64_t> attr;
    auto r = _kvstore->get_attribute(pool, IKVStore::PERCENT_USED, attr, nullptr);
    EXPECT_EQ(S_OK, r);
    if ( S_OK == r )
    {
      EXPECT_GE(100, attr[0]);
      PINF("Percent used %zu", attr[0]);
    }
  }

  ASSERT_LT(0, int64_t(pool));
  for ( auto i = 0; i != get_expand; ++i )
  {
    std::size_t mismatch_count = 0;
    for ( auto &kv : kvv )
    {
      const auto &key = std::get<0>(kv);
      const auto &ev = std::get<1>(kv);
      char value[many_value_length * 2];
      std::size_t value_len = many_value_length * 2;
      void *vp = value;
      auto r = _kvstore->get(pool, key, vp, value_len);
      EXPECT_EQ(S_OK, r);
      if ( S_OK == r )
      {
        EXPECT_EQ(vp, static_cast<void *>(value));
        EXPECT_EQ(ev.size(), value_len);
        mismatch_count += ( ev.size() != value_len || 0 != memcmp(ev.data(), value, ev.size()) );
      }
    }
    EXPECT_EQ(extant_count, mismatch_count);
  }
}

TEST_F(KVStore_test, GetManyAllocating)
{
  ASSERT_NE(nullptr, _kvstore);
  ASSERT_LT(0, int64_t(pool));
  for ( auto i = 0; i != get_expand; ++i )
  {
    std::size_t mismatch_count = 0;
    for ( auto &kv : kvv )
    {
      const auto &key = std::get<0>(kv);
      const auto &ev = std::get<1>(kv);
      void * value = nullptr;
      std::size_t value_len = 0;
      auto r = _kvstore->get(pool, key, value, value_len);
      EXPECT_EQ(S_OK, r);
      if ( S_OK == r )
      {
        EXPECT_EQ(ev.size(), value_len);
        mismatch_count += ( ev.size() != value_len || 0 != memcmp(ev.data(), value, ev.size()) );
        _kvstore->free_memory(value);
      }
    }
    EXPECT_EQ(extant_count, mismatch_count);
  }
}

TEST_F(KVStore_test, GetDirectMany)
{
  ASSERT_NE(nullptr, _kvstore);
  ASSERT_LT(0, int64_t(pool));
  for ( auto i = 0; i != get_expand; ++i )
  {
    std::size_t mismatch_count = 0;
    for ( auto &kv : kvv )
    {
      const auto &key = std::get<0>(kv);
      const auto &ev = std::get<1>(kv);
      char value[many_value_length * 2];
      size_t value_len = many_value_length * 2;
      auto r = _kvstore->get_direct(pool, key, value, value_len);
      EXPECT_EQ(S_OK, r);
      if ( S_OK == r )
      {
        EXPECT_EQ(ev.size(), value_len);
        mismatch_count += ( ev.size() != value_len || 0 != memcmp(ev.data(), value, ev.size()) );
      }
    }
    EXPECT_EQ(extant_count, mismatch_count);
  }
}

TEST_F(KVStore_test, GetRegions)
{
  ASSERT_NE(nullptr, _kvstore);
  ASSERT_LT(0, int64_t(pool));
  nupm::region_descriptor v;
  auto r = _kvstore->get_pool_regions(pool, v);
  EXPECT_EQ(S_OK, r);
  if ( S_OK == r )
  {
    EXPECT_EQ(1, v.address_map().size());
    if ( 1 == v.address_map().size() )
    {
      PMAJOR("Pool region at %p len %zu", ::base(v.address_map().front()), ::size(v.address_map().front()));
      auto iov_base = reinterpret_cast<std::uintptr_t>(::base(v.address_map().front()));
      /* region no longer needs to be well-aligned, but heap_cc still aligns to a
       * page boundary.
       */
      EXPECT_EQ(iov_base & 0xfff, 0);
      EXPECT_GT(::size(v.address_map().front()), many_count_target * 64U * 3U * 2U);
      EXPECT_LT(::size(v.address_map().front()), GiB(512));
    }
  }
}

TEST_F(KVStore_test, LockMany)
{
  ASSERT_NE(nullptr, _kvstore);
  ASSERT_LT(0, int64_t(pool));
  /* Lock for read (should succeed)
   * Lock again for read (should succeed).
   * Lock for write (should fail).
   * Lock a non-existent key for write, creating the key).
   *
   * Undo the three successful locks.
   */
  unsigned ct = 0;
  for ( auto &kv : kvv )
  {
#if __cplusplus < 201703L
    static constexpr auto KEY_NONE = +IKVStore::KEY_NONE;
#else
    static constexpr auto KEY_NONE = IKVStore::KEY_NONE;
#endif
    if ( ct == lock_count ) { break; }
    const auto &key = std::get<0>(kv);
    const auto &ev = std::get<1>(kv);
    const auto key_new = std::get<0>(kv) + "x";
    void *value0 = nullptr;
    std::size_t value0_len = 0;

    shared_lock rk0(_kvstore, &pool);
    const char *kf;
    size_t alignment = 0;
    auto r0 = _kvstore->lock(pool, key, IKVStore::STORE_LOCK_READ, value0, value0_len, alignment, rk0.k, &kf);
    EXPECT_EQ(S_OK, r0);
    EXPECT_NE(KEY_NONE, rk0.k);
    if ( S_OK == r0 && IKVStore::KEY_NONE != rk0.k )
    {
      EXPECT_EQ(many_value_length, value0_len);
      EXPECT_EQ(0, memcmp(ev.data(), value0, ev.size()));
      EXPECT_EQ(0, memcmp(key.data(), kf, key.size()));
    }
    void * value1 = nullptr;
    std::size_t value1_len = 0;
    shared_lock rk1(_kvstore, &pool);
    alignment = 0;
    auto r1 = _kvstore->lock(pool, key, IKVStore::STORE_LOCK_READ, value1, value1_len, alignment, rk1.k, &kf);
    EXPECT_EQ(S_OK, r1);
    EXPECT_NE(KEY_NONE, rk1.k);
    if ( S_OK == r1 && IKVStore::KEY_NONE != rk1.k )
    {
      EXPECT_EQ(many_value_length, value1_len);
      EXPECT_EQ(0, memcmp(ev.data(), value1, ev.size()));
      EXPECT_EQ(0, memcmp(key.data(), kf, key.size()));
    }
    /* Exclusive locking test. */

    void * value2 = nullptr;
    std::size_t value2_len = 0;
    IKVStore::key_t k2 = IKVStore::key_t();
    alignment = 0;
    auto r2 = _kvstore->lock(pool, key, IKVStore::STORE_LOCK_WRITE, value2, value2_len, alignment, k2);
    /* Undocumented behavior: lock conflict returns E_LOCKED */
    EXPECT_EQ(E_LOCKED, r2);
    EXPECT_EQ(KEY_NONE, k2);

    void * value3 = nullptr;
    std::size_t value3_len = many_value_length;
    try
    {
      const char *kfx;
      exclusive_lock m3(_kvstore, &pool);
      alignment = 0;
      auto r3 = _kvstore->lock(pool, key_new, IKVStore::STORE_LOCK_WRITE, value3, value3_len, alignment, m3.k, &kfx);
      /* Used to return S_OK; no longer does so */
      EXPECT_EQ(S_OK_CREATED, r3);
      EXPECT_NE(KEY_NONE, m3.k);
      if ( S_OK == r3 && IKVStore::KEY_NONE != m3.k )
      {
        EXPECT_EQ(many_value_length, value3_len);
        EXPECT_NE(nullptr, value3);
        EXPECT_EQ(0, memcmp(key_new.data(), kfx, key_new.size()));
      }
      ++ct;
    }
    catch (const std::system_error &)
    {
      /* Lock a non-existent key will fail if the key locked for read needs to be moved. */
    }
  }
}

TEST_F(KVStore_test, Size2c)
{
  ASSERT_NE(nullptr, _kvstore);
  /* verify that item still exists */
  { // 770
    void * value = nullptr;
    size_t value_len = 0;
    auto r = _kvstore->get(pool, single_key, value, value_len);
    EXPECT_EQ(S_OK, r);
    if ( S_OK == r )
    {
      PINF("line %d Value=(%.*s) %zu", __LINE__, static_cast<int>(value_len), static_cast<char *>(value), value_len);
      _kvstore->free_memory(value);
    }
  }
  auto count = _kvstore->count(pool);
  /* count should reflect Put, PutMany and LockMany */
  EXPECT_EQ(single_count + many_count_actual + lock_count, count);
}

/* Missing:
 *  - test of invalid parameters
 *    - offsets greater than sizeof data
 *    - non-existent key
 *    - invalid operations
 *  - test of crash recovery
 */
TEST_F(KVStore_test, BasicUpdate)
{
  ASSERT_NE(nullptr, _kvstore);
  /* verify that item still exists */
  { // 796
    void * value = nullptr;
    size_t value_len = 0;
    auto r = _kvstore->get(pool, single_key, value, value_len);
    EXPECT_EQ(S_OK, r);
    if ( S_OK == r )
    {
      PINF("line %d Value=(%.*s) %zu", __LINE__, static_cast<int>(value_len), static_cast<char *>(value), value_len);
      _kvstore->free_memory(value);
    }
  }
  {
    std::vector<std::unique_ptr<IKVStore::Operation>> v;
    v.emplace_back(std::make_unique<IKVStore::Operation_write>(0, 1, "W"));
    v.emplace_back(std::make_unique<IKVStore::Operation_write>(2, 3, "XYZ"));
    std::vector<IKVStore::Operation *> v2;
    std::transform(v.begin(), v.end(), std::back_inserter(v2), [] (const auto &i) { return i.get(); });
    auto r =
      _kvstore->atomic_update(
      pool
      , single_key
      , v2
    );
    EXPECT_EQ(S_OK, r);
  }

  {
    void * value = nullptr;
    size_t value_len = 0;
    auto r = _kvstore->get(pool, single_key, value, value_len);
    EXPECT_EQ(S_OK, r);
    if ( S_OK == r )
    {
      PINF("Value=(%.*s) %zu", static_cast<int>(value_len), static_cast<char *>(value), value_len);
      EXPECT_EQ(single_value_updated_different_size.size(), value_len);
      EXPECT_EQ(0, memcmp(single_value_updated3.data(), value, single_value_updated3.size()));
      _kvstore->free_memory(value);
    }
  }

  auto count = _kvstore->count(pool);
  EXPECT_EQ(single_count + many_count_actual + lock_count, count);
}

TEST_F(KVStore_test, BasicErase)
{
  ASSERT_NE(nullptr, _kvstore);
  {
    std::vector<uint64_t> attr;
    auto r = _kvstore->get_attribute(pool, IKVStore::PERCENT_USED, attr, nullptr);
    EXPECT_EQ(S_OK, r);
    if ( S_OK == r )
    {
      EXPECT_GE(100, attr[0]);
      PINF("Percent used %zu", attr[0]);
    }
  }

  {
    auto r = _kvstore->erase(pool, single_key);
    EXPECT_EQ(S_OK, r);
  }

  auto count = _kvstore->count(pool);
  EXPECT_EQ(many_count_actual + lock_count, count);
}

TEST_F(KVStore_test, EraseMany)
{
  ASSERT_NE(nullptr, _kvstore);
  auto erase_count = 0;
  for ( auto &kv : kvv )
  {
    const auto &key = std::get<0>(kv);
    auto r = _kvstore->erase(pool, key);
    if ( r == S_OK )
    {
      many_erase_okay.insert(key);
      ++erase_count;
    }
    else
    {
      many_erase_fail.insert(key);
    }
  }
  EXPECT_EQ(erase_count, many_erase_okay.size());
  EXPECT_EQ(many_count_actual, many_insert_set.size());
  EXPECT_LE(many_count_actual, erase_count);
  auto count = _kvstore->count(pool);
  EXPECT_EQ(lock_count, count);
}

TEST_F(KVStore_test, AllocDealloc4K)
{
  ASSERT_NE(nullptr, _kvstore);
  {
    std::vector<uint64_t> attr;
    auto r = _kvstore->get_attribute(pool, IKVStore::PERCENT_USED, attr, nullptr);
    EXPECT_EQ(S_OK, r);
    if ( S_OK == r )
    {
      EXPECT_GE(100, attr[0]);
      PINF("Percent used %zu", attr[0]);
    }
  }

  std::set<void *> allocations{};

  /* Many 4K allocations */
  for ( auto i = 0; i != 1000; ++i )
  {
    void *v = nullptr;
    size_t s = 4096;
    auto r = _kvstore->allocate_pool_memory(pool, 8, s, v);
    EXPECT_EQ(S_OK, r);
    EXPECT_EQ(0, allocations.count(v));
    allocations.insert(v);
    EXPECT_EQ(1, allocations.count(v));
  }

  /* close and reopen the pool */
  {
    auto rc = _kvstore->close_pool(pool);
    EXPECT_EQ(S_OK, rc);

    pool = _kvstore->open_pool(pool_name());
    ASSERT_LT(0, int64_t(pool));
  }

  /* Many 4K frees */
  for ( auto v : allocations )
  {
    auto r = _kvstore->free_pool_memory(pool, v, 4096);
    EXPECT_EQ(S_OK, r);
  }
}

TEST_F(KVStore_test, AllocDealloc)
{
  ASSERT_NE(nullptr, _kvstore);
  {
    std::vector<uint64_t> attr;
    auto r = _kvstore->get_attribute(pool, IKVStore::PERCENT_USED, attr, nullptr);
    EXPECT_EQ(S_OK, r);
    if ( S_OK == r )
    {
      EXPECT_GE(100, attr[0]);
      PINF("Percent used %zu", attr[0]);
    }
  }

  void *v = nullptr;
#if 0
  /* alignment is now a "hint", so 0 alignment is no longer an error */
  auto r = _kvstore->allocate_pool_memory(pool, 100, 0, v);
  EXPECT_EQ(component::IKVStore::E_BAD_ALIGNMENT, r); /* zero aligmnent */
#endif
  auto r = _kvstore->allocate_pool_memory(pool, 100, 1, v);
  EXPECT_EQ(component::IKVStore::E_BAD_ALIGNMENT, r); /* alignment less than sizeof(void *) */

  /* allocate various sizes */
  void *v128 = nullptr;
  void *v160 = nullptr;
  void *v704 = nullptr;
  void *v2048 = nullptr;
  void *v0 = nullptr;
  r = _kvstore->allocate_pool_memory(pool, 128, 32, v128);
  EXPECT_EQ(S_OK, r);
  EXPECT_NE(v128, nullptr);
  r = _kvstore->allocate_pool_memory(pool, 160, 8, v160);
  EXPECT_EQ(S_OK, r);
  EXPECT_NE(v160, nullptr);
  r = _kvstore->allocate_pool_memory(pool, 704, 64, v704);
  EXPECT_EQ(S_OK, r);
  EXPECT_NE(v704, nullptr);
  r = _kvstore->allocate_pool_memory(pool, 2048, 2048, v2048);
  EXPECT_EQ(S_OK, r);
  EXPECT_NE(v2048, nullptr);
  r = _kvstore->allocate_pool_memory(pool, 0, 8, v0);
  EXPECT_EQ(S_OK, r);
  EXPECT_NE(v0, nullptr);

  /* close and reopen the pool */
  {
    auto rc = _kvstore->close_pool(pool);
    EXPECT_EQ(S_OK, rc);

    pool = _kvstore->open_pool(pool_name());
    ASSERT_LT(0, int64_t(pool));
  }

  r = _kvstore->free_pool_memory(pool, v0, 0);
  EXPECT_EQ(S_OK, r);
  r = _kvstore->free_pool_memory(pool, v2048, 2048);
  EXPECT_EQ(S_OK, r);
  r = _kvstore->free_pool_memory(pool, v704, 704);
  EXPECT_EQ(S_OK, r);
  r = _kvstore->free_pool_memory(pool, v160, 160);
  EXPECT_EQ(S_OK, r);
  r = _kvstore->free_pool_memory(pool, v128, 128);
  EXPECT_EQ(S_OK, r);
#if 0
  /* Removed: unless the free logic has a sanity check, might
   * corrupt memory rather than failing.
   */
  r = _kvstore->free_pool_memory(pool, v128, 128);
  EXPECT_EQ(E_INVAL, r); /* not found */
#endif
}

TEST_F(KVStore_test, DeletePool)
{
  ASSERT_NE(nullptr, _kvstore);
  {
    std::vector<uint64_t> attr;
    auto r = _kvstore->get_attribute(pool, IKVStore::PERCENT_USED, attr, nullptr);
    EXPECT_EQ(S_OK, r);
    if ( S_OK == r )
    {
      EXPECT_GE(100, attr[0]);
      PINF("Percent used %zu", attr[0]);
    }
  }

  _kvstore->close_pool(pool);
  _kvstore->delete_pool(pool_name());
}

/* copied from mapstore */
TEST_F(KVStore_test, Timestamps)
{
	ASSERT_NE(nullptr, _kvstore);
	(void) KVStore_test::single_count;
	(void) KVStore_test::single_value_size;
	(void) KVStore_test::many_count_actual;
	(void) KVStore_test::extant_count;
	pool = _kvstore->create_pool("timestamp-test.pool", MiB(32), component::IKVStore::FLAGS_CREATE_ONLY);
	ASSERT_NE(0, int64_t(pool));

	/* if timestamping is enabled */
	if ( _kvstore->get_capability(IKVStore::Capability::WRITE_TIMESTAMPS) )
	{
		auto t0 = common::epoch_now();
		unsigned delay = 2;

		std::vector<std::string> keys;
		for ( unsigned i=0; i<10; ++i )
		{
			keys.push_back(common::random_string(8));
		}

		for ( unsigned i=0; i != keys.size(); ++i )
		{
			auto value = common::random_string(16);
			PLOG("adding key-value pair (%s)", keys[i].c_str());
			_kvstore->put(pool, keys[i], value.c_str(), value.size());
			sleep(delay);
		}

		std::size_t key_count = 0;

		/* All keys should be listed */
		_kvstore->map(
			pool
			, [&key_count](
				const void* key
				, const size_t key_len
				, const void* // value
				, const size_t // value_len
				, const common::tsc_time_t timestamp
			) -> bool
			{
				++key_count;
				PLOG("Timestamped record: %.*s @ %lu", int(key_len), static_cast<const char *>(key), timestamp.raw());
				return true;
			}
			, 0
			, 0
		);
		EXPECT_EQ(keys.size(), key_count);

		unsigned wait = 5;
		auto t0_plus_wait = t0;
		t0_plus_wait.add_seconds(wait);

		/* First two keys should not be listed */
		PLOG("After %u seconds", wait);
		key_count = 0;
		_kvstore->map(
			pool
			, [&key_count, wait](
				const void* key
				, const size_t key_len
				, const void* // value
				, const size_t // value_len
				, const common::tsc_time_t timestamp
			) -> bool
			{
				++key_count;
				std::ostringstream ts;
				ts << timestamp;
				PLOG("After %u Timestamped record: %.*s @ %s", wait
					, int(key_len), static_cast<const char *>(key), ts.str().c_str()
				);
				return true;
			}
			/* A time_point, expressed as seconds past epoch */
			, t0_plus_wait
			, 0
		);
		EXPECT_EQ(keys.size() - unsigned(std::ceil(double(wait)/double(delay))), key_count);

		auto t_before_swap = common::epoch_now();

		for ( unsigned i=0; i+1 < keys.size(); ++i )
		{
			auto value = common::random_string(16);
			auto key = common::random_string(8);
			PLOG("swapping keys (%s, %s)", keys[i].c_str(), keys[i+1].c_str());
			_kvstore->swap_keys(pool, keys[i], keys[i+1]);
			sleep(delay);
		}

		auto t_after_swap = common::epoch_now();
		t_after_swap.add_seconds(1); /* adjust for coarse granularity */

		PLOG(
			"Timestamps remaining from before swap [%lu .. %lu]"
			, t_before_swap.seconds()
			, t_after_swap.seconds()
		);
		key_count = 0;
		_kvstore->map(
			pool
			, [&key_count](
				const void* key
				, const size_t key_len
				, const void* // value
				, const size_t // value_len
				, const common::tsc_time_t timestamp
			) -> bool
			{
				++key_count;
				std::ostringstream ts;
				ts << timestamp;
				PLOG("Before swaps record: %.*s @ %s"
					, int(key_len), static_cast<const char *>(key), ts.str().c_str()
				);
				return true;
			}
			, 0
			, t_before_swap.sub_seconds(1) /* adjust for closed interval requirement */
		);
		EXPECT_EQ(0, key_count);

		key_count = 0;
		PLOG("Timestamps during swaps");
			_kvstore->map(
			pool
			, [&key_count](
				const void* key
				, const size_t key_len
				, const void* // value
				, const size_t // value_len
				, const common::tsc_time_t timestamp
			) -> bool
			{
				++key_count;
				std::ostringstream ts;
				ts << timestamp;
				PLOG("During swaps record: %.*s @ %s"
					, int(key_len), static_cast<const char *>(key), ts.str().c_str()
				);
				return true;
			}
			, t_before_swap
			, t_after_swap
		);
		EXPECT_EQ(keys.size(), key_count);

		key_count = 0;
		PLOG("Timestamps after swaps");
			_kvstore->map(
			pool
			, [&key_count](
				const void* key
				, const size_t key_len
				, const void* // value
				, const size_t // value_len
				, const common::tsc_time_t timestamp
			) -> bool
			{
				++key_count;
				std::ostringstream ts;
				ts << timestamp;
				PLOG("After swaps record: %.*s @ %s"
					, int(key_len), static_cast<const char *>(key), ts.str().c_str()
				);
				return true;
			}
			, t_after_swap
			, 0
		);
		EXPECT_EQ(0, key_count);
	}

  PLOG("Closing pool.");
  ASSERT_NE(IKVStore::POOL_ERROR+0, pool);
  ASSERT_EQ(S_OK, _kvstore->close_pool(pool));
}

/* copied from mapstore */
TEST_F(KVStore_test, Iterator)
{
  ASSERT_TRUE(_kvstore);
  pool = _kvstore->create_pool("iterator-test.pool", MiB(32), component::IKVStore::FLAGS_CREATE_ONLY);
  ASSERT_NE(0, int64_t(pool));

  common::epoch_time_t now = 0;

  for ( unsigned i=0; i<10; ++i )
  {
    auto value = common::random_string(16);
    auto key = common::random_string(8);

    if(i==5) { sleep(2); now = common::epoch_now(); }

    PLOG("(%u) adding key-value pair key(%s) value(%s)", i, key.c_str(),value.c_str());
    _kvstore->put(pool, key, value.c_str(), value.size());
  }

  _kvstore->map(
    pool
    , [](const void * key
      , const size_t key_len
      , const void * value
      , const size_t value_len
    ) -> int
    {
      PINF("key:(%p %.*s) value(%.*s)"
        , key, int(key_len), static_cast<const char *>(key), int(value_len),
        static_cast<const char *>(value)
      );
      return 0;
    }
  );

  PLOG("Iterating...");
  status_t rc;
  IKVStore::pool_reference_t ref;
  bool time_match;


  auto iter = _kvstore->open_pool_iterator(pool);
  while((rc = _kvstore->deref_pool_iterator(pool, iter, 0, 0, ref, time_match, true)) == S_OK) {
    PLOG("iterator: key(%.*s) value(%.*s) %lu",
         int(ref.key_len), static_cast<const char *>(ref.key),
         int(ref.value_len), static_cast<const char *>(ref.value),
         ref.timestamp.seconds());
  }
  _kvstore->close_pool_iterator(pool, iter);
  ASSERT_EQ(E_OUT_OF_BOUNDS, rc);

  iter = _kvstore->open_pool_iterator(pool);
  ASSERT_LT(0, now.seconds());
  while((rc = _kvstore->deref_pool_iterator(pool, iter, 0, now, ref, time_match, true)) == S_OK) {
    PLOG("(time-constrained) iterator: key(%.*s) value(%.*s) %lu (match=%s)",
         int(ref.key_len), static_cast<const char *>(ref.key),
         int(ref.value_len), static_cast<const char *>(ref.value),
         ref.timestamp.seconds(),
         time_match ? "y":"n");
  }
  _kvstore->close_pool_iterator(pool, iter);
  ASSERT_EQ(E_OUT_OF_BOUNDS, rc);



  PLOG("Disturbed iteration...");
  unsigned i=0;
  iter = _kvstore->open_pool_iterator(pool);
  while ( (rc = _kvstore->deref_pool_iterator(pool, iter, 0, 0, ref, time_match, true)) == S_OK )
  {
    PLOG("iterator: key(%.*s) value(%.*s) %lu",
         int(ref.key_len), static_cast<const char *>(ref.key),
         int(ref.value_len), static_cast<const char *>(ref.value),
         ref.timestamp.seconds());
    ++i;
    if ( i == 5 ) {
      /* disturb iteration */
      auto value = common::random_string(16);
      auto key = common::random_string(8);
      PLOG("adding key-value pair key(%s) value(%s)", key.c_str(),value.c_str());
      _kvstore->put(pool, key, value.c_str(), value.size());
    }
  }
  ASSERT_EQ(E_ITERATOR_DISTURBED, rc);

  PLOG("Closing pool.");
  ASSERT_NE(+IKVStore::POOL_ERROR, pool);
  ASSERT_EQ(S_OK, _kvstore->close_pool(pool));
}

/* copied from mapstore */
#define ASSERT_OK(X) ASSERT_TRUE(S_OK == X)
TEST_F(KVStore_test, KeySwap)
{
  ASSERT_TRUE(_kvstore);
  pool = _kvstore->create_pool("keyswap", MiB(32), component::IKVStore::FLAGS_CREATE_ONLY);

  std::string left_key = "LeftKey";
  std::string right_key = "RightKey";
  std::string left_value = "This is left";
  std::string right_value = "This is right";

  ASSERT_OK(_kvstore->put(pool, left_key, left_value.c_str(), left_value.length()));
  ASSERT_OK(_kvstore->put(pool, right_key, right_value.c_str(), right_value.length()));

  ASSERT_OK(_kvstore->swap_keys(pool, left_key, right_key));

  ::iovec new_left{}; /* ERROR: uninitialized in map_store test */
  _kvstore->get(pool, left_key, new_left.iov_base, new_left.iov_len);
  ::iovec new_right{}; /* ERROR: uninitialized in map_store test */
  _kvstore->get(pool, right_key, new_right.iov_base, new_right.iov_len);

  PLOG("left: %.*s", int(new_left.iov_len), static_cast<const char *>(new_left.iov_base));
  PLOG("right: %.*s", int(new_right.iov_len), static_cast<const char *>(new_right.iov_base));
  ASSERT_EQ(0, strncmp(static_cast<const char *>(new_left.iov_base), right_value.c_str(), new_left.iov_len));
  ASSERT_EQ(0, strncmp(static_cast<const char *>(new_right.iov_base), left_value.c_str(), new_right.iov_len));
  _kvstore->free_memory(new_left.iov_base);
  _kvstore->free_memory(new_right.iov_base);
  ASSERT_NE(pool, IKVStore::POOL_ERROR+0);
  ASSERT_EQ(S_OK, _kvstore->close_pool(pool));
}

/* copied from mapstore */
TEST_F(KVStore_test, OutOfMemory)
{
  ASSERT_TRUE(_kvstore);
  pool = _kvstore->create_pool("oom", MiB(32), component::IKVStore::FLAGS_CREATE_ONLY);

  {
    status_t rc = S_OK;
    while ( rc == S_OK )
    {
      void *p = 0;
      rc = _kvstore->allocate_pool_memory(pool, MiB(1), 8, p);
      (void) p;
    }
    EXPECT_EQ(IKVStore::E_TOO_LARGE, rc);
  }
  ASSERT_EQ(S_OK, _kvstore->close_pool(pool));
}


} // namespace

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
