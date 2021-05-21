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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <common/utils.h>
#include <common/str_utils.h>
#include <api/components.h>
#include <api/kvstore_itf.h>

#define ASSERT_OK(X) ASSERT_TRUE(S_OK == X)

using namespace component;

static component::IKVStore::pool_t pool;

namespace {

// The fixture for testing class Foo.
class KVStore_test : public ::testing::Test {

 protected:

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  virtual void SetUp() {
    // Code here will be called immediately after the constructor (right
    // before each test).
    {
      /* create object instance through factory */
      component::IBase * comp = component::load_component("libcomponent-mapstore.so",
                                                          component::mapstore_factory);

      ASSERT_TRUE(comp);
      auto fact = make_itf_ref(static_cast<IKVStore_factory *>(comp->query_interface(IKVStore_factory::iid())));

      _kvstore = fact->create(0, {});
    }
  }

  virtual void TearDown() {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

  // Objects declared here can be used by all tests in the test case
  static component::IKVStore * _kvstore;
};

component::IKVStore * KVStore_test::_kvstore;


//TEST_F(KVStore_test, Instantiate)

TEST_F(KVStore_test, OpenPool)
{
  ASSERT_TRUE(_kvstore);
  pool = _kvstore->create_pool("test1.pool", MB(32));

  if(pool == component::IKVStore::POOL_ERROR)
    pool = _kvstore->open_pool("test1.pool");

  ASSERT_TRUE(pool != component::IKVStore::POOL_ERROR);
}


TEST_F(KVStore_test, BasicPut)
{
  ASSERT_TRUE(pool);
  std::string key = "MyKey";
  std::string key2 = "MyKey2";
  std::string value = "Hello world!";
  //  value.resize(value.length()+1); /* append /0 */
  value.resize(KB(8));

  _kvstore->put(pool, key, value.c_str(), value.length());
  _kvstore->put(pool, key2, value.c_str(), value.length());
}

TEST_F(KVStore_test, BasicGet)
{
  std::string key = "MyKey";

  void * value = nullptr;
  size_t value_len = 0;
  _kvstore->get(pool, key, value, value_len);
  PINF("Value=(%.50s) %lu", static_cast<const char *>(value), value_len);

  ASSERT_TRUE(value);
  ASSERT_TRUE(value_len == KB(8));

  value = nullptr;
  value_len = 0;
  _kvstore->get(pool, key, value, value_len);
  PINF("Repeat Value=(%.50s) %lu", static_cast<const char *>(value), value_len);
  auto count = _kvstore->count(pool);
  PINF("Count = %ld", count);
  ASSERT_TRUE(count == 2);
  ASSERT_TRUE(value);
  ASSERT_TRUE(value_len == KB(8));
}

TEST_F(KVStore_test, BasicMap)
{
  _kvstore->map(pool,
                [](const void * key,
                   const size_t key_len,
                   const void * value,
                   const size_t value_len) -> int
                {
                  PINF("key:(%.*s) value(%.*s)", int(key_len), static_cast<const char *>(key), int(value_len), static_cast<const char *>(value));
                  return 0;
                }
                );
  //  _kvstore->erase(pool, "MyKey");
}

TEST_F(KVStore_test, ValueResize)
{
  _kvstore->map(pool,
                [](const void * key,
                   const size_t key_len,
                   const void * value,
                   const size_t value_len) -> int
                {
                  PINF("key:(%.*s) value(%.*s-%lu)", int(key_len), static_cast<const char *>(key),
                       int(value_len), static_cast<const char *>(value),
                       value_len);
                  return 0;
                }
                );

  ASSERT_TRUE(_kvstore->resize_value(pool,
                                     "MyKey",
                                     KB(16),
                                     8) == S_OK);

  _kvstore->map(pool,
                [](const void * key,
                   const size_t key_len,
                   const void * value,
                   const size_t value_len) -> int
                {
                  PINF("key:(%.*s) value(%.*s-%lu)", int(key_len), static_cast<const char *>(key),
                       int(value_len), static_cast<const char *>(value),
                       value_len);
                  return 0;
                }
                );

}


TEST_F(KVStore_test, BasicRemove)
{
  _kvstore->erase(pool, "MyKey");
}

TEST_F(KVStore_test, ClosePool)
{
  ASSERT_TRUE(_kvstore->close_pool(pool) == S_OK);
}

TEST_F(KVStore_test, ReopenPool)
{
  pool = _kvstore->open_pool("test1.pool");
  ASSERT_TRUE(pool != component::IKVStore::POOL_ERROR);
  PLOG("re-opened pool: %p", reinterpret_cast<const void *>(pool));
}

TEST_F(KVStore_test, ReClosePool)
{
  _kvstore->close_pool(pool);
}

TEST_F(KVStore_test, DeletePool)
{
  PLOG("deleting pool: %p", reinterpret_cast<const void *>(pool));
  _kvstore->delete_pool("test1.pool");
}

TEST_F(KVStore_test, Timestamps)
{
  ASSERT_TRUE(_kvstore);
  pool = _kvstore->create_pool("timestamp-test.pool", MB(32));

  /* if timestamping is enabled */
  if(_kvstore->get_capability(IKVStore::Capability::WRITE_TIMESTAMPS)) {

    auto now = common::epoch_now();
    
    for(unsigned i=0;i<10;i++) {
      auto value = common::random_string(16);
      auto key = common::random_string(8);
      PLOG("adding key-value pair (%s)", key.c_str());
      _kvstore->put(pool, key, value.c_str(), value.size());
      sleep(2);
    }


    _kvstore->map(pool, [](const void* key,
                           const size_t key_len,
                           const void* value,
                           const size_t value_len,
                           const common::tsc_time_t timestamp) -> bool {
                    (void)value; // unused
                    (void)value_len; // unused
                    PLOG("Timestamped record: %.*s @ %lu",
			 int(key_len),
			 static_cast<const char *>(key),
			 timestamp.raw());
                    return true;
                  }, 0, 0);

    PLOG("After 5 seconds");
    _kvstore->map(pool, [](const void* key,
                           const size_t key_len,
                           const void* value,
                           const size_t value_len,
                           const common::tsc_time_t timestamp) -> bool
			{
			  (void)value; // unused
			  (void)value_len; // unused
			  PLOG("After 5 Timestamped record: %.*s @ %lu",
			       int(key_len),
			       static_cast<const char *>(key),
			       timestamp.raw());
			  return true;
			}
		  , now.add_seconds(5)
		  , {0,0});
  }

  PLOG("Closing pool.");
  ASSERT_TRUE(pool != IKVStore::POOL_ERROR);
  ASSERT_TRUE(_kvstore->close_pool(pool) == S_OK);
}

TEST_F(KVStore_test, Iterator)
{
  ASSERT_TRUE(_kvstore);
  pool = _kvstore->create_pool("iterator-test.pool", MB(32));

  common::epoch_time_t now = 0;

  for(unsigned i=0;i<10;i++) {
    auto value = common::random_string(16);
    auto key = common::random_string(8);

    if(i==5) { sleep(2); now = common::epoch_now(); }

    PLOG("(%u) adding key-value pair key(%s) value(%s)", i, key.c_str(),value.c_str());
    _kvstore->put(pool, key, value.c_str(), value.size());
  }

  _kvstore->map(pool,
                [](const void * key,
                   const size_t key_len,
                   const void * value,
                   const size_t value_len) -> int
                {
                  PINF("key:(%p %.*s) value(%.*s)", key, int(key_len), static_cast<const char *>(key), int(value_len),
                       static_cast<const char *>(value));
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
  ASSERT_TRUE(rc == E_OUT_OF_BOUNDS);

  iter = _kvstore->open_pool_iterator(pool);
  ASSERT_TRUE(now.seconds() > 0);
  while((rc = _kvstore->deref_pool_iterator(pool, iter, 0, now, ref, time_match, true)) == S_OK) {
    PLOG("(time-constrained) iterator: key(%.*s) value(%.*s) %lu (match=%s)",
         int(ref.key_len), static_cast<const char *>(ref.key),
         int(ref.value_len), static_cast<const char *>(ref.value),
         ref.timestamp.seconds(),
         time_match ? "y":"n");
  }
  _kvstore->close_pool_iterator(pool, iter);
  ASSERT_TRUE(rc == E_OUT_OF_BOUNDS);



  PLOG("Disturbed iteration...");
  unsigned i=0;
  iter = _kvstore->open_pool_iterator(pool);
  while((rc = _kvstore->deref_pool_iterator(pool, iter, 0, 0, ref, time_match, true)) == S_OK) {
    PLOG("iterator: key(%.*s) value(%.*s) %lu",
         int(ref.key_len), static_cast<const char *>(ref.key),
         int(ref.value_len), static_cast<const char *>(ref.value),
         ref.timestamp.seconds());
    i++;
    if(i == 5) {
      /* disturb iteration */
      auto value = common::random_string(16);
      auto key = common::random_string(8);
      PLOG("adding key-value pair key(%s) value(%s)", key.c_str(),value.c_str());
      _kvstore->put(pool, key, value.c_str(), value.size());
    }
  }
  ASSERT_TRUE(rc == E_ITERATOR_DISTURBED);

  PLOG("Closing pool.");
  ASSERT_TRUE(pool != IKVStore::POOL_ERROR);
  ASSERT_TRUE(_kvstore->close_pool(pool) == S_OK);
}


TEST_F(KVStore_test, KeySwap)
{
  ASSERT_TRUE(_kvstore);
  pool = _kvstore->create_pool("keyswap", MB(32));
  ASSERT_TRUE(pool != IKVStore::POOL_ERROR);
  
  std::string left_key = "LeftKey";
  std::string right_key = "RightKey";
  std::string left_value = "This is left";
  std::string right_value = "This is right";

  ASSERT_OK(_kvstore->put(pool, left_key, left_value.c_str(), left_value.length()));
  ASSERT_OK(_kvstore->put(pool, right_key, right_value.c_str(), right_value.length()));

  ASSERT_OK(_kvstore->swap_keys(pool, left_key, right_key));

  iovec new_left, new_right;
  _kvstore->get(pool, left_key, new_left.iov_base, new_left.iov_len);
  _kvstore->get(pool, right_key, new_right.iov_base, new_right.iov_len);

  PLOG("left: %.*s", int(new_left.iov_len), static_cast<const char *>(new_left.iov_base));
  PLOG("right: %.*s", int(new_right.iov_len), static_cast<const char *>(new_right.iov_base));
  ASSERT_TRUE(strncmp(static_cast<const char *>(new_left.iov_base), right_value.c_str(), new_left.iov_len) == 0);
  ASSERT_TRUE(strncmp(static_cast<const char *>(new_right.iov_base), left_value.c_str(), new_right.iov_len) == 0);
  
  ASSERT_TRUE(_kvstore->close_pool(pool) == S_OK);
}

TEST_F(KVStore_test, AlignedLock)
{
  ASSERT_TRUE(_kvstore);
  pool = _kvstore->create_pool("alignedlock", MB(32));
  ASSERT_TRUE(pool != IKVStore::POOL_ERROR);

  std::vector<size_t> alignments = {8,16,32,128,512,2048,4096};

  for(size_t alignment: alignments)
  {
    void * addr = nullptr;
    size_t value_len = 4096;
    IKVStore::key_t handle, handle2;

    ASSERT_TRUE(_kvstore->lock(pool, "key", IKVStore::STORE_LOCK_WRITE, addr, value_len, alignment, handle) == S_OK_CREATED);
    ASSERT_TRUE(check_aligned(addr, alignment));
    ASSERT_TRUE(_kvstore->lock(pool, "key", IKVStore::STORE_LOCK_WRITE, addr, value_len, alignment, handle2) == E_LOCKED);
    ASSERT_OK(_kvstore->unlock(pool, handle));
    ASSERT_OK(_kvstore->erase(pool, "key"));
  }

  ASSERT_TRUE(_kvstore->close_pool(pool) == S_OK);
}

} // namespace

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
