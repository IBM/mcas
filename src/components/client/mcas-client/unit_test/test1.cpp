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
#define LARGE_VALUE_SIZE KB(128);  // MB(4);
#define LARGE_ITERATIONS 10000
#define EXPECTED_OBJECTS 10000

#include <api/components.h>
#include <api/kvstore_itf.h>
#include <common/cpu.h>
#include <common/str_utils.h>
#include <common/task.h>
#include <gtest/gtest.h>
#include <sys/mman.h>

#include <boost/program_options.hpp>
#include <boost/optional.hpp>
#include <chrono> /* milliseconds */
#include <iostream>
#include <thread> /* this_thread::sleep_for */

//#define TEST_PERF_SMALL_PUT
//#define TEST_PERF_SMALL_GET_DIRECT
#define TEST_PERF_LARGE_PUT_DIRECT
#define TEST_PERF_LARGE_GET_DIRECT

//#define TEST_SCALE_IOPS

// #define TEST_PERF_SMALL_PUT_DIRECT

struct {
  std::string                  addr;
  std::string                  pool;
  boost::optional<std::string> device;
  boost::optional<std::string> src_addr;
  boost::optional<std::string> provider;
  unsigned                     debug_level;
  unsigned                     base_core;
  size_t                       value_size;
} Options{};

namespace
{
template <typename T>
boost::optional<T> optional_option(const boost::program_options::variables_map &vm_, const std::string &key_)
{
  return 0 < vm_.count(key_) ? vm_[key_].as<T>() : boost::optional<T>();
}
}  // namespace

component::IKVStore_factory *fact;

using namespace component;

namespace
{
// The fixture for testing class Foo.
struct mcas_client_test : public ::testing::Test {
 protected:
  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  virtual void SetUp()
  {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  virtual void TearDown()
  {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

  // Objects declared here can be used by all tests in the test case
  static component::Itf_ref<component::IKVStore> _mcas;
};

component::Itf_ref<component::IKVStore> mcas_client_test::_mcas;

DECLARE_STATIC_COMPONENT_UUID(mcas_client, 0x2f666078, 0xcb8a, 0x4724, 0xa454, 0xd1, 0xd8, 0x8d, 0xe2, 0xdb, 0x87);
DECLARE_STATIC_COMPONENT_UUID(mcas_client_factory,
                              0xfac66078,
                              0xcb8a,
                              0x4724,
                              0xa454,
                              0xd1,
                              0xd8,
                              0x8d,
                              0xe2,
                              0xdb,
                              0x87);

void basic_test(IKVStore *kv, unsigned shard)
{
  int               rc;
  std::stringstream ss;
  ss << Options.pool << shard;
  std::string pool_name = ss.str();
  auto        pool      = kv->create_pool(pool_name, MB(8), 0, EXPECTED_OBJECTS);

  std::string value = "Hello! Value";  // 12 chars

  void *pv;
  for (unsigned i = 0; i < 10; i++) {
    std::string key = common::random_string(8);
    rc              = kv->put(pool, key.c_str(), value.c_str(), value.length());
    ASSERT_TRUE(rc == S_OK || rc == -2);

    pv            = nullptr;
    size_t pv_len = 0;
    rc            = kv->get(pool, key.c_str(), pv, pv_len);
    ASSERT_TRUE(rc == S_OK);
    ASSERT_TRUE(strncmp(static_cast<char *>(pv), value.c_str(), value.length()) == 0);
  }

  kv->close_pool(pool);
  ASSERT_TRUE(kv->delete_pool(pool_name) == S_OK);
  free(pv);
}

TEST_F(mcas_client_test, SessionControl)
{
  IKVStore_factory::map_create mc{{+IKVStore_factory::k_dest_addr, Options.addr},
                                  {+IKVStore_factory::k_dest_port, 0},
                                  {+IKVStore_factory::k_owner, "dwaddington"}};
  if (Options.src_addr) {
    mc.insert(IKVStore_factory::map_create::value_type(+IKVStore_factory::k_src_addr, *Options.src_addr));
  }
  if (Options.device) {
    mc.insert(IKVStore_factory::map_create::value_type(+IKVStore_factory::k_interface, *Options.device));
  }
  if (Options.provider) {
    mc.insert(IKVStore_factory::map_create::value_type(+IKVStore_factory::k_provider, *Options.provider));
  }

  /* create object instance through factory */
  component::IBase *comp = component::load_component("libcomponent-mcasclient.so", mcas_client_factory);

  ASSERT_TRUE(comp);
  auto fact = make_itf_ref(static_cast<IKVStore_factory *>(comp->query_interface(IKVStore_factory::iid())));

  {
    // auto mcas = make_itf_ref(fact->create(Options.debug_level, "dwaddington",
    // mc));
    auto mcas = make_itf_ref(fact->create(Options.debug_level, mc));
    ASSERT_TRUE(mcas.get());
  }

  {
    auto mcas2 = fact->create(Options.debug_level, mc);
    ASSERT_TRUE(mcas2);
    auto mcas3 = fact->create(Options.debug_level, mc);
    ASSERT_TRUE(mcas3);

    basic_test(mcas2, 0);
    basic_test(mcas3, 1);
  }
}

TEST_F(mcas_client_test, Instantiate)
{
  PMAJOR("Running Instantiate...");
  /* create object instance through factory */
  component::IBase *comp = component::load_component("libcomponent-mcasclient.so", mcas_client_factory);

  ASSERT_TRUE(comp);
  auto fact = make_itf_ref(static_cast<IKVStore_factory *>(comp->query_interface(IKVStore_factory::iid())));

  {
    IKVStore_factory::map_create mc{{+IKVStore_factory::k_dest_addr, Options.addr},
                                    {+IKVStore_factory::k_dest_port, 0},
                                    {+IKVStore_factory::k_owner, "dwaddington"}};
    if (Options.src_addr) {
      mc.insert(IKVStore_factory::map_create::value_type(+IKVStore_factory::k_src_addr, *Options.src_addr));
    }
    if (Options.device) {
      mc.insert(IKVStore_factory::map_create::value_type(+IKVStore_factory::k_interface, *Options.device));
    }
    if (Options.provider) {
      mc.insert(IKVStore_factory::map_create::value_type(+IKVStore_factory::k_provider, *Options.provider));
    }
    _mcas.reset(fact->create(Options.debug_level, mc));
  }
  ASSERT_TRUE(_mcas.get());
}

TEST_F(mcas_client_test, OpenCloseDelete)
{
  PMAJOR("Running OpenCloseDelete...");
  using namespace component;
  IKVStore::pool_t pool, pool2, pool3;

  const std::string poolname = Options.pool + "/OpenCloseDelete";
  ASSERT_TRUE((pool = _mcas->create_pool(poolname, GB(1))) != IKVStore::POOL_ERROR);
  ASSERT_FALSE(pool == IKVStore::POOL_ERROR);
  ASSERT_TRUE(_mcas->close_pool(pool) == S_OK);

  /* pool already exists */
  ASSERT_TRUE(_mcas->create_pool(poolname, GB(1), IKVStore::FLAGS_CREATE_ONLY) == IKVStore::POOL_ERROR);

  /* open two handles to the same pool + create with implicit open */
  ASSERT_TRUE((pool = _mcas->create_pool(poolname, GB(1))) != IKVStore::POOL_ERROR);
  ASSERT_TRUE((pool2 = _mcas->open_pool(poolname)) != IKVStore::POOL_ERROR);

  /* try delete open pool */
  ASSERT_TRUE(_mcas->delete_pool(poolname) == IKVStore::E_ALREADY_OPEN);

  /* open another */
  ASSERT_TRUE((pool3 = _mcas->open_pool(poolname)) != IKVStore::POOL_ERROR);

  /* close two */
  ASSERT_TRUE(_mcas->close_pool(pool) == S_OK);
  ASSERT_TRUE(_mcas->close_pool(pool2) == S_OK);

  /* try to delete open pool */
  ASSERT_TRUE(_mcas->delete_pool(poolname) == IKVStore::E_ALREADY_OPEN);
  ASSERT_TRUE(_mcas->close_pool(pool3) == S_OK);

  /* ok, now we can delete */
  ASSERT_TRUE(_mcas->delete_pool(poolname) == S_OK);
  PLOG("OpenCloseDelete Test OK");
}

#if 0
TEST_F(mcas_client_test, GetNotExist)
{
  PMAJOR("Running PutGet...");
  ASSERT_TRUE(_mcas);
  int rc;

  const std::string poolname = Options.pool + "/GetNotExist";

  auto pool = _mcas->open_pool(poolname, 0);

  if (pool == component::IKVStore::POOL_ERROR) {
    /* ok, try to create pool instead */
    pool = _mcas->create_pool(poolname, GB(1));
  }

  void *      pv;
  size_t      pv_len = 0;
  PINF("performing 'get' to retrieve non existing..");
  rc = _mcas->get(pool, "key0", pv, pv_len);
  PINF("get response:%d (%s) len:%lu", rc, (char *) pv, pv_len);
  //  ASSERT_TRUE(rc == S_OK);
  _mcas->close_pool(pool);
  // ASSERT_TRUE(strncmp((char *) pv, value.c_str(), value.length()) == 0);

  _mcas->delete_pool(poolname);
  free(pv);
  PLOG("GetNotExist OK!");
}
#endif
TEST_F(mcas_client_test, BasicPutAndGet)
{
  PMAJOR("Running BasicPutGet...");
  ASSERT_TRUE(_mcas.get());
  int rc;

  auto pool = _mcas->create_pool(Options.pool, MB(8));

  std::string value = "Hello! Value";  // 12 chars
  void *      pv;
  for (unsigned i = 0; i < 10; i++) {
    rc = _mcas->put(pool, "key0", value.c_str(), value.length());
    PINF("put response:%d", rc);
    ASSERT_TRUE(rc == S_OK || rc == -2);

    pv            = nullptr;
    size_t pv_len = 0;
    PINF("performing 'get' to retrieve what was put..");
    rc = _mcas->get(pool, "key0", pv, pv_len);
    PINF("get response:%d (%s) len:%lu", rc, static_cast<char *>(pv), pv_len);
    ASSERT_TRUE(rc == S_OK);
    ASSERT_TRUE(strncmp(static_cast<char *>(pv), value.c_str(), value.length()) == 0);
  }

  _mcas->delete_pool(Options.pool);
  free(pv);
  PLOG("BasicPutAndGet OK!");
}

#ifdef TEST_SCALE_IOPS

struct record_t {
  std::string key;
  char        value[32];
};

std::mutex    _iops_lock;
static double _iops = 0.0;

class IOPS_task : public common::Tasklet {
 public:
  static constexpr unsigned long ITERATIONS = 1000000;
  static constexpr unsigned long VALUE_SIZE = 32;
  static constexpr unsigned long KEY_SIZE   = 8;

  IOPS_task(unsigned arg) {}

  virtual void initialize(unsigned core) override
  {
    _store.reset(fact->create(Options.debug_level, "dwaddington", Options.addr, Options.device));

    char poolname[64];
    sprintf(poolname, "/dev/dax0.%u", core);

    _pool = _store->create_pool(poolname, GiB(1));

    _data = (record_t *) malloc(sizeof(record_t) * ITERATIONS);
    ASSERT_FALSE(_data == nullptr);

    PLOG("Setting up data worker: %u", core);

    /* set up data */
    for (unsigned long i = 0; i < ITERATIONS; i++) {
      auto val     = common::random_string(VALUE_SIZE);
      _data[i].key = common::random_string(KEY_SIZE);
      memcpy(_data[i].value, val.c_str(), VALUE_SIZE);
    }

    _ready_flag = true;
    _start_time = std::chrono::high_resolution_clock::now();
  }

  virtual bool do_work(unsigned core) override
  {
    if (_iterations == 0) PLOG("Starting worker: %u", core);

    _iterations++;
    status_t rc = _store->put(_pool, _data[_iterations].key, _data[_iterations].value, VALUE_SIZE);

    if (rc != S_OK) throw General_exception("put operation failed:rc=%d", rc);

    assert(rc == S_OK);

    if (_iterations > ITERATIONS) {
      _end_time = std::chrono::high_resolution_clock::now();
      PLOG("Worker: %u complete", core);
      return false;
    }
    return true;
  }

  virtual void cleanup(unsigned core) override
  {
    PLOG("Cleanup %u", core);
    auto secs = std::chrono::duration<double>(_end_time - _start_time).count();
    _iops_lock.lock();
    auto iops = double(ITERATIONS) / secs;
    PLOG("%f iops (core=%u)", iops, core);
    _iops += iops;
    _iops_lock.unlock();
    _store->close_pool(_pool);
    _store->reset(nullptr);
  }

  virtual bool ready() override { return _ready_flag; }

 private:
  std::chrono::high_resolution_clock::time_point _start_time, _end_time;
  bool                                           _ready_flag = false;
  unsigned long                                  _iterations = 0;
  component::Itf_ref<component::IKVStore>        _store;
  record_t *                                     _data;
  component::IKVStore::pool_t                    _pool;
};

TEST_F(mcas_client_test, PerfScaleIops)
{
  PMAJOR("Running PerfScaleIops...");
  static constexpr unsigned NUM_CORES = 8;
  cpu_mask_t                mask;
  for (unsigned c = 0; c < NUM_CORES; c++) mask.add_core(c + Options.base_core);
  {
    common::Per_core_tasking<IOPS_task, unsigned> t(mask, 11911);
    t.wait_for_all();
  }
  PMAJOR("Aggregate IOPS: %2g", _iops);
}
#endif

#ifdef TEST_PERF_SMALL_PUT
TEST_F(mcas_client_test, PerfSmallPut)
{
  PMAJOR("Running SmallPut...");
  ASSERT_TRUE(_mcas);
  int rc;

  const std::string poolname = Options.pool + "/PerfSmallPut";
  auto              pool     = _mcas->create_pool(poolname, GB(4));

  static constexpr unsigned long ITERATIONS = 1000000;
  static constexpr unsigned long VALUE_SIZE = 32;
  static constexpr unsigned long KEY_SIZE   = 8;

  struct record_t {
    std::string key;
    char        value[VALUE_SIZE];
  };

  record_t *data = (record_t *) malloc(sizeof(record_t) * ITERATIONS);
  ASSERT_FALSE(data == nullptr);

  /* set up data */
  for (unsigned long i = 0; i < ITERATIONS; i++) {
    auto val    = common::random_string(VALUE_SIZE);
    data[i].key = common::random_string(KEY_SIZE);
    memcpy(data[i].value, val.c_str(), VALUE_SIZE);
  }

  auto start = std::chrono::high_resolution_clock::now();

  for (unsigned long i = 0; i < ITERATIONS; i++) {
    rc = _mcas->put(pool, data[i].key, data[i].value, VALUE_SIZE);
    ASSERT_TRUE(rc == S_OK || rc == -2);
  }

  auto end  = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration<double>(end - start).count();
  PMAJOR("PerfSmallPut Ops/Sec: %lu", static_cast<unsigned long>(ITERATIONS / secs));

  ::free(data);

  _mcas->close_pool(pool);
  _mcas->delete_pool(poolname);
}
#endif

#ifdef TEST_PERF_SMALL_PUT_DIRECT
TEST_F(mcas_client_test, PerfSmallPutDirect)
{
  PMAJOR("Running SmallPutDirect...");
  int rc;

  /* open or create pool */
  component::IKVStore::pool_t pool = _mcas->open_pool(std::string("/mnt/pmem0/mcas") + Options.pool, 0);

  if (pool == component::IKVStore::POOL_ERROR) {
    /* ok, try to create pool instead */
    pool = _mcas->create_pool(std::string("/mnt/pmem0/mcas") + Options.pool, GB(1));
  }

  static constexpr unsigned long ITERATIONS = 1000000;
  static constexpr unsigned long VALUE_SIZE = 32;
  static constexpr unsigned long KEY_SIZE   = 8;

  struct record_t {
    std::string key;
    char        value[VALUE_SIZE];
  };

  size_t    data_size = sizeof(record_t) * ITERATIONS;
  record_t *data      = (record_t *) aligned_alloc(MiB(2), data_size);
  ASSERT_NE(nullptr, data);
  madvise(data, data_size, MADV_HUGEPAGE);

  ASSERT_FALSE(data == nullptr);
  auto handle = _mcas->register_direct_memory(data, data_size); /* register whole region */

  /* set up data */
  for (unsigned long i = 0; i < ITERATIONS; i++) {
    auto val    = common::random_string(VALUE_SIZE);
    data[i].key = common::random_string(KEY_SIZE);
    memcpy(data[i].value, val.c_str(), VALUE_SIZE);
  }

  auto start = std::chrono::high_resolution_clock::now();

  for (unsigned long i = 0; i < ITERATIONS; i++) {
    /* different value each iteration; tests memory region registration */
    rc = _mcas->put_direct(pool, data[i].key, data[i].value, VALUE_SIZE,
                           handle); /* pass handle from memory registration */
    ASSERT_TRUE(rc == S_OK || rc == -6);
  }

  auto end  = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration<double>(end - start).count();
  PMAJOR("PerfSmallPutDirect Ops/Sec: %lu", static_cast<unsigned long>(ITERATIONS / secs));

  _mcas->unregister_direct_memory(handle);

  _mcas->close_pool(pool);
  _mcas->delete_pool("/mnt/pmem0/mcas", Options.pool);
}
#endif

#ifdef TEST_PERF_LARGE_PUT_DIRECT
TEST_F(mcas_client_test, PerfLargePutDirect)
{
  PMAJOR("Running LargePutDirect...");

  int rc;
  ASSERT_TRUE(_mcas.get());

  const std::string poolname = Options.pool + "/PerfLargePutDirect";

  /* open or create pool */
  auto pool = _mcas->create_pool(poolname, GB(8), 0, EXPECTED_OBJECTS);

  PLOG("Test pool created OK.");

  static constexpr unsigned long PER_ITERATION = 1;
  static constexpr unsigned long ITERATIONS    = LARGE_ITERATIONS;
  static constexpr unsigned long VALUE_SIZE    = LARGE_VALUE_SIZE;
  static constexpr unsigned long KEY_SIZE      = 8;

  struct record_t {
    char key[KEY_SIZE];
    char value[VALUE_SIZE];
  };

  PLOG("Allocating buffer with test data ...");
  size_t    data_size = sizeof(record_t) * PER_ITERATION;
  record_t *data      = static_cast<record_t *>(aligned_alloc(MiB(2), data_size));
  ASSERT_NE(nullptr, data);
  madvise(data, data_size, MADV_HUGEPAGE | MADV_DONTFORK);

  ASSERT_FALSE(data == nullptr);
  auto handle = _mcas->register_direct_memory(data, data_size); /* register whole region */

  PLOG("Filling data...");
  /* set up data */
  for (unsigned long i = 0; i < PER_ITERATION; i++) {
    auto l = common::random_string(KEY_SIZE);
    memcpy(data[i].key, l.c_str(), KEY_SIZE);
    //    memcpy(data[i].value, val.c_str(), VALUE_SIZE);
  }
  PLOG("Starting put_direct operation...");

  auto start = std::chrono::high_resolution_clock::now();

  for (unsigned long i = 0; i < (ITERATIONS * PER_ITERATION); i++) {
    /* different value each iteration; tests memory region registration */
    rc = _mcas->put_direct(pool, data[i % PER_ITERATION].key, data[i % PER_ITERATION].value, VALUE_SIZE,
                           handle); /* pass handle from memory registration */
    ASSERT_TRUE(rc == S_OK);
  }

  auto end  = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration<double>(end - start).count();
  PINF("PerfLargePutDirect Throughput: %.2f MiB/sec (%.2f IOPS)",
       ((PER_ITERATION * ITERATIONS * VALUE_SIZE) / secs) / (1024.0 * 1024), (PER_ITERATION * ITERATIONS) / secs);

  _mcas->close_pool(pool);
  _mcas->delete_pool(poolname);
  _mcas->unregister_direct_memory(handle);
  PLOG("Unregistered memory");
}
#endif

#ifdef TEST_PERF_LARGE_GET_DIRECT
TEST_F(mcas_client_test, PerfLargeGetDirect)
{
  int rc;

  PMAJOR("Running LargeGetDirect...");
  const std::string poolname = Options.pool + "/PerfLargeGetDirect";
  auto              pool     = _mcas->create_pool(poolname, GB(8), 0, EXPECTED_OBJECTS);
  ASSERT_TRUE(pool != IKVStore::POOL_ERROR);

  static constexpr unsigned long PER_ITERATION = 1;
  static constexpr unsigned long ITERATIONS    = LARGE_ITERATIONS;
  static constexpr unsigned long VALUE_SIZE    = LARGE_VALUE_SIZE;
  static constexpr unsigned long KEY_SIZE      = 8;

  std::vector<std::string *> keys;

  PLOG("Allocating buffer with test data ...");
  size_t data_size = VALUE_SIZE * PER_ITERATION;
  char * data      = static_cast<char *>(aligned_alloc(MiB(2), data_size));
  ASSERT_NE(nullptr, data);
  madvise(data, data_size, MADV_HUGEPAGE | MADV_DONTFORK);
  memset(data, 0, data_size);

  ASSERT_FALSE(data == nullptr);
  auto handle = _mcas->register_direct_memory(data, data_size); /* register whole region */

  PLOG("LargeGetDirect: Filling data up front...");
  /* set up data */
  //  char * p = data;
  for (unsigned long i = 0; i < PER_ITERATION; i++) {
    // auto val    = common::random_string(VALUE_SIZE);
    // memcpy(p, val.c_str(), VALUE_SIZE);
    // p+=val.length();
    auto s = new std::string;
    *s     = common::random_string(KEY_SIZE);
    keys.push_back(s);
  }
  PLOG("Using put_direct to fill for get_direct operation...");

  auto start = std::chrono::high_resolution_clock::now();

  for (unsigned long i = 0; i < (ITERATIONS * PER_ITERATION); i++) {
    //    PLOG("Putting key(%s)", keys[i % PER_ITERATION]->c_str());
    /* different value each iteration; tests memory region registration */
    rc = _mcas->put_direct(pool, *keys[i % PER_ITERATION], &data[i % PER_ITERATION], VALUE_SIZE,
                           handle); /* pass handle from memory registration */
    ASSERT_TRUE(rc == S_OK || rc == -6);
  }

  auto end  = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration<double>(end - start).count();
  PINF("LargePutDirect Throughput: %.2f MiB/sec  (%.2f IOPS)",
       ((PER_ITERATION * ITERATIONS * VALUE_SIZE) / secs) / (1024.0 * 1024), ((PER_ITERATION * ITERATIONS) / secs));

  for (unsigned r = 0; r < 2; r++) {
    PINF("LargeGetDirect: Starting .. get_direct phase (%u)", r);

    start = std::chrono::high_resolution_clock::now();

    /* perform get phase */
    for (unsigned long i = 0; i < (ITERATIONS * PER_ITERATION); i++) {
      size_t value_len = VALUE_SIZE;
      //    PLOG("get_direct (key=%s)", keys[i % PER_ITERATION]->c_str());
      rc = _mcas->get_direct(pool, *keys[i % PER_ITERATION], &data[i % PER_ITERATION], value_len, handle);
      ASSERT_TRUE(rc == S_OK);  // || rc == -6);
    }

    end  = std::chrono::high_resolution_clock::now();
    secs = std::chrono::duration<double>(end - start).count();
    PINF("PerfLargeGetDirect Throughput: %.2f MiB/sec (%.2f IOPS)",
         ((PER_ITERATION * ITERATIONS * VALUE_SIZE) / secs) / (1024.0 * 1024), ((PER_ITERATION * ITERATIONS) / secs));

    PINF("LargeGetDirect: Starting phase (2) .. get_direct phase");
  }

  _mcas->close_pool(pool);
  _mcas->delete_pool(poolname);
  _mcas->unregister_direct_memory(handle);
}
#endif

#ifdef TEST_PERF_SMALL_GET_DIRECT
TEST_F(mcas_client_test, PerfSmallGetDirect)
{
  PMAJOR("Running SmallGetDirect...");
  int rc;

  const std::string poolname = Options.pool + "/PerfSmallGetDirect";
  auto              pool     = _mcas->create_pool(poolname, GB(8));

  static constexpr unsigned long PER_ITERATION = 8;
  static constexpr unsigned long ITERATIONS    = 100000;
  static constexpr unsigned long VALUE_SIZE    = 32;
  static constexpr unsigned long KEY_SIZE      = 8;

  struct record_t {
    std::string key;
    char        value[VALUE_SIZE];
  };

  PLOG("Allocating buffer with test data ...");
  size_t    data_size = sizeof(record_t) * PER_ITERATION;
  record_t *data      = (record_t *) aligned_alloc(MiB(2), data_size);
  ASSERT_NE(nullptr, data);
  madvise(data, data_size, MADV_HUGEPAGE);

  ASSERT_FALSE(data == nullptr);
  auto handle = _mcas->register_direct_memory(data, data_size); /* register whole region */

  PLOG("Filling data...");
  /* set up data */
  for (unsigned long i = 0; i < PER_ITERATION; i++) {
    auto val    = common::random_string(VALUE_SIZE);
    data[i].key = common::random_string(KEY_SIZE);
    memcpy(data[i].value, val.c_str(), VALUE_SIZE);
  }
  PLOG("Starting PUT operation...");

  auto start = std::chrono::high_resolution_clock::now();

  for (unsigned long i = 0; i < (ITERATIONS * PER_ITERATION); i++) {
    /* different value each iteration; tests memory region registration */
    rc = _mcas->put_direct(pool, data[i % PER_ITERATION].key, data[i % PER_ITERATION].value, VALUE_SIZE,
                           handle); /* pass handle from memory registration */
    ASSERT_TRUE(rc == S_OK || rc == -2);
  }

  auto end  = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration<double>(end - start).count();
  PINF("PerfSmallGet Prep-Throughput: %.2f MiB/sec",
       ((PER_ITERATION * ITERATIONS * VALUE_SIZE) / secs) / (1024.0 * 1024));

  PINF("Starting .. get phase");

  start = std::chrono::high_resolution_clock::now();

  /* perform get phase */
  for (unsigned long i = 0; i < (ITERATIONS * PER_ITERATION); i++) {
    size_t value_len = VALUE_SIZE;
    rc = _mcas->get_direct(pool, data[i % PER_ITERATION].key, data[i % PER_ITERATION].value, value_len, handle);
    ASSERT_TRUE(rc == S_OK);  // || rc == -6);
  }

  end  = std::chrono::high_resolution_clock::now();
  secs = std::chrono::duration<double>(end - start).count();
  PINF("PerfSmallGet Throughput: %.2f MiB/sec", ((PER_ITERATION * ITERATIONS * VALUE_SIZE) / secs) / (1024.0 * 1024));

  _mcas->close_pool(pool);
  _mcas->delete_pool(poolname);
  _mcas->unregister_direct_memory(handle);
}
#endif

TEST_F(mcas_client_test, Release)
{
  PLOG("Releasing instance...");

  /* release instance */
  _mcas.reset(nullptr);
}

}  // namespace

int main(int argc, char **argv)
{
  //#  option_addr = (argc > 1) ? argv[1] : "10.0.0.41:11911";

  try {
    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()("help", "Show help")("debug", po::value<unsigned>()->default_value(0), "Debug level 0-3")(
        "server-addr", po::value<std::string>()->default_value("10.0.0.21:11911:verbs"),
        "Server address IP:PORT[:PROVIDER]")("device", po::value<std::string>(), "Network device (e.g., mlx5_0)")(
        "source-addr", po::value<std::string>(), "iLocal network address, e.g. 1.0.0.20")(
        "pool", po::value<std::string>()->default_value("myPool"), "Pool name")(
        "value_size", po::value<std::size_t>()->default_value(0), "Value size")(
        "base", po::value<unsigned>()->default_value(0), "Base core.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help") > 0) {
      std::cout << desc;
      return -1;
    }

    Options.addr        = vm["server-addr"].as<std::string>();
    Options.debug_level = vm["debug"].as<unsigned>();
    Options.pool        = vm["pool"].as<std::string>();
    Options.device      = optional_option<std::string>(vm, "device");
    Options.src_addr    = optional_option<std::string>(vm, "source-addr");
    Options.provider    = optional_option<std::string>(vm, "provider");
    Options.base_core   = vm["base"].as<unsigned>();
    Options.value_size  = vm["value_size"].as<std::size_t>();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  }
  catch (...) {
    PLOG("bad command line option configuration");
    return -1;
  }

  return 0;
}
