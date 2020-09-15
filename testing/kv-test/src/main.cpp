#include <api/components.h>
#include <api/mcas_itf.h>
#include <common/cycles.h>
#include <common/str_utils.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <boost/program_options.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib> /* rand */
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

#define ASSERT_OK(X) ASSERT_EQ(S_OK, X)

struct Options {
  unsigned    debug_level;
  unsigned patience;
  boost::optional<std::string> device;
  boost::optional<std::string> src_addr;
  std::string server;
  unsigned    port;
} g_options;

class KV_test : public ::testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

component::Itf_ref<component::IMCAS> mcas;
component::Itf_ref<component::IMCAS> init(const std::string &server_hostname, int port);

component::Itf_ref<component::IMCAS> init(const std::string &server_hostname, int port)
{
  using namespace component;

  IBase *comp = component::load_component("libcomponent-mcasclient.so", mcas_client_factory);

  auto fact = make_itf_ref(static_cast<IMCAS_factory *>(comp->query_interface(IMCAS_factory::iid())));
  if (!fact) throw Logic_exception("unable to create MCAS factory");

  std::stringstream url;
  url << g_options.server << ":" << g_options.port;

  auto mcas = make_itf_ref(fact->mcas_create(g_options.debug_level, g_options.patience, "None", g_options.device, g_options.src_addr, url.str()));

  if (!mcas) throw Logic_exception("unable to create MCAS client instance");

  return mcas;
}

/**
 * SECTION: Tests
 *
 */
TEST_F(KV_test, BasicPoolOperations)
{
  using namespace component;

  const std::string poolname = "pool0";

  auto pool = mcas->create_pool(poolname, MB(1), /* size */
                                0,               /* flags */
                                100);            /* obj count */

  ASSERT_FALSE(pool == IKVStore::POOL_ERROR);

  ASSERT_OK(mcas->close_pool(pool));

  PLOG("Reopen pool (from name) after close");
  pool = mcas->open_pool(poolname);
  ASSERT_FALSE(pool == IKVStore::POOL_ERROR);

  ASSERT_OK(mcas->close_pool(pool));
  PLOG("Deleting pool (from name) after close");
  ASSERT_OK(mcas->delete_pool(poolname));

  PLOG("Re-creating pool");
  pool = mcas->create_pool(poolname, MB(2),             /* size */
                           IMCAS::ADO_FLAG_CREATE_ONLY, /* flags */
                           100);                        /* obj count */
  ASSERT_FALSE(pool == IMCAS::POOL_ERROR);

  ASSERT_TRUE(mcas->create_pool(poolname, MB(2),             /* size */
                                IMCAS::ADO_FLAG_CREATE_ONLY, /* flags */
                                100) == IMCAS::POOL_ERROR);
  ASSERT_OK(mcas->close_pool(pool));
  ASSERT_OK(mcas->delete_pool(poolname));
}

TEST_F(KV_test, OpenCloseDeletePool)
{
  using namespace component;

  const std::string poolname = "pool2";

  auto pool = mcas->create_pool(poolname, MB(64), /* size */
                                0,                /* flags */
                                100);             /* obj count */

  ASSERT_FALSE(pool == IKVStore::POOL_ERROR);

  ASSERT_OK(mcas->close_pool(pool));

  /* re-open pool */
  pool = mcas->open_pool(poolname);

  ASSERT_FALSE(pool == IKVStore::POOL_ERROR);

  ASSERT_OK(mcas->delete_pool(pool));
}

/* test for dawn-311 */
TEST_F(KV_test, DeletePoolOperations)
{
  using namespace component;

  const std::string poolname = "pool1";

  auto pool = mcas->create_pool(poolname, MB(64), /* size */
                                0,                /* flags */
                                100);             /* obj count */

  ASSERT_FALSE(pool == IKVStore::POOL_ERROR);

  ASSERT_OK(mcas->delete_pool(pool));

  ASSERT_TRUE(mcas->open_pool(poolname) == IMCAS::POOL_ERROR);
}

TEST_F(KV_test, BasicPutGetOperations)
{
  using namespace component;

  const std::string poolname = "BasicPutGetOperations";

  auto pool = mcas->create_pool(poolname, MB(32), /* size */
                                0,                /* flags */
                                100);             /* obj count */

  ASSERT_FALSE(pool == IKVStore::POOL_ERROR);

  /* delete pool fails on mapstore when there is something in it. Bug # DAWN-287 */
  std::string key0   = "key0";
  std::string key1   = "key1";
  std::string value0 = "this_is_value_0";
  std::string value1 = "this_is_value_1_and_its_longer";
  ASSERT_OK(mcas->put(pool, key0, value0, 0));

  std::string out_value;
  ASSERT_OK(mcas->get(pool, key0, out_value));
  ASSERT_TRUE(value0 == out_value);

  ASSERT_OK(mcas->put(pool, key0, value1, 0));
  ASSERT_OK(mcas->get(pool, key0, out_value));
  PLOG("value1(%s) out_value(%s)", value1.c_str(), out_value.c_str());
  ASSERT_TRUE(value1 == out_value);

  /* try overwrite with DONT STOMP flag */
  ASSERT_TRUE(mcas->put(pool, key0, value1, IKVStore::FLAGS_DONT_STOMP) == IKVStore::E_KEY_EXISTS);

  ASSERT_OK(mcas->erase(pool, key0));

  /* here inout_value_len is zero, therefore on-demand creation is disabled */
  ASSERT_TRUE(mcas->get(pool, key0, out_value) == IKVStore::E_KEY_NOT_FOUND);

  ASSERT_OK(mcas->close_pool(pool));
  ASSERT_OK(mcas->delete_pool(poolname));
}

class handle
{
public:
  virtual component::IMCAS::memory_handle_t get() const = 0;
  virtual ~handle() {}
};

class handle_real
  : public handle
{
  component::Itf_ref<component::IMCAS> *_m;
  component::IMCAS::memory_handle_t _h;
public:
  handle_real(component::Itf_ref<component::IMCAS> &mcas_, void *addr_, size_t len_)
    : _m(&mcas_)
    , _h((*_m)->register_direct_memory(addr_, len_))
  {}
  ~handle_real()
  {
    EXPECT_EQ(S_OK, (*_m)->unregister_direct_memory(_h));
  }
  component::IMCAS::memory_handle_t get() const override { return _h; }
};

class handle_fake
  : public handle
{
public:
  handle_fake()
  {}
  component::IMCAS::memory_handle_t get() const override { return component::IMCAS::MEMORY_HANDLE_NONE; }
};

auto make_handle_real(component::Itf_ref<component::IMCAS> &mcas_, void *addr_, size_t len_) -> std::unique_ptr<handle>
{
  return std::make_unique<handle_real>(mcas_, addr_, len_);
}

auto make_handle_fake(component::Itf_ref<component::IMCAS> &, void *, size_t) -> std::unique_ptr<handle>
{
  return std::make_unique<handle_fake>();
}

using make_handle_t = auto (*)(component::Itf_ref<component::IMCAS> &, void *, size_t) -> std::unique_ptr<handle>;

/* Get the pool size from MCAS, up to size max. Uses max bytes of memory */
std::size_t get_pool_size(component::Itf_ref<component::IMCAS> &mcas_, make_handle_t mh_, component::IMCAS::pool_t pool_, std::size_t max_)
{
  auto len = max_;
  auto buffer = aligned_alloc(KB(4), max_);
  EXPECT_NE(nullptr, buffer);
  auto mem = mh_(mcas, buffer, max_);
  EXPECT_EQ(S_OK, mcas->get_direct_offset(pool_, 0, len, buffer, mem->get()));
  free(buffer);
  return len;
}

void wipe_and_restore(component::Itf_ref<component::IMCAS> &mcas_, make_handle_t mh_, component::IMCAS::pool_t pool_, std::size_t max_)
{
  auto len = get_pool_size(mcas_, mh_, pool_, max_);
  auto save_buffer = static_cast<char *>(aligned_alloc(KiB(4), len));
  EXPECT_NE(nullptr, save_buffer);
  {
    auto mem = mh_(mcas_, save_buffer, len);
    auto save_len = len;
    EXPECT_EQ(S_OK, mcas_->get_direct_offset(pool_, 0, save_len, save_buffer, mem->get()));
    EXPECT_EQ(len, save_len);
  }

  /* fill the entire pool with xes, writing random sizes until all bytes are filled */
  {
    std::size_t offset = 0;
    for ( ; offset != len ; )
    {
      std::size_t xes_len = ::rand() % 10000000;
      auto xes = static_cast<char *>(aligned_alloc(KiB(4), xes_len));
      EXPECT_NE(nullptr, xes);
      std::memset(xes, 'x', xes_len);
      auto mem = mh_(mcas_, xes, xes_len);
      EXPECT_EQ(S_OK, mcas_->put_direct_offset(pool_, offset, xes_len, xes, mem->get()));
      EXPECT_LE(offset + xes_len, len);
      offset += xes_len;
      free(xes);
    }
  }

  /* read the entire pool using random sizes, expecting all xes */
  {
    std::size_t offset = 0;
    for ( ; offset != len ; )
    {
      std::size_t xes_len = ::rand() % 10000000;
      auto xes = static_cast<char *>(aligned_alloc(KiB(4), xes_len));
      EXPECT_NE(nullptr, xes);
      std::memset(xes, 'y', xes_len);
      auto mem = mh_(mcas_, xes, xes_len);
      EXPECT_EQ(S_OK, mcas_->get_direct_offset(pool_, offset, xes_len, xes, mem->get()));
      EXPECT_LE(offset + xes_len, len);
      EXPECT_EQ(xes + xes_len, std::find_if_not(xes, xes + xes_len, [] (char c) { return c == 'x'; }));
      offset += xes_len;
      free(xes);
    }
  }

  /* restore original value */
  {
    auto mem = mh_(mcas_, save_buffer, len);
    auto restore_len = len;
    EXPECT_EQ(S_OK, mcas_->put_direct_offset(pool_, 0, restore_len, save_buffer, mem->get()));
    EXPECT_EQ(len, restore_len);
  }
  free(save_buffer);
}

void put_direct(component::Itf_ref<component::IMCAS> &mcas, make_handle_t mh_)
{
  using namespace component;

  const std::string poolname = "PutDirect";

  auto pool = mcas->create_pool(poolname, GB(1), /* size */
                                0,               /* flags */
                                100);            /* obj count */

  ASSERT_FALSE(pool == IKVStore::POOL_ERROR);

  //  ASSERT_OK(mcas->put(pool, key0, value0, 0));

  size_t                 user_buffer_len = MB(128);
  void *                 user_buffer     = aligned_alloc(KB(4), user_buffer_len);
  auto mem = mh_(mcas, user_buffer, user_buffer_len);

  ASSERT_OK(mcas->put_direct(pool, "someLargeObject", user_buffer, user_buffer_len, mem->get()));
  ASSERT_OK(mcas->put_direct(pool, "anotherLargeObject", user_buffer, user_buffer_len, mem->get()));

  std::vector<uint64_t> attrs;
  ASSERT_OK(mcas->get_attribute(pool, IMCAS::Attribute::COUNT, attrs));
  ASSERT_TRUE(attrs[0] == 2); /* there should be only one object */

  size_t                 user_buffer2_len = MB(128);
  void *                 user_buffer2     = aligned_alloc(KB(4), user_buffer_len);
  auto mem2 = mh_(mcas, user_buffer2, user_buffer2_len);

  ASSERT_OK(mcas->get_direct(pool, "someLargeObject", user_buffer2, user_buffer2_len, mem2->get()));

  ASSERT_TRUE(memcmp(user_buffer, user_buffer2, user_buffer_len) == 0); /* integrity check */

  ASSERT_OK(mcas->close_pool(pool));
  ASSERT_OK(mcas->delete_pool(poolname));
  free(user_buffer);
  free(user_buffer2);
}

TEST_F(KV_test, PutDirectRegistered)
{
  put_direct(mcas, make_handle_real);
}

TEST_F(KV_test, PutDirectUnregistered)
{
  put_direct(mcas, make_handle_fake);
}

void get_put_direct_offset(component::Itf_ref<component::IMCAS> &mcas, make_handle_t mh_)
{
  const std::string poolname = "GetPutDirectOffset";

  auto pool = mcas->create_pool(poolname, GB(1), /* size */
                                0,               /* flags */
                                100);            /* obj count */

  ASSERT_NE(+component::IKVStore::POOL_ERROR, pool);

  auto pool_size = get_pool_size(mcas, mh_, pool, GiB(2));
  PLOG("Pool size 0x%zx", pool_size);

  //  ASSERT_OK(mcas->put(pool, key0, value0, 0));

  /* A prefix of the data, for us to read and modify */
  const char *data_prefix_someLargeObject = "dataSomeLargeObject";
  auto data_prefix_size_someLargeObject = std::strlen(data_prefix_someLargeObject);
  const char *const key_someLargeObject = "someLargeObject";
  const auto key_size_someLargeObject = std::strlen(key_someLargeObject);
  {
    size_t               user_buffer_len = MB(128);
    void *               user_buffer     = aligned_alloc(KB(4), user_buffer_len);
    ASSERT_NE(nullptr, user_buffer);
    std::memcpy(user_buffer, data_prefix_someLargeObject, data_prefix_size_someLargeObject);
    auto mem = mh_(mcas, user_buffer, user_buffer_len);
    ASSERT_OK(mcas->put_direct(pool, key_someLargeObject, user_buffer, user_buffer_len, mem->get()));
    free(user_buffer);
  }

  /* somewhere in the region is the data value, which begins "dataSomeLargeObject"
   * Use get_locate_offset to try to find it.
   */

  std::size_t scan_buffer_len = MiB(128);
  char *scan_buffer = static_cast<char *>(aligned_alloc(KB(4), scan_buffer_len));
  ASSERT_NE(nullptr, scan_buffer);
  std::size_t offset = 0;

  /* write x's to the entire pool, then restore it.requires 3 * pools_size bytes */
  wipe_and_restore(mcas, mh_, pool, GiB(1));

  auto mem = mh_(mcas, scan_buffer, scan_buffer_len);
  auto rc = mcas->get_direct_offset(pool, offset, scan_buffer_len, scan_buffer, mem->get());
  while ( rc == S_OK )
  {
    PLOG("offset 0x%zx", offset);
    auto it = std::search(scan_buffer, scan_buffer + scan_buffer_len, data_prefix_someLargeObject, data_prefix_someLargeObject + data_prefix_size_someLargeObject);
    if ( it != scan_buffer + scan_buffer_len )
    {
      offset += it - scan_buffer;
      break;
    }
    offset += MiB(127);
    rc = mcas->get_direct_offset(pool, offset, scan_buffer_len, scan_buffer, mem->get());
  }

  ASSERT_EQ(S_OK, rc);

  PLOG("final offset 0x%zx", offset);

  {
    /* Use get_direct_offset to read bytes from the key */
    auto small_delta = 2;
    std::size_t length = data_prefix_size_someLargeObject - small_delta;
    ASSERT_OK(mcas->get_direct_offset(pool, offset+small_delta, length, scan_buffer, mem->get()));
    EXPECT_EQ(data_prefix_size_someLargeObject - small_delta, length);
    ASSERT_TRUE(std::equal(scan_buffer, scan_buffer + length, data_prefix_someLargeObject + small_delta));
  }

  {
    /* Use put_direct_offset to alter "dataSomeLargeObject" to "dataWhatLargeObject" */
    /* Since the registered_memory parameter is not requored, let MCAS fill it in, */
    auto some_delta = 4;
    std::size_t some_size = 4;
    char what[] = "What"; /* Must be writeabe. fabric_its API does not provide a way to reigister emmory fo rread obnly */
    ASSERT_OK(mcas->put_direct_offset(pool, offset+some_delta, some_size, what));

    /* Read back the data through the key. It should have been surreptitiously changed by put_direct_offset */
    std::size_t data_len = scan_buffer_len;
    ASSERT_OK(mcas->get_direct(pool, key_someLargeObject, scan_buffer, data_len, mem->get()));
    ASSERT_EQ(scan_buffer_len, data_len);
    ASSERT_TRUE(std::equal(scan_buffer, scan_buffer + data_prefix_size_someLargeObject, "dataWhatLargeObject"));
  }

  ASSERT_OK(mcas->close_pool(pool));
  ASSERT_OK(mcas->delete_pool(poolname));
  free(scan_buffer);
}

TEST_F(KV_test, GetDirectOffsetRegistered)
{
  get_put_direct_offset(mcas, make_handle_real);
}

TEST_F(KV_test, GetDirectOffsetUnregistered)
{
  get_put_direct_offset(mcas, make_handle_fake);
}

void async_get_direct_offset(component::Itf_ref<component::IMCAS> &mcas, make_handle_t mh_)
{
  const std::string poolname = "AsyncGetDirectOffset";

  auto pool = mcas->create_pool(poolname, GB(1), /* size */
                                0,               /* flags */
                                100);            /* obj count */

  ASSERT_NE(+component::IKVStore::POOL_ERROR, pool);

  //  ASSERT_OK(mcas->put(pool, key0, value0, 0));

  const char *key_someLargeObject = "someLargeObject";
  auto key_size_someLargeObject = std::strlen(key_someLargeObject);
  {
    size_t                 user_buffer_len = MB(128);
    void *                 user_buffer     = aligned_alloc(KB(4), user_buffer_len);
    ASSERT_NE(nullptr, user_buffer);

    auto mem = mh_(mcas, user_buffer, user_buffer_len);

    ASSERT_OK(mcas->put_direct(pool, key_someLargeObject, user_buffer, user_buffer_len, mem->get()));
    free(user_buffer);
  }

  /* somewhere in the region is the text "someLargeObject", not necessarily including a trailing NUL.
   * Use get_locate_offset to try to find it.
   */

  size_t scan_buffer_len = MiB(128);
  char *scan_buffer = static_cast<char *>(aligned_alloc(KB(4), scan_buffer_len));
  ASSERT_NE(nullptr, scan_buffer);
  std::size_t offset = 0;
  auto mem = mh_(mcas, scan_buffer, scan_buffer_len);
  auto rc = mcas->get_direct_offset(pool, offset, scan_buffer_len, scan_buffer, mem->get());
  while ( rc == S_OK && scan_buffer_len == MiB(128) )
  {
    PLOG("offset %zu", offset);
    auto it = std::search(scan_buffer, scan_buffer + scan_buffer_len, key_someLargeObject, key_someLargeObject + key_size_someLargeObject);
    if ( it != scan_buffer + scan_buffer_len )
    {
      offset += it - scan_buffer;
      break;
    }
    offset += MiB(127);
    rc = mcas->get_direct_offset(pool, offset, scan_buffer_len, scan_buffer, mem->get());
  }

  PLOG("final offset %zu", offset);

  {
    /* as a final test, get_direct_offset at some location in the key */
    auto small_delta = 2;
    component::IMCAS::async_handle_t handle = component::IMCAS::ASYNC_HANDLE_INIT;
    std::size_t length = key_size_someLargeObject - small_delta;
    auto rc2 = mcas->async_get_direct_offset(pool, offset+2, length, scan_buffer, handle, mem->get());
    ASSERT_EQ(S_OK, rc2);
    EXPECT_EQ(key_size_someLargeObject - small_delta, length);
    int iterations = 0;
    while ( mcas->check_async_completion(handle) == E_BUSY )
    {
      ASSERT_TRUE(iterations < 1000000);
      iterations++;
    }
    ASSERT_TRUE(std::equal(scan_buffer, scan_buffer + key_size_someLargeObject - small_delta, key_someLargeObject + small_delta));
  }

  ASSERT_EQ(S_OK, rc);

  ASSERT_OK(mcas->close_pool(pool));
  ASSERT_OK(mcas->delete_pool(poolname));
  free(scan_buffer);
}

TEST_F(KV_test, AsyncGetDirectOffsetRegistered)
{
  async_get_direct_offset(mcas, make_handle_real);
}

TEST_F(KV_test, AsyncGetDirectOffsetUnregistered)
{
  async_get_direct_offset(mcas, make_handle_fake);
}

TEST_F(KV_test, AsyncPutErase)
{
  using namespace component;

  const std::string poolname = "AsyncPutErase";

  auto pool = mcas->create_pool(poolname, MB(32), /* size */
                                0,                /* flags */
                                100);             /* obj count */

  ASSERT_FALSE(pool == IKVStore::POOL_ERROR);

  std::string value0 = "this_is_value_0";

  IMCAS::async_handle_t handle = IMCAS::ASYNC_HANDLE_INIT;
  ASSERT_OK(mcas->async_put(pool, "testKey", value0.data(), value0.length(), handle));
  ASSERT_TRUE(handle != nullptr);

  ASSERT_OK(mcas->erase(pool, "testKey"));

  int iterations = 0;
  while (mcas->check_async_completion(handle) == E_BUSY) {
    ASSERT_TRUE(iterations < 1000000);
    iterations++;
  }

  constexpr int            batch_size = 32;  // see client_fabric_transport.h
  std::vector<std::string> keys;
  std::vector<std::string> values;
  for (int i = 0; i < batch_size; i++) {
    keys.push_back(common::random_string(8));
    values.push_back(common::random_string(48));
  }

  std::queue<IMCAS::async_handle_t> issued;

  /* do multiple runs */
  for (unsigned j = 0; j < 100; j++) {
    /* issue batch */
    for (int i = 0; i < batch_size; i++) {
      IMCAS::async_handle_t handle = IMCAS::ASYNC_HANDLE_INIT;
      ASSERT_OK(mcas->async_put(pool, keys[i], values[i].data(), values[i].length(), handle));
      ASSERT_TRUE(handle != nullptr);
      issued.push(handle);
    }

    /* wait for completions */
    while (!issued.empty()) {
      status_t s = mcas->check_async_completion(issued.front());
      ASSERT_TRUE(s == S_OK || s == E_BUSY);
      if (s == S_OK) issued.pop();
    }

    /* now erase them */
    for (int i = 0; i < batch_size; i++) {
      IMCAS::async_handle_t handle = IMCAS::ASYNC_HANDLE_INIT;
      ASSERT_OK(mcas->async_erase(pool, keys[i], handle));
      ASSERT_TRUE(handle != nullptr);
      issued.push(handle);
    }

    /* wait for completions */
    while (!issued.empty()) {
      status_t s = mcas->check_async_completion(issued.front());
      ASSERT_TRUE(s == S_OK || s == E_BUSY);
      if (s == S_OK) issued.pop();
    }

    std::vector<uint64_t> attr;
    ASSERT_OK(mcas->get_attribute(pool, IMCAS::Attribute::COUNT, attr));
    ASSERT_TRUE(attr[0] == 0);
  }

  ASSERT_OK(mcas->close_pool(pool));
  ASSERT_OK(mcas->delete_pool(poolname));
}

void async_put_direct(component::Itf_ref<component::IMCAS> &mcas, make_handle_t mh_)
{
  using namespace component;

  const std::string poolname = "AsyncPutDirect";

  auto pool = mcas->create_pool(poolname, GB(1), /* size */
                                0,               /* flags */
                                100);            /* obj count */

  ASSERT_FALSE(pool == IKVStore::POOL_ERROR);

  IMCAS::async_handle_t  out_handle      = IMCAS::ASYNC_HANDLE_INIT;
  size_t                 user_buffer_len = MB(128);
  void *                 user_buffer     = aligned_alloc(KB(4), user_buffer_len);
  auto mem = mh_(mcas, user_buffer, user_buffer_len);
  ASSERT_OK(mcas->async_put_direct(pool, "testKey", user_buffer, user_buffer_len, out_handle, mem->get()));
  ASSERT_TRUE(out_handle != nullptr);

  int iterations = 0;
  while (mcas->check_async_completion(out_handle) == E_BUSY) {
    ASSERT_TRUE(iterations < 100000000);
    iterations++;
  }

  void *                 user_buffer2 = aligned_alloc(KB(4), user_buffer_len);
  auto  mem2 = mh_(mcas, user_buffer2, user_buffer_len);

  ASSERT_OK(mcas->get_direct(pool, "testKey", user_buffer2, user_buffer_len, mem2->get()));

  ASSERT_TRUE(memcmp(user_buffer, user_buffer2, user_buffer_len) == 0); /* integrity check */
  ::free(user_buffer);
  ::free(user_buffer2);

  std::vector<uint64_t> attr;
  ASSERT_OK(mcas->get_attribute(pool, IMCAS::Attribute::COUNT, attr));
  ASSERT_TRUE(attr[0] == 1);

  ASSERT_OK(mcas->close_pool(pool));
  ASSERT_OK(mcas->delete_pool(poolname));
}

TEST_F(KV_test, AsyncPutDirectRegistered)
{
  async_put_direct(mcas, make_handle_real);
}

TEST_F(KV_test, AsyncPutDirectUnregistered)
{
  async_put_direct(mcas, make_handle_fake);
}

void async_get_direct(component::Itf_ref<component::IMCAS> &mcas, make_handle_t mh_)
{
  using namespace component;

  const std::string poolname = "AsyncPutDirect";

  auto pool = mcas->create_pool(poolname, GB(1), /* size */
                                0,               /* flags */
                                100);            /* obj count */

  ASSERT_FALSE(pool == IKVStore::POOL_ERROR);

  constexpr size_t user_buffer_len = MB(32);
  constexpr size_t user_buffer_ct = 4;
  std::vector<void *> user_buffer;
  for (auto i = 0; i != user_buffer_ct; ++i)
  {
    user_buffer.push_back(aligned_alloc(KB(4), user_buffer_len));
    auto mem = mh_(mcas, user_buffer.back(), user_buffer_len);
    ASSERT_OK(mcas->put_direct(pool, "testKey" + std::to_string(i), user_buffer[i], user_buffer_len, mem->get()));
  }

  /* read buffer slightly large than write buffer, to test reads into a differently-sized buffer */
  constexpr size_t user_buffer_len2 = MB(33);
  std::vector<IMCAS::async_handle_t> out_handle;
  std::vector<void *> user_buffer2;
  std::vector<std::unique_ptr<handle>> mem2;
  for (auto i = 0; i != user_buffer_ct; ++i)
  {
    out_handle.push_back(+IMCAS::ASYNC_HANDLE_INIT);
    user_buffer2.push_back(aligned_alloc(KB(4), user_buffer_len2));
    mem2.push_back(mh_(mcas, user_buffer2.back(), user_buffer_len2));
  }
  for (auto i = 0; i != user_buffer_ct; ++i)
  {
    auto len = user_buffer_len2;
    ASSERT_OK(mcas->async_get_direct(pool, "testKey" + std::to_string(i), user_buffer2[i], len, out_handle[i], mem2[i]->get()));
    EXPECT_EQ(user_buffer_len, len);
    ASSERT_NE(nullptr, out_handle[i]);
  }

  for ( auto i = 0; i != user_buffer_ct; ++i )
  {
    int iterations = 0;
    while (mcas->check_async_completion(out_handle[i]) == E_BUSY) {
      ASSERT_TRUE(iterations < 100000000);
      iterations++;
    }
    ASSERT_TRUE(memcmp(user_buffer[i], user_buffer2[i], user_buffer_len) == 0); /* integrity check */
    ::free(user_buffer[i]);
    ::free(user_buffer2[i]);
  }

  std::vector<uint64_t> attr;
  ASSERT_OK(mcas->get_attribute(pool, IMCAS::Attribute::COUNT, attr));
  EXPECT_EQ(user_buffer_ct, attr[0]);

  ASSERT_OK(mcas->close_pool(pool));
  ASSERT_OK(mcas->delete_pool(poolname));
}

TEST_F(KV_test, AsyncGetDirectRegistered)
{
  async_get_direct(mcas, make_handle_real);
}

TEST_F(KV_test, AsyncGetDirectUnregistered)
{
  async_get_direct(mcas, make_handle_fake);
}

TEST_F(KV_test, MultiPool)
{
  using namespace component;
  std::map<std::string, IMCAS::pool_t> pools;

  const unsigned COUNT = 32;

  for (unsigned i = 0; i < COUNT; i++) {
    auto poolname = common::random_string(16);

    auto pool = mcas->create_pool(poolname, KB(32), /* size */
                                  0,                /* flags */
                                  100);             /* obj count */

    ASSERT_FALSE(pool == IKVStore::POOL_ERROR);

    pools[poolname] = pool;
  }

  for (auto &p : pools) {
    ASSERT_OK(mcas->close_pool(p.second));
    ASSERT_TRUE(mcas->close_pool(p.second) == E_INVAL);
    ASSERT_OK(mcas->delete_pool(p.first));
  }
}

TEST_F(KV_test, PoolCapacity)
{
  auto poolname = common::random_string(16);

  const unsigned long OBJ_COUNT = getenv("OBJECT_COUNT") ? std::stoul(getenv("OBJECT_COUNT")) : 6000;
  auto           pool      = mcas->create_pool(poolname, MB(32), /* size */
                                0,                /* flags */
                                OBJ_COUNT);       /* obj count */

  for (unsigned i = 0; i < OBJ_COUNT; i++) {
    ASSERT_EQ(S_OK, mcas->put(pool, common::random_string(16), common::random_string(KB(4))));
  }
}

TEST_F(KV_test, BadPutGetOperations)
{
  using namespace component;

  const std::string poolname = "BadPutGetOperations";

  auto pool = mcas->create_pool(poolname, MB(32), /* size */
                                0,                /* flags */
                                100);             /* obj count */

  ASSERT_FALSE(pool == IKVStore::POOL_ERROR);

  /* delete pool fails on mapstore when there is something in it. Bug # DAWN-287 */
  std::string key0   = "key0";
  std::string key1   = "key1";
  std::string value0 = "this_is_value_0";
  std::string value1 = "this_is_value_1_and_its_longer";
  ASSERT_OK(mcas->put(pool, key0, value0, 0));

  std::string out_value;
  ASSERT_OK(mcas->get(pool, key0, out_value));
  ASSERT_TRUE(value0 == out_value);

  /* bad parameters */
  ASSERT_TRUE(mcas->put(pool, key1, nullptr, 0) == E_INVAL);
  ASSERT_TRUE(mcas->put(pool, key1, value0.c_str(), 0) == E_INVAL);
  ASSERT_TRUE(mcas->put(pool, key1, nullptr, 100) == E_INVAL);
  ASSERT_TRUE(mcas->put(0x0, key1, nullptr, 100) == E_INVAL);

  ASSERT_OK(mcas->close_pool(pool));
  ASSERT_OK(mcas->delete_pool(poolname));
}

int main(int argc, char *argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  namespace po = boost::program_options;

  try {
    po::options_description desc("Options");

    desc.add_options()("help", "Show help")("server", po::value<std::string>(), "Server hostname")
        ("src_addr", po::value<std::string>(), "Source IP address")
        ("device", po::value<std::string>(), "Device (e.g. mlx5_0)")
        ("port", po::value<std::uint16_t>()->default_value(0), "Server port. Default 0 (mapped to 11911 for verbs, 11921 for sockets)")
        ("debug", po::value<unsigned>()->default_value(0), "Debug level")
        ("patience", po::value<unsigned>()->default_value(30), "Patience with server (seconds)")
      ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help") > 0) {
      std::cout << desc;
      return -1;
    }

    if (vm.count("server") == 0) {
      std::cout << "--server option is required\n";
      return -1;
    }

    g_options.server      = vm["server"].as<std::string>();
    if ( vm.count("src_addr") )
    {
      g_options.src_addr = vm["src_addr"].as<std::string>();
    }
    if ( vm.count("device") )
    {
      g_options.device = vm["device"].as<std::string>();
    }
    if ( ! g_options.src_addr && ! g_options.device )
    {
      g_options.device = "mlx5_0";
    }
    g_options.port        = vm["port"].as<uint16_t>();
    g_options.debug_level = vm["debug"].as<unsigned>();
    g_options.patience = vm["patience"].as<unsigned>();

    mcas = init(g_options.server, g_options.port);
    assert(mcas);

    auto r = RUN_ALL_TESTS();
    PLOG("kv-test has finished");
  }
  catch (const Exception &e) {
    printf("failed with exception: %s\n", e.cause());
    return -1;
  }
  catch (const po::error &e) {
    printf("bad command line option: %s\n", e.what());
    return -1;
  }

  return 0;
}
