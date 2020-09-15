#include <api/ado_itf.h>
#include <api/components.h>
#include <common/str_utils.h>
#include <common/utils.h>
#include <common/logging.h>
#include <gtest/gtest.h>
#include <nupm/devdax_manager.h>
#include <nupm/mcas_mod.h>
#include <thread>

using namespace component;
using namespace common;
using namespace std;

namespace {
// The fixture for testing class Foo.
class IADO_manager_proxy_test : public ::testing::Test {
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
};

unsigned token = 667;
size_t g_size = 1024;

TEST_F(IADO_manager_proxy_test, Instantiate) {
  /* create object instance through factory */
  // allocate memory
  vector<nupm::Devdax_manager::config_t> conf;
  nupm::Devdax_manager::config_t c;
  c.path = "/dev/dax0.0";
  c.addr = 0x9000000000;
  c.region_id = 0;
  conf.push_back(c);
  nupm::Devdax_manager ddm(conf, true);

  nupm::revoke_memory(token);
  size_t size = g_size;
  char *pop = static_cast<char *>(ddm.create_region(1234, 0, size));
  memset(pop, 0, size);
  PLOG("touched memory.");
  strcpy(pop, "Hello!");
  PLOG("I have put the memory (%p): %s", reinterpret_cast<void*>(pop), pop);

  status_t rc = nupm::expose_memory(token, pop, size);
  if(rc != S_OK) PLOG("rc=%d", rc);
  ASSERT_TRUE(rc == S_OK);
}

TEST_F(IADO_manager_proxy_test, Map) {
  size_t size = g_size;
  char *addr = static_cast<char *>(nupm::mmap_exposed_memory(token, size, reinterpret_cast<void *>(0xa00000000)));
  ASSERT_TRUE(addr);
  PLOG("reading memory: %s, size: %zu", addr, size);
  strcpy(addr, "World!");
  PLOG("I have changed the memory: %s",addr);
}

// TEST_F(IADO_manager_proxy_test, Shutdown) {
//   size_t size;
//   char *re = (char *)nupm::mmap_exposed_memory(token, size);
//   ASSERT_TRUE(re);
//   cout << "Read changed memory: " << re << endl;
//   ASSERT_TRUE(nupm::revoke_memory(0) == S_OK);
// }

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
