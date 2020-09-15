#include <api/ado_itf.h>
#include <api/components.h>
#include <common/str_utils.h>
#include <common/utils.h>
#include <gtest/gtest.h>

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
    delete _ado_manager;
  }

  // Objects declared here can be used by all tests in the test case
  static component::IADO_manager_proxy *_ado_manager;
};

component::IADO_manager_proxy *IADO_manager_proxy_test::_ado_manager;

TEST_F(IADO_manager_proxy_test, Instantiate) {
  /* create object instance through factory */
  component::IBase *comp = component::load_component(
      "libcomponent-adomgrproxy.so", component::ado_manager_proxy_factory);

  ASSERT_TRUE(comp);
  auto fact =
    make_itf_ref(
      static_cast<IADO_manager_factory *>(
        comp->query_interface(IADO_manager_factory::iid())
      )
    );

  _ado_manager = fact->create(1, 0);
}

TEST_F(IADO_manager_proxy_test, createADOProx) {
  PINF("Run create ADO proxy");
  string filename = "~/mcas/build/src/components/ado/daemon/ado";
  vector<string> args;
  ASSERT_TRUE(_ado_manager);
  IADO_proxy *ado = _ado_manager->create(filename, args, 0, 0);
  ASSERT_TRUE(ado);
  _ado_manager->shutdown(ado);
}

/*
TEST_F(IADO_manager_proxy_test, Clean) { _kvindex->clear(); }

TEST_F(IADO_manager_proxy_test, Insert) {
  string key = "MyKey1";
  _kvindex->insert(key);
  key = "MyKey2";
  _kvindex->insert(key);
  key = "abc";
  _kvindex->insert(key);
  PINF("Size: %ld", _kvindex->count());
}

TEST_F(IADO_manager_proxy_test, Get) {
  std::string a = _kvindex->get(0);
  PINF("Key= %s", a.c_str());
  a = _kvindex->get(1);
  PINF("Key= %s", a.c_str());
  a = _kvindex->get(2);
  PINF("Key= %s", a.c_str());
}

TEST_F(IADO_manager_proxy_test, FIND) {
  string regex = "abc";
  uint64_t end = _kvindex->count() - 1;
  string key;
  _kvindex->find(regex, 0, IKVIndex::FIND_TYPE_EXACT, end, key);
  PINF("Key= %s", key.c_str());
}

TEST_F(IADO_manager_proxy_test, Erase) { _kvindex->erase("MyKey"); }

TEST_F(IADO_manager_proxy_test, Count) { PINF("Size: %lu", _kvindex->count()); }
*/

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
