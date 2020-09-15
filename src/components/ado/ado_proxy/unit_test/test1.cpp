/* note: we do not include component source, only the API definition */
#include <api/components.h>
#include <api/kvindex_itf.h>
#include <common/str_utils.h>
#include <common/utils.h>
#include <gtest/gtest.h>
#include <ctime>

#define COUNT 1000000
#define LENGTH 16

using namespace component;
using namespace common;
using namespace std;

namespace
{
// The fixture for testing class Foo.
class KVIndex_test : public ::testing::Test {
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
  static component::IKVIndex *_kvindex;
};

component::IKVIndex *KVIndex_test::_kvindex;

TEST_F(KVIndex_test, Instantiate)
{
  /* create object instance through factory */
  component::IBase *comp = component::load_component(
      "libcomponent-indexrbtree.so", component::rbtreeindex_factory);

  ASSERT_TRUE(comp);
  auto fact =
    make_itf_ref(
      static_cast<IKVIndex_factory *>(comp->query_interface(IKVIndex_factory::iid()))
   );

  _kvindex = fact->create("owner", "name");
}

TEST_F(KVIndex_test, InsertPerf)
{
  string *keys = new string[COUNT];
  for (int i = 0; i < COUNT; i++) {
    keys[i] = random_string(LENGTH);
  }
  clock_t start = clock();
  for (int i = 0; i < COUNT; i++) {
    _kvindex->insert(keys[i]);
  }
  delete[] keys;
  double duration = (clock() - start) / (double) CLOCKS_PER_SEC;
  PINF("Time sec: %lf", duration);
  PINF("Size: %ld", _kvindex->count());
}

TEST_F(KVIndex_test, Clean) { _kvindex->clear(); }

TEST_F(KVIndex_test, Insert)
{
  string key = "MyKey1";
  _kvindex->insert(key);
  key = "MyKey2";
  _kvindex->insert(key);
  key = "abc";
  _kvindex->insert(key);
  PINF("Size: %ld", _kvindex->count());
}

TEST_F(KVIndex_test, Get)
{
  std::string a = _kvindex->get(0);
  PINF("Key= %s", a.c_str());
  a = _kvindex->get(1);
  PINF("Key= %s", a.c_str());
  a = _kvindex->get(2);
  PINF("Key= %s", a.c_str());
}

TEST_F(KVIndex_test, FIND)
{
  string   regex = "abc";
  uint64_t end   = _kvindex->count() - 1;
  string   key;
  _kvindex->find(regex, 0, IKVIndex::FIND_TYPE_EXACT, end, key);
  PINF("Key= %s", key.c_str());
}

TEST_F(KVIndex_test, Erase) { _kvindex->erase("MyKey"); }

TEST_F(KVIndex_test, Count) { PINF("Size: %lu", _kvindex->count()); }


}  // namespace

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
