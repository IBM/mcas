/* note: we do not include component source, only the API definition */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <ccpm/cca.h>
#include <common/errors.h>
#include <common/logging.h>
#include <common/utils.h>

#include <ccpm/value_tracked.h>
#include <ccpm/container_cc.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Weffc++"
#include <EASTL/bitset.h>
#include <EASTL/iterator.h>
#include <EASTL/list.h>
#include <EASTL/vector.h>
#pragma GCC diagnostic pop


//#define GPERF_TOOLS

#ifdef GPERF_TOOLS
#include <gperftools/profiler.h>
#endif
#include <libpmem.h>
#include <cstdlib> // alligned_alloc
#include <string> // stoull

struct {
  uint64_t uuid;
} Options;

namespace
{
	struct pmem_persister final
		: public ccpm::persister
	{
		void persist(common::byte_span s) override
		{
			::pmem_persist(::base(s), ::size(s));
		}
	};
}

// The fixture for testing class Foo.
class Libccpm_test : public ::testing::Test {
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
};

// Test adding new regions to the heap
//
TEST_F(Libccpm_test, heap_exhaustion)
{
  std::size_t size = KB(4*20);
  auto pr = aligned_alloc(4096,size);
  ASSERT_NE(nullptr, pr);

  pmem_persister persister;
  ccpm::region_vector_t rv(ccpm::region_vector_t::value_type(common::make_byte_span(pr, size)));

  ccpm::cca heap(&persister, rv);

  std::vector<void*> ptrs;
  void * p;
  for(unsigned i=0;i<50;i++) {
  retry:
    try {
      p = heap.allocate(4096, 4096);
      PLOG("allocated @ %p", p);
    }
    catch(...) {
      auto additional = aligned_alloc(0xFFFFFF,size);
      ccpm::region_vector_t additional_rv(ccpm::region_vector_t::value_type(common::make_byte_span(additional, size)));
      heap.add_regions(additional_rv);
      goto retry;
    }
  }  
}

// using logged_ptr_to_int = ccpm::value_tracked<int *, ccpm::tracker_log>;
// using logged_shared_ptr_to_int = ccpm::value_tracked<std::shared_ptr<int>, ccpm::tracker_log>;


// Test adding new regions to the heap
//
TEST_F(Libccpm_test, list_container)
{
  std::size_t size = KB(128);
  auto pr = aligned_alloc(4096,size);
  ASSERT_NE(nullptr, pr);

  /* persister defines a function used to persist/flush data */
  pmem_persister persister;

  /* create region vector */
  ccpm::region_vector_t rv(ccpm::region_vector_t::value_type(common::make_byte_span(pr, size)));

 	using logged_int = ccpm::value_tracked<int, ccpm::tracker_log>;
	using cc_list = ccpm::container_cc<eastl::list<logged_int, ccpm::allocator_tl>>;

  /* create new instance */
  {
    ccpm::cca heap(&persister, rv);
    
    void *root = heap.allocate_root(sizeof(cc_list)); //heap.allocate(list_size);
    ASSERT_TRUE(root);

    auto ccl = new (root) cc_list(&persister, heap);
    PLOG("ccl: %p", reinterpret_cast<void*>(ccl));
    
    ccl->container->push_back(6);
    ccl->container->push_back(7);
    ccl->container->push_back(8);
    ccl->commit();
    ASSERT_TRUE(ccl->container->size() == 3);
  }

  /* let's try reconsistution */
  {
    /* first get the heap back */
    ccpm::cca heap(&persister, rv, ccpm::accept_all);

    /* then cast the root pointer */
    auto ccl = reinterpret_cast<cc_list*>(::base(heap.get_root()));
    PLOG("ccl: %p", reinterpret_cast<void*>(ccl));
    ccl->rollback();


    ccl->container->push_back(6);
    ccl->commit();
    
    PLOG("list size: %lu", ccl->container->size());
    ASSERT_TRUE(ccl->container->size() == 4);
  }  
}

// Test adding new regions to the heap
//
TEST_F(Libccpm_test, list_container_2)
{
  std::size_t size = KB(128);
  auto pr = aligned_alloc(4096,size);
  ASSERT_NE(nullptr, pr);

  /* persister defines a function used to persist/flush data */
  pmem_persister persister;

  /* create region vector */
  ccpm::region_vector_t rv(ccpm::region_vector_t::value_type(common::make_byte_span(pr, size)));

 	using logged_int = ccpm::value_tracked<uint64_t, ccpm::tracker_log>;
	using cc_list = ccpm::container_cc<eastl::list<logged_int, ccpm::allocator_tl>>;

  ccpm::cca heap(&persister, rv);
  
  void *root = heap.allocate_root(sizeof(cc_list)); //heap.allocate(list_size);
  ASSERT_TRUE(root);

  auto ccl = new (root) cc_list(&persister, heap);
  PLOG("ccl: %p", reinterpret_cast<void*>(ccl));
    
  ccl->container->push_back(6);
  ccl->container->push_back(7);
  ccl->container->push_back(8);
  ccl->commit();
  ASSERT_TRUE(ccl->container->size() == 3);

  /* iterate */
  unsigned count = 0;
  for ( auto it = ccl->container->begin(); it != ccl->container->end(); ++it )
    {
      auto i = *it;
      std::cout << "value: " << i << "\n";
      count++;
    }

  ASSERT_TRUE(count == 3);
}


typedef struct {
  void * ptr;
}
Element;

TEST_F(Libccpm_test, list_container_3)
{
  std::size_t size = KB(128);
  auto pr = aligned_alloc(4096,size);
  ASSERT_NE(nullptr, pr);

  /* persister defines a function used to persist/flush data */
  pmem_persister persister;

  /* create region vector */
  ccpm::region_vector_t rv(ccpm::region_vector_t::value_type(common::make_byte_span(pr, size)));

 	using logged_element = ccpm::value_tracked<Element, ccpm::tracker_log>;
	using cc_list = ccpm::container_cc<eastl::list<logged_element, ccpm::allocator_tl>>;

  ccpm::cca heap(&persister, rv);
  
  void *root = heap.allocate_root(sizeof(cc_list)); //heap.allocate(list_size);
  ASSERT_TRUE(root);

  auto ccl = new (root) cc_list(&persister, heap);
  PLOG("ccl: %p", reinterpret_cast<void*>(ccl));
    
  ccl->container->push_back(Element{nullptr});
  ccl->container->push_back(Element{nullptr});
  ccl->container->push_back(Element{nullptr});

  ASSERT_TRUE(ccl->container->size() == 3);
}


int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  if (argc > 1) {
    Options.uuid = std::stoull(argv[1]);
  }
  (void) Options.uuid; // unused
  auto r = RUN_ALL_TESTS();

  return r;
}
