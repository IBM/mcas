/* note: we do not include component source, only the API definition */
#include <common/cycles.h>
#include <common/rand.h>
#include <common/utils.h>
#include <boost/icl/split_interval_map.hpp>
#include <gtest/gtest.h>
#include <chrono>
#include <list>

#include "heap_allocator.h"
#include "arena_alloc.h"
#include "devdax_manager.h"
#include "rc_alloc_avl.h"
#include "rc_alloc_lb.h"
#include "tx_cache.h"

//#define GPERF_TOOLS

#ifdef GPERF_TOOLS
#include <gperftools/profiler.h>
#endif

struct {
  uint64_t uuid;
} Options;

// The fixture for testing class Foo.
class Libnupm_test : public ::testing::Test {
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

// static bool memcheck(void * p, char val, size_t len)
// {
//   for(size_t i=0; i<len; i++) {
//     if(((char *)p)[i] != val)
//       return false;
//   }
//   return true;
// }

// #define RUN_AVL_RANGE_ALLOCATOR_TESTS
// #define RUN_RPALLOCATOR_TESTS
// #define RUN_VMEM_ALLOCATOR_TESTS
#define RUN_DEVDAX_TEST
// #define RUN_AVL_RCA_TEST
// #define RUN_AVL_STRESS_TEST
// #define RUN_AVL_RECONST_TEST
// #define RUN_MALLOC_STRESS_TEST
// #define RUN_LB_TEST
// #define RUN_LB_STRESS_TEST
// #define RUN_LB_INTEGRITY_TEST
// #define RUN_LB_RECONST_TEST

using namespace std;
using namespace boost::icl;

typedef split_interval_set<addr_t> interval_set_t;

#ifdef RUN_AVL_RANGE_ALLOCATOR_TESTS
TEST_F(Libnupm_test, AVLRange)
{
  const addr_t BASE = 0x900000000;
  const size_t ARENA_SIZE = GB(32);
  const size_t COUNT = 100000;
  init_genrand64(0xF00B);


  std::vector<iovec> log;
  interval_set_t iset;

  PLOG("Building AVL...");
  std::string before, after;
  {
    core::AVL_range_allocator avl(BASE, ARENA_SIZE);
    for (size_t i = 1; i < COUNT; i++) {
      size_t s = round_up(((genrand64_int64() % 4096) + 8), 8);
      assert(s % 8 == 0);
      auto mr = avl.alloc(s, 8);
      void * p = mr->paddr();
      ASSERT_TRUE(check_aligned(p, 8));
      ASSERT_TRUE(((addr_t)p) >= BASE);

      /* closed, both value included in range */
      auto ival = interval<addr_t>::closed((addr_t)p, ((addr_t)p) + s - 1);

      /* Check for overlap */
      auto itRes = iset.equal_range(ival);
      for( auto it = itRes.first; it != itRes.second; ++it ) {
        PERR("detected existing region: %lx-%lx", it->lower(), it->upper());
        ASSERT_TRUE(false);
      }

      iset += ival;
      log.push_back({p, s});
    }
    avl.dump_info(&before);
  }

  /* now do reconstitution */
  PLOG("Reconstituting AVL...");
  {
    core::AVL_range_allocator avl(BASE, ARENA_SIZE);

    for(auto a : log) {
      avl.alloc_at((addr_t)a.iov_base, a.iov_len);
    }
    avl.dump_info(&after);
  }
  ASSERT_TRUE(before==after);

}
#endif

#ifdef RUN_AVL_RECONST_TEST
TEST_F(Libnupm_test, RcAllocatorAVLReconstitute)
{
  const size_t ARENA_SIZE = GB(32);
  const size_t COUNT = 100000;
  const size_t MAX_SMALL_OBJ = 4096; // see mapper.h
  void * p = aligned_alloc(GB(1), ARENA_SIZE);
  ASSERT_TRUE(p);
  init_genrand64(0xF00B);

  std::vector<iovec> log;
  interval_set_t iset;

  /* set up AVL allocator */
  std::string state_A, state_B;

  PLOG("Populating AVL allocator...");
  {
    nupm::Rca_AVL rca;
    rca.add_managed_region(p, ARENA_SIZE, 0);

    for (size_t i = 1; i < COUNT; i++) {
      size_t s = round_up(((genrand64_int64() % MAX_SMALL_OBJ) + 8), 8);
      //      s+= KB(4); // force to large objects
      assert(s % 8 == 0);
      void *p = rca.alloc(s, 0 /* numa */, 8 /* alignment */);
      ASSERT_TRUE(check_aligned(p, 8));

      /* closed, both value included in range */
      auto ival = interval<addr_t>::closed((addr_t)p, ((addr_t)p) + s - 1);

      /* Check for overlap */
      auto itRes = iset.equal_range(ival);
      for( auto it = itRes.first; it != itRes.second; ++it ) {
        PERR("detected existing region: %lx-%lx", it->lower(), it->upper());
        ASSERT_TRUE(false);
      }

      iset += ival;
      log.push_back({p, s});
    }

    rca.debug_dump(&state_A);
  }

  /* now do reconstitution */
  PLOG("Reconstituting AVL allocator...");
  {
    nupm::Rca_AVL rca;
    rca.add_managed_region(p, ARENA_SIZE, 0);

    for(auto a : log) {
      rca.inject_allocation(a.iov_base, a.iov_len, 0 /* numa */);
    }
    rca.debug_dump(&state_B);
  }
  ASSERT_TRUE(state_A == state_B);

}
#endif

#ifdef RUN_LB_RECONST_TEST
TEST_F(Libnupm_test, RcAllocatorLBReconstitute)
{
  const size_t ARENA_SIZE = GB(32);
  const size_t COUNT = 100000;
  const size_t MAX_SMALL_OBJ = 4096; // see mapper.h
  void * p = (void*) 0x900000000; //aligned_alloc(GB(1), ARENA_SIZE);
  ASSERT_TRUE(p);

  struct info_t {
    void *   p;
    size_t   size;

    bool operator< (info_t cmp)  {
      return p < cmp.p;
    }
  };

  struct compare {
    bool operator() (info_t a, info_t b)
    {
      return a.p == b.p && a.size == b.size;
    }
  };

  std::list<info_t> log;
  interval_set_t iset;
  std::string before;

  /* do allocations */
  PLOG("Populating LB allocator...");
  {
    nupm::Rca_LB rca;
    rca.add_managed_region(p, ARENA_SIZE, 0);
    init_genrand64(0xF00B);

    for (size_t i = 1; i < COUNT; i++) {
      size_t s = round_up(((genrand64_int64() % MAX_SMALL_OBJ) + 8), 8);
      assert(s % 8 == 0);
      void *p = rca.alloc(s, 0 /* numa */, 8 /* alignment */);
      ASSERT_TRUE(check_aligned(p, 8));

      /* closed, both value included in range */
      auto ival = interval<addr_t>::closed((addr_t)p, ((addr_t)p) + s - 1);

      /* Check for overlap */
      auto itRes = iset.equal_range(ival);
      for( auto it = itRes.first; it != itRes.second; ++it ) {
        PERR("detected existing region: %lx-%lx", it->lower(), it->upper());
        ASSERT_TRUE(false);
      }

      iset += ival;

      log.push_back({p, s});
    }
    /* capture state */
    rca.debug_dump(&before);
  }

  std::string before_repeat;
  /* do same again */
  PLOG("Again, populating LB allocator...");
  {
    nupm::Rca_LB rca;
    rca.add_managed_region(p, ARENA_SIZE, 0);
    init_genrand64(0xF00B);

    for (size_t i = 1; i < COUNT; i++) {
      size_t s = round_up(((genrand64_int64() % 4096) + 8), 8);
      assert(s % 8 == 0);
      void *p = rca.alloc(s, 0 /* numa */, 8 /* alignment */);
      ASSERT_TRUE(check_aligned(p, 8));
    }
    /* capture state */
    rca.debug_dump(&before_repeat);
  }
  ASSERT_TRUE(before == before_repeat);

  PLOG("Checking for duplicates...");
  /* check for duplicates */
  {
    std::list<info_t> tmplog = log;
    tmplog.sort();
    size_t before_uniq = tmplog.size();
    tmplog.unique(compare());
    ASSERT_TRUE(before_uniq == log.size());
  }

  /* now do reconstitution */
  PLOG("Reconstituting LB allocator...");
  std::string after;
  {
    nupm::Rca_LB rca;
    rca.add_managed_region(p, ARENA_SIZE, 0);

    for(auto a : log) {
      ASSERT_TRUE(a.size <= 4104);
      rca.inject_allocation(a.p, a.size, 0 /* numa */);
    }
    rca.debug_dump(&after);
  }
  PLOG("before state len=%lu / after state len = %lu",
       before.length(), after.length());
  ASSERT_TRUE(before == after);


}
#endif

#ifdef RUN_LB_INTEGRITY_TEST
TEST_F(Libnupm_test, RcAllocatorLBIntegrity)
{
  const size_t ARENA_SIZE = GB(32);
  void * p = aligned_alloc(GB(1), ARENA_SIZE);
  ASSERT_TRUE(p);
  uint64_t tag = 1;

  nupm::Rca_LB rca;
  rca.add_managed_region(p, ARENA_SIZE, 0);

  struct info_t {
    void *   p;
    size_t   size;
    uint64_t tag;
  };

  std::vector<info_t> log;

  /* populate with 1M entries */
  PLOG("Populating ...");
  tag = 1;
  const size_t COUNT = 1000000;
  for (size_t i = 1; i < COUNT; i++) {
    size_t s = round_up(((genrand64_int64() % 4096) + 8), 8);
    assert(s % 8 == 0);
    void *p = rca.alloc(s, 0 /* numa */, 8 /* alignment */);
    ASSERT_TRUE(check_aligned(p, 8));
    uint64_t *iptr = (uint64_t *) p;
    for (size_t i = 0; i < s / 8; i++) {
      iptr[i] = tag;
    }
    log.push_back({p, s, tag});
    tag++;
  }

  /* check data */
  PLOG("Checking integrity...");
  tag                = 1;
  unsigned alloc_num = 0;
  for (auto &e : log) {
    uint64_t *iptr = (uint64_t *) e.p;
    for (size_t i = 0; i < e.size / 8; i++) {
      if (iptr[i] != e.tag) PLOG("iptr[%lu] = %lu", iptr[i], e.tag);
      ASSERT_TRUE(iptr[i] == e.tag);
    }

    alloc_num++;
    tag++;
  }

  /* free some */
  PLOG("Freeing half....");
  for (size_t i = 1; i < COUNT / 2; i++) {
    auto e = log.back();
    log.pop_back();
    rca.free(e.p, 0, e.size);
  }

  PLOG("Re-checking integrity...");
  tag       = 1;
  alloc_num = 0;
  for (auto &e : log) {
    uint64_t *iptr = (uint64_t *) e.p;
    for (size_t i = 0; i < e.size / 8; i++) {
      if (iptr[i] != e.tag) PLOG("iptr[%lu] = %lu", iptr[i], e.tag);
      ASSERT_TRUE(iptr[i] == e.tag);
    }

    alloc_num++;
    tag++;
  }

  PLOG("Freeing rest...");
  while (!log.empty()) {
    auto e = log.back();
    log.pop_back();
    rca.free(e.p, 0, e.size);
  }

  {
    /* All objects freed. Frees should return unused buckets to the arena.
     * Check that dump, which enumerates free buckets, is a trivially small
     * size.
     */
    std::string d;
    rca.debug_dump(&d);
    EXPECT_GT(1000, d.size());
  }

  free(p);
}
#endif

#ifdef RUN_LB_STRESS_TEST
TEST_F(Libnupm_test, RcAllocatorLBStress)
{
  const size_t ARENA_SIZE = GB(32);
  void *       p          = aligned_alloc(GB(1), ARENA_SIZE);
  ASSERT_TRUE(p);

  nupm::Rca_LB rca;
  rca.add_managed_region(p, ARENA_SIZE, 0);

#ifdef GPERF_TOOLS
  ProfilerStart("cpu_profile");
#endif

  __sync_synchronize();
  auto start_time = std::chrono::high_resolution_clock::now();
  /* populate with 1M entries */
  const size_t COUNT = 10000000;
  for (size_t i = 0; i < COUNT; i++) {
    size_t s = 8;  // (genrand64_int64() % 256) + 8;
    rca.alloc(s, 0 /* numa */, 0 /* alignment */);
  }
  __sync_synchronize();

#ifdef GPERF_TOOLS
  ProfilerStop();
#endif

  auto   end_time = std::chrono::high_resolution_clock::now();
  auto   secs     = std::chrono::duration<double>(
                    end_time - start_time)
                    .count();
  double per_sec = double(COUNT) / secs;
  PINF("Non-aligned, 64 byte size, insertion rate: %.0fK /sec (mean "
       "latency:%.0f usec)",
       per_sec / 1000.0, 1000000.0 / per_sec);

  free(p);
}
#endif

#ifdef RUN_LB_TEST
TEST_F(Libnupm_test, RcAllocatorLB)
{
  void *p = aligned_alloc(GB(1), GB(8));
  ASSERT_TRUE(p);
  void *q = aligned_alloc(GB(1), GB(8));
  ASSERT_TRUE(q);

  PLOG("p=%p", p);
  PLOG("q=%p", q);

  std::vector<iovec> allocations;

  std::string state_A, state_B;
  { /* create allocator */
    nupm::Rca_LB rca;
    rca.add_managed_region(p, GB(8), 0);
    rca.add_managed_region(q, GB(8), 1);

    for (unsigned i = 0; i < 10; i++) {
      void *a = rca.alloc(64, 0, 16);
      ASSERT_TRUE(a != nullptr);
      PLOG("Allocated: %p", a);
      allocations.push_back({a, (i + 1) * 32});
    }

    /* logical power-fail */
    rca.debug_dump(&state_A);

    /* now we should be able to free */
    for (auto &i : allocations) {
      rca.free(i.iov_base, 0);
      PLOG("Freed: %p", i.iov_base);
    }
  }

  // {
  //   nupm::Rca_LB rca;
  //   rca.add_managed_region(p, GB(8), 0);
  //   rca.add_managed_region(q, GB(8), 1);

  //   for(auto& i: allocations) {
  //     rca.inject_allocation(i.iov_base, i.iov_len, 0);
  //   }

  //   rca.debug_dump(&state_B);

  //   /* state A and B should be the same */
  //   ASSERT_TRUE(state_A == state_B);

  //   /* now we should be able to free */
  //   for(auto& i: allocations) {
  //     rca.free(i.iov_base, 0);
  //   }
  // }

  free(p);
  free(q);
}
#endif

#ifdef RUN_DEVDAX_TEST
TEST_F(Libnupm_test, DevdaxManager)
{
  {
    size_t s = nupm::get_dax_device_size("/dev/dax0.0");
    PLOG("sizeof device /dev/dax0.0 is %ld bytes", s);
    ASSERT_TRUE(s > 0);
  }

  nupm::Devdax_manager::config_t config;
  config.path = "/dev/dax0.0";
  config.addr = 0x900000000;
  config.region_id = 0;
  
  {
    nupm::Devdax_manager ddm({config},true);
  }

  nupm::Devdax_manager ddm({config});  // rebuild

  
  size_t   p_len = 0;
  uint64_t uuid  = Options.uuid;

  if(uuid == 0) uuid = 1;

  size_t size = MB(256);
  ddm.debug_dump(0); /* region id 0 */

  PLOG("Opening existing region..");
  void *p = ddm.open_region(uuid, 0, &p_len);
  if (p) {
    PLOG("Opened existing region %p OK", p);
    ddm.debug_dump(0);
    PLOG("Now erasing it...");
    ddm.erase_region(uuid, 0);
    ddm.debug_dump(0);
  }

  p = ddm.create_region(uuid, 0, size);
  if (p) PLOG("created region %p ", p);
  ASSERT_TRUE(p);
  memset(p, 0, 4096);
  ddm.debug_dump(0);

  PLOG("Erase...");
  ddm.erase_region(uuid, 0);
  ddm.debug_dump(0);

  PLOG("Re-create...");
  void *q = ddm.create_region(uuid, 0, size);
  ASSERT_TRUE(q == p);

  ddm.debug_dump(0);
  ddm.erase_region(uuid, 0);

}
#endif

#ifdef RUN_AVL_RCA_TEST
TEST_F(Libnupm_test, RcAllocatorAVL)
{
  void *p = aligned_alloc(GB(1), GB(8));
  ASSERT_TRUE(p);
  void *q = aligned_alloc(GB(1), GB(8));
  ASSERT_TRUE(q);

  PLOG("p=%p", p);
  PLOG("q=%p", q);

  std::vector<iovec> allocations;

  std::string state_A, state_B;
  { /* create allocator */
    nupm::Rca_AVL rca;
    rca.add_managed_region(p, GB(8), 0);
    rca.add_managed_region(q, GB(8), 1);

    for (unsigned i = 0; i < 10; i++) {
      void *a = rca.alloc((i + 1) * 32, 0, 16);
      allocations.push_back({a, (i + 1) * 32});
    }

    /* logical power-fail */
    rca.debug_dump(&state_A);
  }

  {
    nupm::Rca_AVL rca;
    rca.add_managed_region(p, GB(8), 0);
    rca.add_managed_region(q, GB(8), 1);

    for (auto &i : allocations) {
      rca.inject_allocation(i.iov_base, i.iov_len, 0);
    }

    rca.debug_dump(&state_B);

    /* state A and B should be the same */
    ASSERT_TRUE(state_A == state_B);

    /* now we should be able to free */
    for (auto &i : allocations) {
      rca.free(i.iov_base, 0);
    }
  }

  free(p);
  free(q);
}
#endif

#ifdef RUN_AVL_STRESS_TEST
TEST_F(Libnupm_test, RcAllocatorStress)
{
  const size_t ARENA_SIZE = GB(32);
  void *       p          = aligned_alloc(GB(1), ARENA_SIZE);
  ASSERT_TRUE(p);

  nupm::Rca_AVL rca;
  rca.add_managed_region(p, ARENA_SIZE, 0);

  __sync_synchronize();
  auto start_time = std::chrono::high_resolution_clock::now();
  /* populate with 1M entries */
  const size_t COUNT = 10000000;
  for (size_t i = 0; i < COUNT; i++) {
    size_t s = 32;  //(genrand64_int64() % 32) + 1;
    rca.alloc(s, 0 /* numa */, 8 /* alignment */);
  }
  __sync_synchronize();
  auto   end_time = std::chrono::high_resolution_clock::now();
  auto   secs     = std::chrono::duration<double>(end_time - start_time) .count();
  double per_sec = double(COUNT) / secs;
  PINF("Non-aligned, 32 byte size, insertion rate: %.0fK /sec (mean "
       "latency:%.0f usec)",
       per_sec / 1000.0, 1000000.0 / per_sec);

  free(p);
}
#endif

#ifdef RUN_AVL_STRESS_TEST_MEMKIND
TEST_F(Libnupm_test, RcAllocatorStressMemkind)
{
  const size_t ARENA_SIZE = GB(32);
  void *       p          = aligned_alloc(GB(1), ARENA_SIZE);
  ASSERT_TRUE(p);

  nupm::Rca_AVL rca;
  rca.add_managed_region("/dev/dax0.0", 0);

  //  std::vector<iovec> allocations;
  //  allocations.push_back({a, s});
  __sync_synchronize();
  auto start_time = std::chrono::high_resolution_clock::now();
  /* populate with 1M entries */
  const size_t COUNT = 10000000;
  for (size_t i = 0; i < COUNT; i++) {
    size_t s = 32;  //(genrand64_int64() % 32) + 1;
    rca.alloc(s, 0 /* numa */, 0 /* alignment */);
  }
  __sync_synchronize();
  auto   end_time = std::chrono::high_resolution_clock::now();
  double secs     = std::chrono::duration<double>(end_time - start_time).count();
  double per_sec = double(COUNT) / secs;
  PINF("Non-aligned, 32 byte size, insertion rate: %.0fK /sec (mean "
       "latency:%.0f usec)",
       per_sec / 1000.0, 1000000.0 / per_sec);

  free(p);
}
#endif

#ifdef RUN_MALLOC_STRESS_TEST
TEST_F(Libnupm_test, MallocStress)
{
  //  std::vector<iovec> allocations;
  //  allocations.push_back({a, s});
  __sync_synchronize();
  auto start_time = std::chrono::high_resolution_clock::now();
  /* populate with 1M entries */
  const size_t COUNT = 10000000;
  for (size_t i = 0; i < COUNT; i++) {
    size_t s = 32;  //(genrand64_int64() % 32) + 1;
    void * p = ::malloc(s);
    assert(p);
  }
  __sync_synchronize();
  auto end_time = std::chrono::high_resolution_clock::now();
  auto secs     = std::chrono::duration<double>(end_time - start_time) .count();
  PLOG("malloc: secs %.2f", secs);
  double per_sec = double(COUNT) / secs;
  PINF("malloc: Non-aligned, 32 byte size, insertion rate: %.2fM /sec",
       per_sec / 1000000.0);
}
#endif

#if 0
TEST_F(Libnupm_test, NdControl)
{
  nupm::ND_control ctrl;
  PLOG("ND_control init complete!");
}
#endif

#if 0
TEST_F(Libnupm_test, TxCache)
{
  size_t NUM_PAGES = 2048;
  size_t p_size = NUM_PAGES * MB(2);
  void * p = nupm::allocate_virtual_pages(p_size/MB(2), MB(2), 0x900000000ULL);

  cpu_time_t start = rdtsc();
  for(size_t i=0;i<NUM_PAGES;i++) {
    ((char*)p)[i*MB(2)]='a';
  }
  cpu_time_t delta = (rdtsc() - start)/NUM_PAGES;
  PLOG("Mean PF cost: (%f usec) %lu", common::cycles_to_usec(delta), delta);
  // { // important
  //   core::Heap_allocator<char> heap(p, p_size, "heapA");

  //   char * q = heap.allocate(128);
  //   memset(q, 0, 128);
  // }

#if 0
  char * c = (char*) p;
  cpu_time_t start = rdtsc();
  c[0] = 'a';
  cpu_time_t delta = rdtsc() - start;
  c[4096] = 'a';
  PLOG("touched! in %ld cycles (%f usec)", delta, common::cycles_to_usec(delta));
#endif

  nupm::free_virtual_pages(p);
}
#endif


#if 0
TEST_F(Libnupm_test, Arena)
{
  nupm::Arena_allocator_volatile allocator;

  std::chrono::system_clock::time_point start, end;
  PLOG("SingleThreadAlloc running...");

  const unsigned ITERATIONS = 3000;
  std::vector<void*> alloc_v;
  alloc_v.reserve(ITERATIONS);

  start = std::chrono::high_resolution_clock::now();
  for(unsigned i=0;i<ITERATIONS;i++) {
    void * p = allocator.alloc(i % 2);
    ASSERT_TRUE(p);
    alloc_v.push_back(p);
    //memset(p, 0xe, s);
    ((char*)p)[0] = 0xe;
  }
  end = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration<double>(end - start).count();
  PINF("Arena::SingleThreadAlloc: %lu allocs/sec", static_cast<unsigned long>(double(ITERATIONS) / secs));

  for(auto& a : alloc_v) {
    allocator.free(a);
  }
}
#endif

#if 0
TEST_F(Libnupm_test, Basic)
{
  nupm::Vmem_allocator allocator;

  void * ptr0 = allocator.alloc(0, MB(2));
  PLOG("ptr0: %p", ptr0);
  ASSERT_TRUE(ptr0);
  memset(ptr0, 0xe, MB(2));
  void * ptr1 = allocator.alloc(1, MB(2));
  ASSERT_TRUE(ptr1);
  PLOG("ptr1: %p", ptr1);
  memset(ptr1, 0xe, MB(2));

  allocator.free(ptr0);
  allocator.free(ptr1);
  PLOG("frees OK!");
}
#endif

#ifdef RUN_VMEM_ALLOCATOR_TESTS

TEST_F(Libnupm_test, VmemSingleThreadAlloc)
{
  std::chrono::system_clock::time_point start, end;
  nupm::Vmem_allocator                  allocator;
  PLOG("Vmem::SingleThreadAlloc running...");

  const unsigned      ITERATIONS = 10000000;
  std::vector<void *> alloc_v;
  alloc_v.reserve(ITERATIONS);

  start = std::chrono::high_resolution_clock::now();
  for (unsigned i = 0; i < ITERATIONS; i++) {
    size_t s = genrand64_int64() % 8096;
    void * p = allocator.alloc(i % 2 /* numa zone */, s);
    ASSERT_TRUE(p);
    alloc_v.push_back(p);
    // memset(p, 0xe, s);
    //    ((char*)p)[0] = 0xe;
  }
  end = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration<double>(end - start).count();
  PINF("Vmem::SingleThreadAlloc: %lu allocs/sec",
       static_cast<unsigned longi>(double(ITERATIONS) / secs);

  PLOG("Vmem::SingleThreadAlloc: Freeing allocotions...");
  for (auto &a : alloc_v) {
    allocator.free(a);
  }
}

#endif

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  if (argc > 1) {
    Options.uuid = std::stoul(argv[1]);
  }
  auto r = RUN_ALL_TESTS();

  return r;
}
