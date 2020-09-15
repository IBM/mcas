#include <api/ado_itf.h>
#include <api/components.h>
#include <chrono>
#include <common/str_utils.h>
#include <common/utils.h>
#include <common/logging.h>
#include <gtest/gtest.h>
#include <nupm/devdax_manager.h>
#include <nupm/mcas_mod.h>
#include <boost/program_options.hpp>

#include <common/exceptions.h>
#include <common/str_utils.h>
#include <api/components.h>
#include <api/mcas_itf.h>

static void perform_RDMA_test(void * buffer, size_t buffer_len);

using namespace component;
using namespace common;
using namespace std;

#define PAGE_COUNT 1024 // 4MB of 4K pages

unsigned token = 1099;

int main(int argc, char **argv)
{
  namespace po = boost::program_options;

  po::variables_map vm;
  size_t size;

  try {
    po::options_description desc("Options");

    desc.add_options()("help", "Show help")
      ("size", po::value<unsigned>()->default_value(4096*PAGE_COUNT), "Size of memory in bytes")
      ;

    po::store(po::parse_command_line(argc, argv, desc), vm);

    size = vm["size"].as<unsigned>();
    PLOG("size: %lu", size);
  }
  catch (const po::error &) {
    printf("bad command line option\n");
    return -1;
  }

  size_t mapped_size = 0;
  uint64_t *addr = static_cast<uint64_t*>(nupm::mmap_exposed_memory(token, mapped_size, reinterpret_cast<void *>(0x900000000)));
  PLOG("mapped size: %lu", mapped_size);
  assert(addr);

  PLOG("mapped exposed memory OK (addr=%p)", reinterpret_cast<void*>(addr));

  auto count = mapped_size / sizeof(uint64_t);
  auto count_per_page = PAGE_SIZE /sizeof(uint64_t);

  /* read back and check */
  for(uint64_t i=0;i<count;i++) {
    if(i % count_per_page == 0) {
      if(i==0) {
	if(addr[i] != 0) {
	  PERR("marker error: addr[%lu]=%lx", i, addr[i]);
	  throw General_exception("bad data");
	}
      }
      else {
	if(addr[i] != (i / count_per_page)) {
	  PERR("marker error: addr[%lu]=%lx", i, addr[i]);
	  throw General_exception("bad data");
	}
      }
    }
    else {
      if(addr[i] != 0xAAAABBBBCCCCDDDD) {
	PERR("data error: addr[%lu]=%lx", i, addr[i]);
	throw General_exception("bad data");
      }
    }
  }

  PLOG("memory verified OK.");

  perform_RDMA_test(reinterpret_cast<void*>(addr), mapped_size);

  /* change data */
  for(uint64_t i=0;i<count;i++) {
    if(i % count_per_page == 0) {
      if(i==0) addr[i] = 0;
      else addr[i] = i / count_per_page;
    }
    else {
      addr[i] = 0x1111222233334444;
    }
  }

  PMAJOR("Done!");
}


static void perform_RDMA_test(void * buffer, size_t buffer_len)
{

    /* load component and create factory */
  IBase *comp = load_component("libcomponent-mcasclient.so", mcas_client_factory);
  auto factory = static_cast<IMCAS_factory *>(comp->query_interface(IMCAS_factory::iid()));
  assert(factory);

  /* create instance of MCAS client session */
  auto mcas = factory->mcas_create(1 /* debug level, 0=off */,
				   10, /* patience */
                                   getlogin(),
                                   "10.0.0.101:11911", /* MCAS server endpoint */
                                   "mlx5_0"); /* see mcas_client.h */

  factory->release_ref();
  assert(mcas);

  /* open existing pool or create one */
  const std::string poolname = "myBasicPool";
  auto pool = mcas->open_pool(poolname, 0);

  if (pool == IKVStore::POOL_ERROR) {
    /* ok, try to create pool instead */
    pool = mcas->create_pool(poolname, MB(32));
  }
  assert(pool != IKVStore::POOL_ERROR);

  auto key = common::random_string(8);
  std::string value = "This is my value " + common::random_string(8);

  component::IMCAS::memory_handle_t mr = mcas->register_direct_memory(buffer, buffer_len);
  assert(mr);

  /* add new item to pool */
  if(mcas->put_direct(pool,
		      key,
		      buffer,
		      buffer_len,
		      mr) != S_OK)
    throw General_exception("put_direct failed unexpectedly.");

  /* clear memory */
  memset(buffer, 0, buffer_len);

  /* re-read from store */
  if(mcas->get_direct(pool,
		      key,
		      buffer,
		      buffer_len,
		      mr) != S_OK)
    throw General_exception("put_direct failed unexpectedly.");

  mcas->unregister_direct_memory(mr);

  /* close pool */
  mcas->close_pool(pool);

  /* cleanup */
  mcas->release_ref();

}
