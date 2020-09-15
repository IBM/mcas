#include <api/ado_itf.h>
#include <api/components.h>
#include <common/str_utils.h>
#include <common/utils.h>
#include <common/logging.h>
#include <gtest/gtest.h>
#include <nupm/devdax_manager.h>
#include <nupm/mcas_mod.h>
#include <boost/program_options.hpp>

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
  }
  catch (const po::error &) {
    printf("bad command line option\n");
    return -1;
  }

  vector<nupm::Devdax_manager::config_t> conf;
  nupm::Devdax_manager::config_t c;
  c.path = "/dev/dax0.0";
  c.addr = 0x9000000000;
  c.region_id = 0;
  conf.push_back(c);
  nupm::Devdax_manager ddm(conf, true);

  nupm::revoke_memory(token);

  uint64_t *addr = reinterpret_cast<uint64_t *>
    (ddm.create_region(1234 /* uuid */, 0, size)); //size*8));

  auto count = size / sizeof(uint64_t);
  auto count_per_page = PAGE_SIZE /sizeof(uint64_t);

  /* write data */
  for(uint64_t i=0;i<count;i++) {
    if(i % count_per_page == 0) {
      if(i==0) addr[i] = 0;
      else addr[i] = i / count_per_page;
    }
    else {
      addr[i] = 0xAAAABBBBCCCCDDDD;
    }
  }

  /* read back and check */
  for(uint64_t i=0;i<count;i++) {
    if(i % count_per_page == 0) {
      if(i==0) {
	if(addr[i] != 0) {
	  PERR("marker error: addr[%lu]=%lu", i, addr[i]);
	  throw General_exception("bad data");
	}
      }
      else {
	if(addr[i] != (i / count_per_page)) {
	  PERR("marker error: addr[%lu]=%lu", i, addr[i]);
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

  status_t rc = nupm::expose_memory(token, addr, size);
  if(rc != S_OK) PLOG("Exposed memory rc=%d", rc);

  PLOG("Press return to continue...");
  getchar();

  PLOG("Starting check for changes...");

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
      if(addr[i] != 0x1111222233334444) {
	PERR("data error: addr[%lu]=%lx", i, addr[i]);
	throw General_exception("bad data");
      }
    }
  }

  PMAJOR("Verified OK!");
}


