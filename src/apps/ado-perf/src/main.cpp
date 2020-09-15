#include <api/components.h>
#include <api/mcas_itf.h>
#include <common/cycles.h>
#include <common/str_utils.h>
#include <common/task.h>
#include <stdio.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

struct options {
  unsigned    debug_level;
  std::string server;
  std::string device;
  std::string poolname;
  std::string blastkey;
  std::uint16_t port;
  bool        async;
  bool        pause;
  std::string test;
  unsigned base_core;
  unsigned threads;
  unsigned patience;
} g_options{};



component::Itf_ref<component::IMCAS> init(const std::string& server_hostname, std::uint16_t port);
void do_throughput_work(component::IMCAS* mcas);
void do_blast_work(component::IMCAS* mcas, const std::string& blastkey, unsigned core = 0);


class Tasklet;

int main(int argc, char* argv[])
{
  namespace po = boost::program_options;

  try {
    po::options_description            desc("Options");
    po::positional_options_description g_pos; /* no positional options */

    desc.add_options()
      ("help", "Show help")
      ("server", po::value<std::string>()->default_value("10.0.0.21"),"Server hostname")
      ("device", po::value<std::string>()->default_value("mlx5_0"), "Device (e.g. mlnx5_0)")
      ("base-core", po::value<unsigned>()->default_value(0), "Base core")
      ("threads", po::value<unsigned>()->default_value(0), "Threads")
      ("poolname", po::value<std::string>()->default_value("adoperf_pool_default"), "Pool name")
      ("port", po::value<std::uint16_t>()->default_value(0), "Server port. Default 0 (mapped to 11911 for verbs, 11921 for sockets)")
      ("debug", po::value<unsigned>()->default_value(0), "Debug level")
      ("async", "Use asynchronous invocation")
      ("pause", "Pause after data set up")
      ("blastkey", po::value<std::string>(), "Do repeated invoke_ado on this key")
      ("test", po::value<std::string>()->default_value("put"), "Test to run (put, get, erase)")
      ("patience", po::value<unsigned>()->default_value(30), "Patience with werver (seconds)")
    ;

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(g_pos).run(), vm);

    if (vm.count("help") > 0) {
      std::cout << desc;
      return -1;
    }

    if (vm.count("server") == 0) {
      std::cout << "--server option is required\n";
      return -1;
    }

    if (vm.count("blastkey")) {
      g_options.blastkey  = vm["blastkey"].as<std::string>();
    }

    g_options.server      = vm["server"].as<std::string>();
    g_options.device      = vm["device"].as<std::string>();
    g_options.port        = vm["port"].as<std::uint16_t>();
    g_options.debug_level = vm["debug"].as<unsigned>();
    g_options.async       = vm.count("async");
    g_options.pause       = vm.count("pause");
    g_options.test        = vm["test"].as<std::string>();
    g_options.poolname    = vm["poolname"].as<std::string>();
    g_options.patience    = vm["patience"].as<unsigned>();
    g_options.threads     = vm["threads"].as<unsigned>();
    g_options.base_core   = vm["base-core"].as<unsigned>();

    if(g_options.threads == 0 || g_options.threads == 1) {
      PLOG("Using single-threaded process ..");
      auto mcasptr = init(g_options.server, g_options.port);
      
      if(g_options.blastkey.empty())
        do_throughput_work(&*mcasptr);
      else
        do_blast_work(&*mcasptr, g_options.blastkey);
    }
    else {
      PLOG("Using tasklet ..");
      cpu_mask_t mask;
      for (unsigned c = 0; c < g_options.threads; c++) mask.add_core(c + g_options.base_core);
      
      common::Per_core_tasking<Tasklet, int> t(mask,0); //,false,10000);
      t.wait_for_all();
    }
  }
  catch (const po::error &) {
    printf("bad command line option\n");
    return -1;
  }

  return 0;
}



class Tasklet : public common::Tasklet {
public:

  Tasklet(int) : _mcas(init(g_options.server, g_options.port))
  {
    PMAJOR("Tasklet: init complete");
  }
  
  virtual void initialize(unsigned  ) override
  {
  }

  virtual bool do_work(unsigned core) override
  {
    PMAJOR("Tasklet: starting work on core %u", core);

    if(g_options.blastkey.empty())
      do_throughput_work(&*_mcas);
    else
      do_blast_work(&*_mcas, g_options.blastkey, core);

    return false;
  }

  virtual void cleanup(unsigned) override
  {
  }

private:
  component::Itf_ref<component::IMCAS> _mcas;
};

component::Itf_ref<component::IMCAS> init(const std::string&, // server_hostname
                                          std::uint16_t // port
)
{
  using namespace component;

  IBase* comp = component::load_component("libcomponent-mcasclient.so", mcas_client_factory);

  auto fact = make_itf_ref(static_cast<IMCAS_factory*>(comp->query_interface(IMCAS_factory::iid())));
  if (!fact) throw Logic_exception("unable to create MCAS factory");

  std::stringstream url;
  url << g_options.server << ":" << g_options.port;

  auto mcas = make_itf_ref(fact->mcas_create(g_options.debug_level, g_options.patience, "None", url.str(), g_options.device));

  if (!mcas) throw Logic_exception("unable to create MCAS client instance");

  return mcas;
}

void do_blast_work(component::IMCAS* mcas, const std::string& blastkey, unsigned core)
{
  using namespace component;
  using clock     = std::chrono::high_resolution_clock;

  std::stringstream ss;
  
  // single pool shard across threads
  //  ss << g_options.poolname; // << "-" << core;
  //  const std::string poolname = ss.str();
  auto poolname = g_options.poolname;
  
  PMAJOR("Doing ADO blast...");
  PMAJOR("Creating pool (%s)...", poolname.c_str());
  
  auto pool = mcas->create_pool(poolname, GB(1), 0, /* flags */
                                1000000);           /* obj count */
  
  const unsigned iterations = 100000;

  if(pool == IMCAS::POOL_ERROR) throw General_exception("unable to create pool (%s)", poolname.c_str());

  PMAJOR("Open opened OK, pool (%s, blast-key=%s)...", poolname.c_str(), blastkey.c_str());

  ss << blastkey << "-" << core ; // different keys

  auto bk = ss.str();
  mcas->put(pool, bk, "BlastValue");

  while(1) {

    auto start_time = clock::now();
    
    /* perform invoke_ado repeatedly */
    std::vector<component::IMCAS::ADO_response> response;
    for(unsigned i=0;i<iterations;i++) {
      mcas->invoke_ado(pool, bk, "BLAST ME!", 0, response);
    }

    __sync_synchronize();

    auto secs = std::chrono::duration<double>(clock::now() - start_time).count();
    double per_sec = double(iterations) / secs; 
    PINF("Time: %.2f secs", secs);
    PINF("Rate (%u): %.0f /sec", core, per_sec);

  }


  mcas->close_pool(pool);
}

void do_throughput_work(component::IMCAS* mcas)
{
  using namespace component;

  const std::string poolname = g_options.poolname;

  PMAJOR("Creating pool (%s)...", poolname.c_str());
  auto pool = mcas->create_pool(poolname, GB(1), 0, /* flags */
                                1000000);           /* obj count */
  
  if (pool == IKVStore::POOL_ERROR) 
    throw General_exception("create_pool (%s) failed", poolname.c_str());

  std::vector<std::string> key_samples;
  std::vector<std::string> value_samples;

  PMAJOR("Setting up data...");
  unsigned       iterations  = 1000000;
  const unsigned num_strings = iterations;
  for (unsigned i = 0; i < num_strings; i++) {
    auto s = common::random_string(8);  //(rdtsc() % 32) + 8);
    key_samples.push_back(s);
  }

  for (unsigned i = 0; i < num_strings; i++) {
    auto s = common::random_string(16);  //(rdtsc() % 32) + 8);
    value_samples.push_back(s);
  }

  std::string                                 request;
  std::vector<component::IMCAS::ADO_response> response;

  auto flags = IMCAS::ADO_FLAG_CREATE_ON_DEMAND;
  if (g_options.async) {
    flags |= IMCAS::ADO_FLAG_ASYNC;
  }

  if(g_options.pause) {
    PMAJOR("Press return key to start.");
    getchar();
  }

  PMAJOR("Starting invoke_put_ado sequence (pool=%s) ...", poolname.c_str());

  using clock     = std::chrono::high_resolution_clock;
  auto start_time = clock::now();

  for (unsigned i = 0; i < iterations; i++) {
    if(S_OK != mcas->invoke_put_ado(pool, key_samples[i], "put-" + std::to_string(i), value_samples[i], 0, flags, response))
      throw General_exception("invoke_put_ado failed");
  }

  __sync_synchronize();

  auto secs = std::chrono::duration<double>(clock::now() - start_time).count();

  double per_sec = double(iterations) / secs;
  PINF("Synchronous ADO invoke_put_ado RTT");
  PINF("Time: %.2f sec", secs);
  PINF("Rate: %.0f /sec", per_sec);

  if(g_options.pause) {
      PMAJOR("Press return key to start.");
      getchar();
  }

  PMAJOR("Starting invoke_ado ...");

  start_time = clock::now();

  for (unsigned i = 0; i < iterations; i++) {
    if(S_OK != mcas->invoke_ado(pool, key_samples[i], "put-" + std::to_string(i), flags, response))
      throw General_exception("invoke_ado failed");
  }

  __sync_synchronize();

  secs = std::chrono::duration<double>(clock::now() - start_time).count();
  per_sec = double(iterations) / secs;

  PINF("Synchronous ADO invoke_ado RTT");
  PINF("Time: %.2f sec", secs);
  PINF("Rate: %.0f /sec", per_sec);


  if (g_options.test == "get") {
    start_time = clock::now();
    for (unsigned i = 0; i < iterations; i++) {
      mcas->get(pool, key_samples[i], value_samples[i]);
    }
    __sync_synchronize();

    secs = std::chrono::duration<double>(clock::now() - start_time).count();

    per_sec = double(iterations) / secs;
    PINF("Synchronous ADO RTT");
    PINF("Time: %.2f sec", secs);
    PINF("Rate: %.0f /sec", per_sec);
  }
  else if (g_options.test == "erase") {
    start_time = clock::now();
    for (unsigned i = 0; i < iterations; i++) {
      mcas->invoke_ado(pool, key_samples[i], "erase", flags, response);
    }
    __sync_synchronize();
    secs = std::chrono::duration<double>(clock::now() - start_time).count();
    PLOG("now calculating erase throghput....");
  }

  //mcas->delete_pool(pool);
}
