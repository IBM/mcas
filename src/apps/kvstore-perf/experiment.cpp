#include "experiment.h"

#include "data.h"
#include "get_vector_from_string.h"
#include "program_options.h"

#include "common/json.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

#include <boost/filesystem.hpp>
#if defined(__powerpc64__)
#include <gperftools/heap-profiler.h>
#else
#include <gperftools/profiler.h>
#endif
#include <sys/mman.h> /* madvise, MADV_HUGEPAGE */
#include <sys/sysmacros.h> /* major, minor */

#include <algorithm> /* transform - should follow local includes */
#include <cctype> /* isdigit */
#include <cstddef>
#include <cmath>
#include <exception>
#include <fstream> /* ifstream, ofstream */
#include <iostream> /* cerr */
#include <stdexcept> /* logic_error, domain_error */
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define MCAS_PATH "libcomponent-mcasclient.so"

#define HT_SIZE_FACTOR 1 /* already factor 3 in hstore */

Data * Experiment::g_data;
std::mutex Experiment::g_write_lock;
double Experiment::g_iops;

namespace
{
  /* copied from nd_utils.h, which is not accessible to the build */
  std::size_t get_dax_device_size(const std::string& dax_path)
  {
    int fd = ::open(dax_path.c_str(), O_RDWR, 0666);
    if (fd == -1)
    {
      auto e = errno;
      throw General_exception("inaccessible devdax path (%s): %s", dax_path.c_str(), strerror(e));
    }

    struct ::stat statbuf;
    if ( ::fstat(fd, &statbuf) == -1 )
    {
      auto e = errno;
      throw General_exception("fstat(%s) failed:  %s", dax_path.c_str(), strerror(e));
    }

    char spath[PATH_MAX];
    snprintf(spath, PATH_MAX, "/sys/dev/char/%u:%u/size",
      major(statbuf.st_rdev), minor(statbuf.st_rdev)
    );
    std::ifstream sizestr(spath);
    size_t size = 0;
    sizestr >> size;
    return size;
  }

  std::size_t round_up_to_pow2(std::size_t n)
  {
    /* while more than one bit is set in n ... */
    while ( ( n & (n-1) ) != 0 )
    {
      /* increment by the rightmost 1 bit */
      n += ( n & (-n) );
    }
    /* at most one bit is set in the result */
    return n;
  }

  // returns file size in bytes
  long GetFileSize(std::string filename)
  {
    struct ::stat stat_buf;

    auto rc = stat(filename.c_str(), &stat_buf);

    return rc == 0 ? stat_buf.st_size : -1;
  }

  long GetBlockSize(std::string path)
  {
    struct ::stat stat_buf;

    auto rc = stat(path.c_str(), &stat_buf);

    return rc == 0 ? stat_buf.st_blksize : -1;
  }

  std::string get_time_string()
  {
    time_t rawtime;
    ::tm *timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    char buffer[80];

    // want YYYY_MM_DD_HH_MM format
    strftime(buffer, sizeof(buffer), "%Y_%m_%d_%H_%M", timeinfo);
    std::string timestring(buffer);

    return timestring;
  }
}

Experiment::Experiment(std::string name_, const ProgramOptions &options)
  : common::log_source(options.debug_level)
  , _pool_path(options.path ? *options.path : "./data/")
  , _pool_name(options.pool_name)
  , _pool_name_local()
  , _owner(options.owner)
  , _pool_size(options.size)
  , _pool_flags(options.flags)
  , _pool_num_objects(options.elements)
  , _cores(options.cores)
  , _devices(options.devices)
  , _core_list()
  , _start_time(options.start_time)
  , _duration_directed(options.duration ? std::chrono::duration_cast<std::chrono::high_resolution_clock::duration>(std::chrono::seconds(*options.duration)) : boost::optional<std::chrono::high_resolution_clock::duration>())
  , _end_time_directed()
  , _component(options.component)
  , _results_path("./results")
  , _report_filename(options.report_file_name)
  , _do_json_reporting(options.do_json_reporting)
  , _test_name(name_)
  , _store()
  , _pool(component::IKVStore::POOL_ERROR)
  , _server_address(options.server_address)
  , _provider(options.provider)
  , _port(options.port)
  , _port_increment(options.port_increment ? *options.port_increment : 0)
  , _device_name(options.device_name)
  , _src_addr(options.src_addr)
  , _pci_address(options.pci_addr)
  , _first_iter(true)
  , _ready(false)
  , timer()
  , _memory_handle(debug_level())
  , _verbose(options.verbose)
  , _summary(options.summary)
  , _element_size_on_disk(std::size_t(-1))
  , _element_size(std::size_t(-1))
  , _elements_in_use(0)
  , _pool_element_start(0)
  , _pool_element_end(0)
  , _elements_stored(0)
  , _total_data_processed(0)
  , _bin_count(options.bin_count)
  , _bin_threshold_min(options.bin_threshold_min)
  , _bin_threshold_max(options.bin_threshold_max)
  , _core_to_device_map(make_core_to_device_map(_cores, _devices))
{
}

Experiment::~Experiment()
{
}

void Experiment::initialize_custom(unsigned /* core */)
{
  // does nothing by itself; put per-experiment initialization functions here
  std::cerr << "no initialize_custom function used\n";
}

void Experiment::cleanup_custom(unsigned /* core */)
{
  // does nothing by itself; put per-experiment cleanup functions in its place
  std::cerr << "no cleanup_custom function used\n";
}

void Experiment::_print_highest_count_bin(BinStatistics& stats, unsigned core)
{
  if ( _summary )
  {
    stats.print_highest_count_bin(std::cerr, core);
  }
}

std::size_t Experiment::GetDataInputSize(std::size_t index)
{
  std::string value = g_data->value(index);

  return value.size();
}

auto Experiment::make_core_to_device_map(const std::string &cores, const std::string &devices) -> core_to_device_map_t
{
  auto core_v = get_vector_from_string<unsigned>(cores);
  auto device_v = get_vector_from_string<dotted_pair<unsigned>>(devices);
  if ( device_v.size() < core_v.size() )
  {
    throw
      std::domain_error(
        "core list '" + cores
        + "' length " + std::to_string(core_v.size())
        + " is longer than device list '" + devices
        + "' length " + std::to_string(device_v.size())
      );
  }

  core_to_device_map_t core_to_device_map;
  std::transform(
    core_v.begin()
    , core_v.end()
    , device_v.begin()
    , std::inserter(
      core_to_device_map
      , core_to_device_map.end())
    , [] (unsigned c, dotted_pair<unsigned> d) { return std::make_pair(c, d); }
  );
  return core_to_device_map;
}

int Experiment::initialize_store(unsigned core)
{
  using namespace component;

  IBase * comp;

  try
  {
    if( component_is( "mcas" ) ) {

      DECLARE_STATIC_COMPONENT_UUID(mcas_factory, 0xfac66078,0xcb8a,0x4724,0xa454,0xd1,0xd8,0x8d,0xe2,0xdb,0x87);  // TODO: find a better way to register arbitrary components to promote modular use
      comp = load_component(MCAS_PATH, mcas_factory);
    }
    else if ( component_is( "hstore" ) ) {
      comp = load_component("libcomponent-hstore.so", hstore_factory);
    }
    else if ( component_is( "mapstore" ) ) {
      comp = load_component("libcomponent-mapstore.so", mapstore_factory);
    }
    else throw General_exception("unknown --component option (%s)", _component.c_str());
  }
  catch ( const Exception &e )
  {
    PERR("error during load_component: %s. Aborting experiment.", e.cause());
    throw;
  }
  catch ( const std::exception &e )
  {
    PERR("error during load_component: %s. Aborting experiment.", e.what());
    throw;
  }
  catch(...)
  {
    PERR("%s", "error during load_component.");
    return 1;
  }

  if (_verbose)
    {
      PINF("[%u] component address: %p", core, static_cast<const void *>(&comp));
    }

  if (!comp)
  {
    PERR("%s", "comp loaded, but returned invalid value");
    return 1;
  }

  try
  {
    auto fact = make_itf_ref(static_cast<IKVStore_factory *>(comp->query_interface(IKVStore_factory::iid())));

    if ( component_is( "mcas" ) ) {
      auto port = _port;
      if ( _port_increment )
      {
        port += (_get_core_index(core) / _port_increment);
      }
      auto server_url = _server_address + ":" + std::to_string(port);
      PLOG("(%d) server url: (%s)", _get_core_index(core), server_url.c_str());
      IKVStore_factory::map_create mc
        {
          { +IKVStore_factory::k_dest_addr, _server_address }
          , { +IKVStore_factory::k_dest_port, std::to_string(port) }
        };
      if ( _src_addr )
      {
        mc.insert(IKVStore_factory::map_create::value_type(+IKVStore_factory::k_src_addr, *_src_addr));
      }
      if ( _device_name )
      {
        mc.insert(IKVStore_factory::map_create::value_type(+IKVStore_factory::k_interface, *_device_name));
      }
      if ( _provider )
      {
        mc.insert(IKVStore_factory::map_create::value_type(+IKVStore_factory::k_provider, *_provider));
      }

      _store.reset(fact->create(debug_level(), mc));

      PMAJOR("mcas component instance: %p", static_cast<const void *>(_store.get()));
    }
    else if ( component_is( "hstore" ) || component_is("dummystore") ) {
      auto device = core_to_device(core);
      std::size_t dax_base = 0x7000000000;
      /* at least the dax size, rounded for alignment */

      if ( ! _device_name )
      {
        throw std::runtime_error("Component " + _component + " has no device_name");
      }
      std::size_t dax_stride = round_up_to_pow2(dev_dax_max_size(*_device_name));
      std::size_t dax_node_stride = round_up_to_pow2(dev_dax_max_count_per_node()) * dax_stride;

      unsigned region_id = 0;
      /* stride ignores dax "major" number, so /dev/dax0.n and /dev/dax1.n map to the same memory */
      std::uint64_t addr = dax_base + dax_node_stride * device.first + dax_stride * device.second;
#if 0
      std::ostringstream addr_o;
      addr_o << std::showbase << std::hex << addr;
#endif
      std::ostringstream device_full_name;
      device_full_name << *_device_name << (std::isdigit(_device_name->back()) ? "." : "") << device;
      namespace c_json = common::json;
      using json = c_json::serializer<c_json::dummy_writer>;

      auto device_map = json::array(
        json::object(
          json::member("region_id", region_id)
          , json::member("path", device_full_name.str())
#if 0
          , json::member("addr", addr_o.str())
#else
          , json::member("addr", addr)
#endif
        )
      );

      _store.reset(
        fact->create(
          debug_level()
          , {
            { +component::IKVStore_factory::k_owner, _owner}
            , { +component::IKVStore_factory::k_dax_config, device_map.str() }
          }
        )
      );
    }
    else {
      _store.reset(fact->create(0, {}));
    }
  }
  catch ( const Exception &e )
  {
    PERR("factory creation step failed, aborting experiment: %s", e.cause());
    throw;
  }
  catch ( const std::exception &e )
  {
    PERR("factory creation step failed, aborting experiment: %s", e.what());
    throw;
  }
  catch(...)
    {
      PERR("%s", "factory creation step failed");
      return 1;
    }

  return 0;
}

void Experiment::initialize(unsigned core)
try
{
  if ( auto rc = initialize_store(core) )
  {
    auto e = "initialize returned error " + std::to_string(rc);
    PERR("%s", e.c_str());
    throw std::runtime_error(e);
  }

  try
  {
    if ( component_uses_direct_memory() )
    {
      _memory_handle = direct_memory_registered_kv(debug_level(), _store.get(), g_data->value_space, g_data->value_space_size);
    }
  }
  catch ( const Exception &e ) {
    PERR("failed during direct memory setup: %s.", e.cause());
    throw;
  }
  catch ( const std::exception &e ) {
    PERR("failed during direct memory setup: %s.", e.what());
    throw;
  }
  catch(...) {
    PERR("%s", "failed during direct memory setup");
    throw;
  }

  // make sure path is available for use
  boost::filesystem::path dir(_pool_path);
  if (boost::filesystem::create_directory(dir) && _verbose) {
    std::cout << "Created directory for testing: " << _pool_path << std::endl;
  }

  // initialize experiment
  auto core_index = _get_core_index(core);
  std::string poolname = _pool_name + "." + std::to_string(core_index);
  _pool_name_local = std::string(poolname);
  auto path = _pool_path + poolname;

  if ( ! component_is("mcas") && boost::filesystem::exists(path)) {
    bool might_be_dax = boost::filesystem::exists(path);
    try
    {
      // pool already exists. Delete it.
      if (_verbose)
      {
        std::cout << "pool might already exist at " << path << ". Attempting to delete it...";
      }

      _store->delete_pool(_pool_path + poolname);

      if (_verbose)
      {
        std::cout << " pool deleted!" << std::endl;
      }
    }
    catch ( const Exception &e )
    {
      if ( ! might_be_dax )
      {
        PERR("open+delete existing pool %s failed: %s.", poolname.c_str(), e.cause());
      }
    }
    catch ( const std::exception &e )
    {
      if ( ! might_be_dax )
      {
        PERR("open+delete existing pool %s failed: %s.", poolname.c_str(), e.what());
      }
    }
    catch(...)
    {
      if ( ! might_be_dax )
      {
        PERR("open+delete existing pool %s failed.", poolname.c_str());
        std::cerr << "open+delete existing pool failed" << std::endl;
      }
    }
  }

  PLOG("Creating pool %s for worker %u ...", _pool_name_local.c_str(), core);
  try
  {
    _pool = _store->create_pool(_pool_path + _pool_name_local, _pool_size, _pool_flags, _pool_num_objects * HT_SIZE_FACTOR);
    if ( _pool == component::IKVStore::POOL_ERROR )
    {
      PERR("create_pool failed");
      throw std::runtime_error("create_pool failed: returned IKVStore::POOL_ERROR. Aborting experiment.");
    }
  }
  catch ( const Exception &e )
  {
    PERR("create_pool failed: %s. Aborting experiment.", e.cause());
    throw;
  }
  catch ( const std::exception &e )
  {
    PERR("create_pool failed: %s. Aborting experiment.", e.what());
    throw;
  }
  catch(...)
  {
    PERR("%s", "create_pool failed! Aborting experiment.");
    throw;
  }

  PLOG("Created pool for worker %u...OK!", core);

  {
    std::lock_guard<std::mutex> g(g_write_lock);
    if ( is_json_reporting() )
    {
      // initialize experiment report

      rapidjson::Document document = _get_report_document();
      if (!document.HasMember("experiment"))
      {
        _initialize_experiment_report(document);
      }
    }

    g_iops = 0;
  }

  try
  {
    initialize_custom(core);
  }
  catch ( const Exception &e )
  {
    PERR("initialize_custom failed: %s. Aborting experiment.", e.cause());
    throw;
  }
  catch ( const std::exception &e )
  {
    PERR("initialize_custom failed: %s. Aborting experiment.", e.what());
    throw;
  }
  catch ( ... )
  {
    PERR("initialize_custom failed: %s. Aborting experiment.", "(non-exception)");
    throw;
  }

#ifdef PROFILE
  ProfilerRegisterThread();
#endif
  _ready = true;
}
catch ( ... )
{
  _ready = true;
  throw;
}

// if experiment should be delayed, stop here and wait. Otherwise, start immediately
void Experiment::wait_for_delayed_start(unsigned core)
{
  if ( _start_time )
  {
    std::this_thread::sleep_until(*_start_time);
    PINF("[%u] starting experiment now", core);
  }
}

/* maximun size of any dax device. Limitation: considers only dax devices specified in the device string */
std::size_t Experiment::dev_dax_max_size(const std::string & dev_dax_prefix_)
{
  std::size_t size = 0;
  for ( auto & it : _core_to_device_map )
  {
    std::ostringstream s;
    s << dev_dax_prefix_ << it.second;
    size = std::max(size, get_dax_device_size(s.str()));
  }
  return size;
}

/* maximun number of dax devices in any numa node.
 * Limitations:
 *   Considers only dax devices specified in the device string.
 */
unsigned Experiment::dev_dax_max_count_per_node()
{
  std::map<unsigned, unsigned> dev_count {};
  for ( auto & it : _core_to_device_map )
  {
    auto &c = dev_count[it.first];
    c = std::max(c, it.second.second + 1);
  }

  unsigned size = 0;
  for ( auto & it : dev_count )
  {
    size = std::max(size, it.second);
  }

  return size;
}

dotted_pair<unsigned> Experiment::core_to_device(unsigned core)
{
  auto core_it = _core_to_device_map.find(core);
  if ( core_it == _core_to_device_map.end() )
  {
    throw std::logic_error("no core " + std::to_string(core_it->first) + " in map");
  }
  return core_it->second;
}

void Experiment::_update_aggregate_iops(double iops)
{
  g_iops += iops;

  if (_verbose)
    {
      PLOG("_update_aggregate_iops done. Currently = %lu", static_cast<unsigned long>(g_iops)) ;
    }
}

void Experiment::summarize()
{
  PINF("[TOTAL] %s %s IOPS: %lu", _cores.c_str(), _test_name.c_str(), static_cast<unsigned long>(g_iops));
}

void Experiment::cleanup(unsigned core) noexcept
try
{
  try
  {
    cleanup_custom(core);
  }
  catch ( const Exception &e )
  {
    PERR("cleanup_custom failed: %s. Aborting experiment.", e.cause());
    throw;
  }
  catch ( const std::exception &e )
  {
    PERR("cleanup_custom failed: %s. Aborting experiment.", e.what());
    throw;
  }
  catch(...)
  {
      PERR("%s", "cleanup_custom failed!");
    throw;
  }

  if ( _pool != component::IKVStore::POOL_ERROR )
  {
    try
    {
      auto rc = _store->close_pool(_pool);

      if (rc != S_OK)
      {
        PERR("close_pool returned error code %d", rc);
        throw std::runtime_error("close_pool returned error code " + std::to_string(rc));
      }
    }
    catch(...)
    {
      PERR("%s", "close_pool failed!");
      throw;
    }
  }
}
catch ( ... )
{
PERR("cleanup of core %u was incomplete.", core);
}

void Experiment::_debug_print(unsigned core, std::string text, bool limit_to_core0)
{
  if (_verbose)
  {
    if ( ( ! limit_to_core0 ) || core == 0 )
    {
      std::cout << "[" << core << "]: " << text << std::endl;
    }
  }
}

unsigned Experiment::_get_core_index(unsigned core)
{
  int core_int = int(core);

  if (_core_list.empty())
  {
    // construct list
    _core_list = get_vector_from_string<unsigned>(_cores);
  }

  // this is inefficient, but number of cores should be relatively small (hundreds at most)
  auto index = std::find(_core_list.begin(), _core_list.end(), core_int);

  if (index == _core_list.end())
  {
    auto e = "_get_core_index couldn't find core " + std::to_string(core_int);
    PERR("%s! Exiting.", e.c_str());
    throw std::runtime_error(e);
  }

  return unsigned(index - _core_list.begin());
}

rapidjson::Document Experiment::_get_report_document()
{
  rapidjson::Document document;

  if (_report_filename.empty())
  {
    auto e = "filename for report is empty";
    PERR("%s!", e);
    throw std::runtime_error(e);
  }

  try
  {
    std::unique_ptr<FILE, int (*)(FILE *)> pFile(::fopen(_report_filename.c_str(), "r"), ::fclose);
    if ( ! pFile )
    {
      auto er = errno;
      auto e = std::string("get_report_document failed fopen call opening '") + _report_filename + "'";
      PERR("%s: %s", e.c_str(), std::strerror(er));
      throw std::system_error(std::error_code(er, std::system_category()), e);
    }

    auto file_size = GetFileSize(_report_filename);

    /* error indication is silently ignored */
    auto buffer_size = file_size == -1 ? 0 : std::size_t(file_size);

    const size_t MIN_READ_BUFFER_SIZE = 4;  // if FileReadStream has a buffer smaller than this, it'll assert
    buffer_size = std::max(buffer_size, MIN_READ_BUFFER_SIZE);

    std::vector<char> readBuffer(buffer_size);

    rapidjson::FileReadStream is(pFile.get(), &readBuffer[0], buffer_size);
    document.ParseStream<0>(is);

    if (document.HasParseError())
    {
      auto e = " parsing error in document, code = " + std::to_string(document.GetParseError());
      PERR("%s", e.c_str());
      throw std::runtime_error(e);
    }
  }
  catch(...)
  {
    PERR("%s", "failed while reading in existing json document");
    throw;
  }

  _debug_print(0, "returning report document");

  return document;
}

void Experiment::_initialize_experiment_report(rapidjson::Document& document)
{
  if (_verbose)
  {
    PINF("%s", "writing experiment parameters to file");
  }
  rapidjson::Document::AllocatorType &allocator = document.GetAllocator();

  rapidjson::Value temp_object(rapidjson::kObjectType);

  // experiment parameters
  temp_object
    .AddMember("component", rapidjson::StringRef(_component.c_str()), allocator)
    .AddMember("cores", rapidjson::StringRef(_cores.c_str()), allocator)
    .AddMember("key_length", int(g_data->key_len()), allocator)
    .AddMember("value_length", int(g_data->value_len()), allocator)
    .AddMember("elements", int(_pool_num_objects), allocator)
    .AddMember("pool_size", double(_pool_size), allocator)
    .AddMember("pool_flags", _pool_flags, allocator)
    ;

  // first experiment could take some time; parse out start time from the filename we're using
  const std::string REPORT_NAME_START = "results_";
  const std::string REPORT_NAME_END = ".json";
  auto time_start = _report_filename.find(REPORT_NAME_START);
  auto time_end = _report_filename.find(REPORT_NAME_END);
  std::string timestring = _report_filename.substr(time_start + REPORT_NAME_START.length(), time_end - time_start - REPORT_NAME_START.length());

  temp_object.AddMember("date", rapidjson::StringRef(timestring.c_str()), allocator);

  document.AddMember("experiment", temp_object, allocator);

  rapidjson::StringBuffer strbuf;

  try
  {
    // write back to file
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(strbuf);
    document.Accept(writer);
  }
  catch(...)
  {
    PERR("%s", "failed during write to json document");
  }

  try
  {
    std::ofstream outf(_report_filename.c_str());
    outf << strbuf.GetString() << std::endl;
  }
  catch(...)
  {
    PERR("%s", "failed while writing to ofstream");
    throw;
  }
}

void Experiment::_report_document_save(rapidjson::Document& document, unsigned core, rapidjson::Value& new_info)
{
  _debug_print(core, "_report_document_save started");

  if (_test_name.empty())
  {
    auto e = "_test_name is empty";
    PERR("%s!", e);
    throw std::logic_error(e);
  }

  rapidjson::Value temp_object(rapidjson::kObjectType);
  rapidjson::StringBuffer strbuf;

  std::string core_string = std::to_string(core);
  rapidjson::Value core_value(rapidjson::StringRef(core_string.c_str()));

  try
  {
    if (document.IsObject() && !document.HasMember(_test_name.c_str()))
    {
      temp_object.AddMember(core_value, new_info, document.GetAllocator());
      document.AddMember(rapidjson::StringRef(_test_name.c_str()), temp_object, document.GetAllocator());
    }
    else
    {
      rapidjson::Value &items = document[_test_name.c_str()];

      items.AddMember(core_value, new_info, document.GetAllocator());
    }

    // write back to file
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(strbuf);
    document.Accept(writer);
  }
  catch(...)
  {
    PERR("%s", "failed during write to json document");
  }

  _debug_print(core, "_report_document_save: writing to ofstream");
  try
  {
    std::ofstream outf(_report_filename.c_str());
    outf << strbuf.GetString() << std::endl;
  }
  catch(...)
  {
    PERR("%s", "failed while writing to ofstream");
    throw;
  }

  _debug_print(core, "_report_document_save finished");
}

rapidjson::Value Experiment::_add_statistics_to_report(BinStatistics& stats, rapidjson::Document& document)
{
  rapidjson::Value bin_info(rapidjson::kObjectType);
  rapidjson::Value temp_array(rapidjson::kArrayType);

  // latency bin info
  bin_info
    .AddMember("bin_count", int(stats.getBinCount()), document.GetAllocator())
    .AddMember("threshold_min", double(stats.getMinThreshold()), document.GetAllocator())
    .AddMember("threshold_max", double(stats.getMaxThreshold()), document.GetAllocator())
    .AddMember("increment", double(stats.getIncrement()), document.GetAllocator())
    ;

  for (unsigned i = 0; i < stats.getBinCount(); i++)
  {
    // PushBack requires unique object
    rapidjson::Value temp_object(rapidjson::kObjectType);

    temp_object
      .AddMember("count", double(stats.getBin(i).getCount()), document.GetAllocator())
      .AddMember("min", double(stats.getBin(i).getMin()), document.GetAllocator())
      .AddMember("max", double(stats.getBin(i).getMax()), document.GetAllocator())
      .AddMember("mean", double(stats.getBin(i).getMean()), document.GetAllocator())
      .AddMember("std", double(stats.getBin(i).getStd()), document.GetAllocator())
      ;

    temp_array.PushBack(temp_object, document.GetAllocator());
  }

  // add new info to report
  rapidjson::Value bin_object(rapidjson::kObjectType);

  bin_object
    .AddMember("info", bin_info, document.GetAllocator())
    .AddMember("bins", temp_array, document.GetAllocator())
    ;

  return bin_object;
}

BinStatistics Experiment::_compute_bin_statistics_from_vectors(std::vector<double> data, std::vector<double> data_bins, unsigned bin_count, double bin_min, double bin_max, std::size_t elements)
{
  if (data.size() != data_bins.size())
  {
    PERR("data size %lu and data_bins size %lu aren't the same!", data.size(), data_bins.size());
  }

  BinStatistics stats(bin_count, bin_min, bin_max);

  for (std::size_t i = 0; i < elements; i++)
  {
    stats.update_value_for_bin(data[i], data_bins[i]);
  }

  return stats;
}

BinStatistics Experiment::_compute_bin_statistics_from_vector(std::vector<double> data, unsigned bin_count, double bin_min, double bin_max)
{
  BinStatistics stats(bin_count, bin_min, bin_max);

  for ( const auto &d : data )
  {
    stats.update(d);
  }

  return stats;
}

  /* create_report: output a report in JSON format with experiment data
   * Report format:
   *      experiment object - contains experiment parameters
   *      data object - actual results
   */
std::string Experiment::create_report(const std::string component_)
{
  PLOG("%s", "creating JSON report");
  std::string timestring = get_time_string();

  // create json document/object
  rapidjson::Document document;
  document.SetObject();

  // write to file
  std::string results_path = "./results";
  boost::filesystem::path dir(results_path);
  if (boost::filesystem::create_directory(dir))
  {
    std::cout << "Created directory for testing: " << results_path << std::endl;
  }

  std::string specific_results_path = results_path + "/" + component_;

  boost::filesystem::path sub_dir(specific_results_path);
  if (boost::filesystem::create_directory(sub_dir))
  {
    std::cout << "Created directory for testing: " << specific_results_path << std::endl;
  }

  rapidjson::StringBuffer sb;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(sb);
  document.Accept(writer);

  std::string output_file_name = specific_results_path + "/results_" + timestring + ".json";
  std::ofstream outf(output_file_name);

  if ( outf )
  {
    outf << sb.GetString() << std::endl;
    PLOG("created report with filename '%s'", output_file_name.c_str());
  }
  else
  {
    PERR("couldn't open report file %s to write.", output_file_name.c_str());
  }

  return output_file_name;
}
unsigned long Experiment::GetElementSize(unsigned core, std::size_t index)
{
  if ( _element_size == 0 || _element_size == std::size_t(-1) )
  {
    std::string path = _pool_path + "/" +  _pool_name + "." + std::to_string(core) + "/" + g_data->key(index);
    auto file_size = GetFileSize(path);

    if ( file_size == -1 )  // this means GetFileSize failed, maybe due to RDMA
    {
      _element_size = GetDataInputSize(index);
    }
    else
    {
      _element_size = std::size_t(file_size);
    }
  }

  if ( _element_size_on_disk == std::size_t(-1) )  // -1 is reserved value and impossible (must be positive size)
  {
    auto block_size = GetBlockSize(_pool_path);

    // take the larger of the two
    if (_element_size > std::size_t(block_size) || component_uses_rdma())
    {
      _element_size_on_disk = _element_size;
    }
    else
    {
      _element_size_on_disk = std::size_t(block_size);
    }
  }

  return _element_size_on_disk;
}

void Experiment::_update_data_process_amount(unsigned core, std::size_t index)
{
  if (_element_size == std::size_t(-1))  // -1 is reserved and impossible
  {
    std::string path = _pool_path + "/" +  _pool_name + "." + std::to_string(core) + "/" + g_data->key(index);
    auto file_size = GetFileSize(path);

    if ( file_size == -1 )  // this means GetFileSize failed, maybe due to RDMA
    {
      _element_size = g_data->value_len();
      //      _element_size = GetDataInputSize(index);
    }
    else
    {
      _element_size = std::size_t(file_size);
    }
  }
  assert(_element_size > 0);
  _total_data_processed += _element_size;
}

// throughput = Mib/s here
double Experiment::_calculate_current_throughput()
{
  if (_verbose)
    {
      PINF("throughput calculation: %lu data (element size %zd)", _total_data_processed, _element_size);
    }

  double size_mb = double(_total_data_processed) * 0.000001;  // bytes -> MB
  double time = timer.get_time_in_seconds();
  double throughput = size_mb / time;

  return throughput;
}

void Experiment::_populate_pool_to_capacity(unsigned core, component::IKVStore::memory_handle_t memory_handle)
{
  // how much space do we have?
  if (_verbose)
  {
    std::cout << "_populate_pool_to_capacity start: _pool_num_components = " << _pool_num_objects << ", _elements_stored = " << _elements_stored << ", _pool_element_end = " << _pool_element_end << std::endl;
  }

  auto current = _pool_element_end;  // first run: should be 0 (start index)
  long maximum_elements = -1;
  _pool_element_start = current;

  if (_verbose)
  {
    std::stringstream debug_start;
    debug_start << "current = " << current << ", end = " << _pool_element_end;
    _debug_print(core, debug_start.str());
  }

  bool can_add_more_in_batch = (current - _pool_element_start) != maximum_elements;
  bool can_add_more_overall = std::size_t(current) != _pool_num_objects;

  _element_size = g_data->value_len();
  void * readback_buffer = aligned_alloc(PAGE_SIZE,
                                         round_up_page(_element_size));
  if ( readback_buffer == nullptr )
  {
    throw std::bad_alloc();
  }
  auto readback_memory_handle = direct_memory_registered_kv(debug_level(), _store.get(), readback_buffer, _element_size);

  /* NOTE: put_direct handles long data, but does nothing for long keys.
   */
  while ( can_add_more_in_batch && can_add_more_overall )
  {
    try
    {
      assert(g_data->value_len() > 0);
      auto current_sz = std::size_t(current);
      // auto write_memory_handle = direct_memory_registered_kv(_store.get(), const_cast<void *>(g_data->value(current_sz)), g_data->value_len());
      int rc =
        memory_handle == component::IKVStore::HANDLE_NONE
        ? _store->put(_pool, g_data->key(current_sz), g_data->value(current_sz), g_data->value_len())
        : _store->put_direct(_pool, g_data->key(current_sz), g_data->value(current_sz), g_data->value_len(), memory_handle)
        ;

      if (rc != S_OK)
      {
        std::ostringstream e;
        e << "put or put_direct returned " << rc << " (-55 is out of space)";
        PERR("%s: %s.", __func__, e.str().c_str());
        throw std::runtime_error(e.str());
      }

      /* read back elements that have been put - this will make sure they are register on
         the server side when mapstore is being used. */
      size_t s = _element_size;
      rc = memory_handle == component::IKVStore::HANDLE_NONE
        ? _store->get(_pool, g_data->key(current_sz), readback_buffer, s)
        : _store->get_direct(_pool, g_data->key(current_sz), readback_buffer, s, readback_memory_handle.mr());
      ;

      if (rc != S_OK)
      {
        std::ostringstream e;
        e << "readback get or get_direct returned " << rc << " (-55 is out of space)";
        PERR("%s: %s.", __func__, e.str().c_str());
        throw std::runtime_error(e.str());
      }

      ++_elements_stored;
    }
    catch ( const std::exception &e )
    {
      std::ostringstream s;
      s << __func__ << ": " << "failed at put call " << current << "/" << _pool_num_objects << ": " << e.what();
      PERR("%s", s.str().c_str());
      throw;
    }
    catch(...)
    {
      std::ostringstream s;
      s << "populate_pool_to_capacity failed at put call " << current << "/" << _pool_num_objects;
      PERR("%s", s.str().c_str());
      throw;
    }

    // calculate maximum number of elements we can put in pool at one time
    if (_element_size_on_disk == 0)
    {
      _element_size_on_disk = GetElementSize(core, std::size_t(current));

      if (_verbose)
      {
        std::stringstream debug_element_size;
        debug_element_size << "element size is " << _element_size_on_disk;
        _debug_print(core, debug_element_size.str());
      }
    }

    if ( maximum_elements == -1 )
    {
      maximum_elements = long(_pool_size / _element_size_on_disk);

      if (_verbose)
      {
        std::stringstream debug_element_max;
        debug_element_max << "maximum element count: " << maximum_elements;
        _debug_print(core, debug_element_max.str());
      }
    }

    ++current;

    can_add_more_in_batch = (current - _pool_element_start) != maximum_elements;
    can_add_more_overall = std::size_t(current) != _pool_num_objects;
  }

  free(readback_buffer);

  if ( ! can_add_more_in_batch )
  {
    _debug_print(core, "reached capacity", true);
  }

  if ( ! can_add_more_overall )
  {
    _debug_print(core, "reached last element", true);
  }

  _pool_element_end = current;

  {
    /* cheat: signal the mcas server to start profiling */
    std::vector<uint64_t> v(1);
    auto rc = _store->get_attribute(_pool, component::IKVStore::COUNT, v);
    assert(rc == S_OK );
  }
  if (_verbose)
  {
    std::cout << "_pool_element_end = " << _pool_element_end << std::endl;
    std::stringstream range_info;
    range_info << "current = " << current << ", end = " << _pool_element_end;
    _debug_print(core, range_info.str(), true);

    range_info = std::stringstream();
    range_info << "elements added to pool: " << current - _pool_element_start << ". Last = " << current;
    _debug_print(core, range_info.str(), true);
  }
  PLOG("pool populated to capacity - OK.");
}

// assumptions: i_ is tracking current element in use
void Experiment::_enforce_maximum_pool_size(unsigned core, std::size_t i_)
{
  ++_elements_in_use;

  // erase elements that exceed pool capacity and start again
  if ((_elements_in_use * _element_size_on_disk) >= _pool_size)
  {
    bool timer_running_at_start = timer.is_running();  // if timer was running, pause it

    if (timer_running_at_start)
    {
      timer.stop();

      if (_verbose)
      {
        PLOG("%s", "enforce_maximum_pool_size pausing timer");
      }
    }

    if(_verbose)
    {
      std::stringstream debug_message;
      debug_message << "exceeded acceptable pool size of " << _pool_size << ". Erasing " << _elements_in_use << " elements of size " << _element_size_on_disk << " (" << _elements_in_use * _element_size_on_disk << " total)";

      _debug_print(core, debug_message.str(), true);
    }

    try
    {
      for (auto i = i_ - 1; i > (i_ - _elements_in_use); --i)
      {
        auto rc =_store->erase(_pool, g_data->key(i));
        if (rc != S_OK && core == 0)
        {
          // throw exception
          std::string error_string = "erase returned !S_OK: ";
          error_string.append(std::to_string(rc));
          error_string.append(", i = " + std::to_string(i) + ", i_ = " + std::to_string(i_));
          perror(error_string.c_str());
        }
      }
    }
    catch(...)
    {
      PERR("%s", "failed during erase step");
      throw;
    }

    _elements_in_use = 0;

    if (_verbose)
    {
      std::stringstream debug_end;
      debug_end << "done. i_ = " << i_;

      _debug_print(core, debug_end.str(), true);
    }

    if (timer_running_at_start)
    {
      if (_verbose)
      {
        PLOG("%s", "enforce_maximum_pool_size restarting timer");
      }

      timer.start();
    }
  }
}

void Experiment::_erase_pool_entries_in_range(pool_entry_offset_t start, pool_entry_offset_t finish)
{
  if (_verbose)
  {
    std::cout << "erasing pool entries in [" << start << ".." << finish << ")" << std::endl;
  }

  try
  {
    for (auto i = start; i < finish; ++i)
    {
      auto rc = _store->erase(_pool, g_data->key(std::size_t(i)));

      if (rc != S_OK)
      {
        auto e = "IKVStore::erase returned " + std::to_string(rc);
        PERR("%s.", e.c_str());
        throw std::runtime_error(e);
      }
    }
  }
  catch ( ... )
  {
    PERR("%s", "erase step failed");
    throw;
  }
}
