#ifndef __EXPERIMENT_H__
#define __EXPERIMENT_H__

#include "task.h"

#include "direct_memory_registered.h"
#include "dotted_pair.h"
#include "statistics.h"
#include "stopwatch.h"

#include "rapidjson/document.h"

#include <boost/optional.hpp>
#include <common/logging.h> /* log_source */

#include <api/kvstore_itf.h>

#include <chrono>
#include <map> /* map - should follow local includes */
#include <cstddef> /* size_t */
#include <memory>
#include <mutex>
#include <string>
#include <vector>

class Data;
class ProgramOptions;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
class Experiment : public common::Tasklet, protected common::log_source
{
  using direct_memory_registered_kv = direct_memory_registered<component::IKVStore>;
private:
  std::string _pool_path;
  std::string _pool_name;
  std::string _pool_name_local;  // full pool name, e.g. "Exp.pool.0" for core 0
  std::string _owner;
  unsigned long long int _pool_size;
  std::uint32_t _pool_flags;
  std::size_t _pool_num_objects;
  std::string _cores;
  std::string _devices;
  std::vector<unsigned> _core_list;
  boost::optional<std::chrono::system_clock::time_point> _start_time; // default behavior: start now
public:
  using pool_entry_offset_t = std::vector<bool>::difference_type;
  boost::optional<std::chrono::high_resolution_clock::duration> _duration_directed; // default behavior: number of elements determines duration
  boost::optional<std::chrono::high_resolution_clock::time_point> _end_time_directed;
private:
  std::string _component;
  std::string _results_path;
  std::string _report_filename;
  bool _do_json_reporting;
  std::string _test_name;

  // common experiment parameters
  std::unique_ptr<component::IKVStore>  _store;
  component::IKVStore::pool_t           _pool;
  std::string                           _server_address;
  boost::optional<std::string>          _provider;
  unsigned                              _port;
  unsigned                              _port_increment;
  boost::optional<std::string>          _device_name;
  boost::optional<std::string>          _src_addr;
  boost::optional<std::string>          _pci_address;
public:
  bool                                  _first_iter;
private:
  bool                                  _ready;
public:
  Stopwatch timer;
protected:
  direct_memory_registered_kv _memory_handle;
private:
  bool _verbose;
  bool _summary;

  // member variables for tracking pool sizes
  std::size_t _element_size_on_disk; // floor: filesystem block size
  std::size_t _element_size; // raw amount of file data (bytes)
  std::size_t _elements_in_use;
  pool_entry_offset_t _pool_element_start;
public:
  pool_entry_offset_t _pool_element_end;
private:
  long _elements_stored;
  unsigned long _total_data_processed;  // for use with throughput calculation

  // bin statistics
  unsigned _bin_count;
  double _bin_threshold_min;
  double _bin_threshold_max;

  using core_to_device_map_t = std::map<unsigned, dotted_pair<unsigned>>;
  core_to_device_map_t _core_to_device_map;

  static core_to_device_map_t make_core_to_device_map(const std::string &cores, const std::string &devices);

public:
  static Data * g_data;
  static std::mutex g_write_lock;
  static double g_iops;

  Experiment(std::string name_, const ProgramOptions &options);

  Experiment(const Experiment &) = delete;
  Experiment& operator=(const Experiment &) = delete;

  virtual ~Experiment();

  auto test_name() const -> std::string { return _test_name; }
  unsigned bin_count() const { return _bin_count; }
  double bin_threshold_min() const { return _bin_threshold_min; }
  double bin_threshold_max() const { return _bin_threshold_max; }
  component::IKVStore *store() const { return _store.get(); }
  component::IKVStore::pool_t pool() const { return _pool; }
  std::size_t pool_num_objects() const { return _pool_num_objects; }
  bool is_verbose() const { return _verbose; }
  bool is_summary() const { return _summary; }
  unsigned long total_data_processed() const { return _total_data_processed; }
  bool is_json_reporting() const { return _do_json_reporting; }
  auto pool_element_start() const { return _pool_element_start; }
  auto pool_element_end() const { return _pool_element_end; }
  bool component_is(const std::string &c) const { return _component == c; }
  unsigned long long pool_size() const { return _pool_size; }
  component::IKVStore::memory_handle_t memory_handle() const { return _memory_handle.mr(); }

  void initialize(unsigned core) override;

  // if experiment should be delayed, stop here and wait. Otherwise, start immediately
  void wait_for_delayed_start(unsigned core);

  /* maximun size of any dax device. Limitation: considers only dax devices specified in the device string */
  std::size_t dev_dax_max_size(const std::string & dev_dax_prefix_);

  /* maximun number of dax devices in any numa node.
   * Limitations:
   *   Considers only dax devices specified in the device string.
   */
  unsigned dev_dax_max_count_per_node();

  dotted_pair<unsigned> core_to_device(unsigned core);

  int initialize_store(unsigned core);

  virtual void initialize_custom(unsigned core);

  bool ready() override
  {
    return _ready;
  }

  virtual void cleanup_custom(unsigned core);

  void _update_aggregate_iops(double iops);

  void summarize();

  void cleanup(unsigned core) noexcept override;

  bool component_uses_direct_memory()
  {
    return component_is( "mcas" );
  }

  bool component_uses_rdma()
  {
    return component_is( "mcas" );
  }

  void _debug_print(unsigned core, std::string text, bool limit_to_core0=false);

  unsigned _get_core_index(unsigned core) ;

  rapidjson::Document _get_report_document() ;

  void _initialize_experiment_report(rapidjson::Document& document) ;

  void _report_document_save(rapidjson::Document& document, unsigned core, rapidjson::Value& new_info) ;

  void _print_highest_count_bin(BinStatistics& stats, unsigned core);

  rapidjson::Value _add_statistics_to_report(BinStatistics& stats, rapidjson::Document& document) ;

  BinStatistics _compute_bin_statistics_from_vectors(std::vector<double> data, std::vector<double> data_bins, unsigned bin_count, double bin_min, double bin_max, std::size_t elements) ;

  BinStatistics _compute_bin_statistics_from_vector(std::vector<double> data, unsigned bin_count, double bin_min, double bin_max) ;

  /* create_report: output a report in JSON format with experiment data
   * Report format:
   *      experiment object - contains experiment parameters
   *      data object - actual results
   */
  static std::string create_report(const std::string component_);

  std::size_t GetDataInputSize(std::size_t index);

  unsigned long GetElementSize(unsigned core, std::size_t index) ;

  void _update_data_process_amount(unsigned core, std::size_t index) ;

  // throughput = Mib/s here
  double _calculate_current_throughput() ;

  void _populate_pool_to_capacity(unsigned core, component::IKVStore::memory_handle_t memory_handle = component::IKVStore::HANDLE_NONE) ;

  // assumptions: i_ is tracking current element in use
  void _enforce_maximum_pool_size(unsigned core, std::size_t i_) ;

  void _erase_pool_entries_in_range(pool_entry_offset_t start, pool_entry_offset_t finish);

};
#pragma GCC diagnostic pop


#endif //  __EXPERIMENT_H__
