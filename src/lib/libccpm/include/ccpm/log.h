/*
  Copyright [2020] [IBM Corporation]
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#ifndef MCAS_CCPM_LOG_H__
#define MCAS_CCPM_LOG_H__

#include <ccpm/interfaces.h>
#include <gsl/pointers>
#include <cstring>
#include <cstddef>

namespace ccpm
{
struct block_header;

struct log
  : public ILog
{
  using heap_type = gsl::not_null<IHeap_expandable *>;
  using persister_type = gsl::not_null<ccpm::persister *>;
private:
  /*
   * The log needs to be stored persistently, and to use persistent storage.
   * Use persister_type for the former and heap_type for the latter.
   */
  persister_type _persister; // not owned
  heap_type _mr; // not owned
  /* The log needs a root */
  block_header *_root; // owned
  void clear_top();

  static constexpr size_t min_log_extend = std::size_t(65536U);

  void extend(std::size_t size);
public:
  explicit log(persister_type, heap_type mr_);

  log(const log &) = delete;
  log &operator=(const log &) = delete;

  ~log();
  /*
   * Make a record of an old value
   */
  void add(void *begin, std::size_t size) override;
  /*
   * Make a record of an allocate
   */
  void allocated(void *&p, std::size_t size) override;
  /*
   * Make a record of a free
   */
  void freed(void *&p, std::size_t size) override;
  /*
   * commit and discard all log entries
   */
  void commit() override;
  /*
   * Restore all data areas added after initialization (or the most recent clear)
   * to the values present at their last add command.
   */
  void rollback() override;

  bool includes(const void *addr) const { return _mr->includes(addr); }
};

struct bad_alloc_log
	: public std::bad_alloc
{
	bad_alloc_log()
	{}
	const char *what() const noexcept override
	{
		return "bad undo log allocate";
	}
};

}
#endif
