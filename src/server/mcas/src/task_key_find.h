#ifndef __mcas_SERVER_TASK_KEY_FIND_H__
#define __mcas_SERVER_TASK_KEY_FIND_H__

#include <common/logging.h>
#include <gsl/pointers>
#include <unistd.h>
#include <string>

#include "task.h"

namespace mcas
{
/**
 * Key search task.  We limit the number of hops we search so as to bound
 * the worst case execution time.
 *
 */
class Key_find_task : public Shard_task,
                      private common::log_source
{
  static constexpr unsigned MAX_COMPARES_PER_WORK = 5;

 public:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"  // uninitialized _expr, _out_key, _type
  Key_find_task(const std::string& expression,
                const offset_t offset,
                Connection_handler* handler,
                gsl::not_null<component::IKVIndex*> index,
                const unsigned debug_level)
      : Shard_task(handler),
        log_source(debug_level),
        _offset(offset),
        _index(index)
  {
    using namespace component;
    _index->add_ref();
    PNOTICE("--");
    CPLOG(1, "Key_find_task: offset=%lu", offset);
    CPLOG(1, "Key_find_task: expr(%s)", expression.c_str());

    if (expression == "next:") {
      _type = IKVIndex::FIND_TYPE_NEXT;
      _expr = expression.substr(5);
    }
    else if (expression.substr(0, 6) == "regex:") {
      _type = IKVIndex::FIND_TYPE_REGEX;
      _expr = expression.substr(6);
    }
    else if (expression.substr(0, 6) == "exact:") {
      _type = IKVIndex::FIND_TYPE_EXACT;
      _expr = expression.substr(6);
    }
    else if (expression.substr(0, 7) == "prefix:") {
      _type = IKVIndex::FIND_TYPE_PREFIX;
      _expr = expression.substr(7);
    }
    else
      throw Logic_exception("unhandled expression");

  }
#pragma GCC diagnostic pop

  Key_find_task(const Key_find_task&) = delete;
  Key_find_task& operator=(const Key_find_task&) = delete;

  status_t do_work() override
  {
    using namespace component;

    status_t hr;
    try {
      hr = _index->find(_expr, _offset, _type, _offset, _out_key, MAX_COMPARES_PER_WORK);

      if (hr == E_MAX_REACHED) {
        _offset++;
        return component::IKVStore::S_MORE;
      }
      else if (hr == S_OK) {
        CPLOG(1, "matched: (%s)", _out_key.c_str());
        return S_OK;
      }
      else {
        _out_key.clear();
        return hr;
      }
    }
    catch (...) {
      return E_FAIL;
    }

    throw Logic_exception("unexpected code path (hr=%d)", hr);
  }

  const void* get_result() const override { return _out_key.data(); }

  size_t get_result_length() const override { return _out_key.length(); }

  offset_t matched_position() const override { return _offset; }

 private:
  std::string                             _expr;
  std::string                             _out_key;
  component::IKVIndex::find_t             _type;
  offset_t                                _offset;
  component::Itf_ref<component::IKVIndex> _index;
};

}  // namespace mcas
#endif  // __mcas_SERVER_TASK_KEY_FIND_H__
