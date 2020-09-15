/*
  Copyright [2017-2020] [IBM Corporation]
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
#ifndef __BUFFER_MGR_H__
#define __BUFFER_MGR_H__

#ifdef __cplusplus

#include <common/chksum.h>
#include <common/delete_copy.h>
#include <common/logging.h>
#include <common/moveable_ptr.h>
#include <api/components.h>
#include <api/fabric_itf.h>
#include <api/kvstore_itf.h>
#include <sys/mman.h>

#include "mcas_config.h"
#include "memory_registered.h"
#include <gsl/pointers> /* not_null */
#include <cassert>
#include <cstring>      /* memset */
#include <memory>       /* unique_ptr */

namespace
{
  auto alloc_base(std::size_t len) -> gsl::not_null<void *>
  {
    auto b = ::aligned_alloc(MiB(2), len);
    if (b == nullptr) {
      throw std::bad_alloc();
    }
    ::memset(b, 0, len);
    return gsl::not_null<void *>(b);
  }
}

namespace mcas
{
template <class Transport>
class Buffer_manager : private common::log_source {

  static constexpr const char *_cname       = "Buffer_manager";

public:
  static constexpr size_t DEFAULT_BUFFER_COUNT = NUM_SHARD_BUFFERS;
  static constexpr size_t BUFFER_LEN           = MiB(2); /* corresponds to huge page see below */
  using memory_registered_t                    = memory_registered<Transport>;

  using memory_region_t = typename Transport::memory_region_t;

  struct iov_mem_lock
  {
  private:
    common::moveable_ptr<::iovec> _iov;
  public:
    iov_mem_lock(::iovec *iov_)
      : _iov(iov_)
    {
      ::madvise(_iov->iov_base, _iov->iov_len, MADV_HUGEPAGE);
      ::mlock(_iov->iov_base, _iov->iov_len);
    }
    iov_mem_lock(iov_mem_lock &&) noexcept = default;
    ~iov_mem_lock()
    {
      if ( _iov )
      {
        ::munlock(_iov->iov_base, _iov->iov_len);
      }
    }
  };

  /* Although client tried to hide it with a reinterpret_cast, buffer_base is
   used as a component::memory_region_t */

  struct buffer_base : public component::IKVStore::Opaque_memory_region, private common::log_source
  {
  private:
    static constexpr const char *_cname = "buffer_base";
    void *_base;
    std::size_t _original_length;
    memory_registered_t _region;
    const unsigned magic;

  public:
    buffer_base(unsigned    debug_level_,
             Transport * transport_,
             void *      base_,
             size_t      length_
    )
      : common::log_source(debug_level_)
      , _base(base_)
      , _original_length(length_)
      , _region(memory_registered_t(debug_level_, transport_, base_, length_, 0, 0))
      , magic(0xC0FFEE)
    {
      assert(length_);
      CPLOG(2, "%s::%s %p transport %p region %p", _cname, __func__, static_cast<void *>(this),
            static_cast<void *>(transport()), static_cast<void *>(region()));
    }

    void *get_desc() const { return _region.desc(); }

    DELETE_COPY(buffer_base);

  protected:
    ~buffer_base()
    {
      CPLOG(2, "%s::%s %p transport %p region %p", _cname, __func__, static_cast<void *>(this),
            static_cast<void *>(transport()), static_cast<void *>(region()));
    }

  public:
    size_t original_length() const { return _original_length; }
    inline gsl::not_null<void *> base() const { return _base; }
    auto region() const { return _region.mr(); }
    Transport * transport() const { return _region.transport(); }

#if 0
    /* unused */
    inline bool check_magic() const { return magic == 0xC0FFEE; }
    unsigned int crc32() const { return common::chksum32(_base, _length); }
#endif
  };

  struct free_deleter
  {
    void operator()(void *v_) { ::free(v_); }
  };

  struct buffer_internal
    : private std::unique_ptr<void, free_deleter>
    , public buffer_base
  {
    ::iovec iov[2];
    void *desc[2];
    iov_mem_lock _ml;
    using completion_t = void (*)(void *, buffer_internal *);
    completion_t   completion_cb;
    void *         value_adjunct;
    inline void set_completion(completion_t completion_) { completion_cb = completion_; }
    void set_completion(completion_t completion_, void *value_adjunct_)
    {
      completion_cb = completion_;
      value_adjunct = value_adjunct_;
    }
    buffer_internal(unsigned debug_level_,
             Transport *         transport_,
             size_t              length_
    )
      : std::unique_ptr<void, free_deleter>(alloc_base(length_))
      , buffer_base(debug_level_, transport_, std::unique_ptr<void, free_deleter>::get(), length_)
      , iov{{this->base(), this->original_length()}, {nullptr, 0}}
      , desc{this->get_desc(), nullptr}
      , _ml(&this->iov[0])
      , completion_cb(nullptr)
      , value_adjunct(nullptr)
    {
    }

    DELETE_COPY(buffer_internal);

    void set_length(size_t s) { iov->iov_len = s; }

    inline void reset_length()
    {
      assert(this->original_length() > 1);
      this->iov[0].iov_len  = this->original_length();
      value_adjunct = nullptr;
      completion_cb = nullptr;
    }

    size_t length() const { return iov[0].iov_len; }

    unsigned int crc32() const { return common::chksum32(iov->iov_base, iov->iov_len); }
  };

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"  // missing initializers
  Buffer_manager(unsigned debug_level_,
                 Transport *transport,
                 size_t buffer_count = DEFAULT_BUFFER_COUNT)
    : common::log_source(debug_level_),
      _buffer_count(buffer_count),
      _transport(transport)
  {
    for (unsigned i = 0; i < _buffer_count; i++) {
      const auto len  = BUFFER_LEN;
      _buffers.emplace_back(std::make_unique<buffer_internal>(debug_level_, _transport, len));
      _free.push_back(_buffers.back().get());
    }
    PLOG("%s %p allocated %lu buffers", __func__, static_cast<void *>(this), buffer_count);
  }
#pragma GCC diagnostic pop

  Buffer_manager(Buffer_manager &&) noexcept = default;

  ~Buffer_manager()
  {
  }

  using completion_t = void (*)(void *, buffer_internal *);

  buffer_internal *allocate(completion_t completion_)
  {
    if (UNLIKELY(_free.empty())) throw Program_exception("Buffer_manager: no shard buffers remaining");
    auto iob = _free.back();
    _free.pop_back();
    CPLOG(3, "%s::%s %p (%lu free)", _cname, __func__, static_cast<const void *>(iob), _free.size());
    assert(iob);
    iob->reset_length();
    iob->set_completion(completion_);
    return iob;
  }

  void free(buffer_internal *iob)
  {
    assert(iob);

    CPLOG(3, "%s::%s %p (%lu free)", _cname, __func__, static_cast<const void *>(iob), _free.size());
    iob->reset_length();
    iob->set_completion(nullptr);
    _free.push_back(iob);
  }

private:
  // static constexpr size_t buffer_len() { return BUFFER_LEN; }

  using pool_t = component::IKVStore::pool_t;
  using key_t  = std::uint64_t;

  const size_t                           _buffer_count;
  gsl::not_null<Transport *>             _transport;
  std::vector<std::unique_ptr<buffer_internal>> _buffers;
  std::vector<buffer_internal *>        _free;
};
}  // namespace mcas

#endif
#endif
