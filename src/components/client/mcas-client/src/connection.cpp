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

#include "connection.h"

#include "protocol.h"
#include "range.h"

#include <city.h>
#include <common/cycles.h>
#include <common/delete_copy.h>
#include <common/utils.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <memory>

//#define DEBUG_NPC_RESPONSES

using namespace component;

using memory_registered_fabric = mcas::memory_registered<IFabric_client>;

struct remote_fail : public std::runtime_error {
private:
  int _status;

 public:
  remote_fail(int status_) : runtime_error("remote fail"), _status(status_) {}
  int status() const { return _status; }
};

namespace mcas
{
namespace client
{
/* The various embedded returns and throws suggest that the allocated
 * iobs should be automatically freed to avoid leaks.
 */

struct iob_free {
private:
  Connection_handler *_h;

 public:
  iob_free(Connection_handler *h_) : _h(h_) {}
  void operator()(Connection_handler::buffer_t *iob) { _h->free_buffer(iob); }
};

struct memory_registered_not_owned {
private:
  common::moveable_ptr<void> _desc;

 public:
  memory_registered_not_owned(Registrar_memory_direct *  // mcas
                              ,
                              const mcas::range<char *> &  // range registered
                              ,
                              void *desc_)
      : _desc(desc_)
  {
  }
  virtual ~memory_registered_not_owned() {}
  void *desc() const { return _desc; }
};

struct memory_registered_owned {
private:
  common::moveable_ptr<Registrar_memory_direct> _rmd;
  component::IMCAS::memory_handle_t _h;

 public:
  memory_registered_owned(Registrar_memory_direct *  rmd_,
                          const mcas::range<char *> &range_  // range to register
                          ,
                          void *  // desc
                          )
      : _rmd(rmd_),
        _h(_rmd->register_direct_memory(range_.first, range_.length()))
  {
  }
  DELETE_COPY(memory_registered_owned);
  memory_registered_owned(memory_registered_owned &&) noexcept = default;
  virtual ~memory_registered_owned()
  {
    if (_rmd) {
      _rmd->unregister_direct_memory(_h);
    }
  }
  void *desc() const { return static_cast<client::Fabric_transport::buffer_base *>(_h)->get_desc(); }
};

/**
 * @brief Used to track buffers for asynchronous invocations
 *
 * is an async handle.
 */
struct async_buffer_set_t : public component::IMCAS::Opaque_async_handle, protected common::log_source {
  using iob_ptr = std::unique_ptr<client::Fabric_transport::buffer_t, iob_free>;

 protected:
  iob_ptr        iobs;
  iob_ptr        iobr;

  async_buffer_set_t(unsigned debug_level_, iob_ptr &&iobs_, iob_ptr &&iobr_) noexcept
      : component::IMCAS::Opaque_async_handle{},
        common::log_source(debug_level_),
        iobs(std::move(iobs_)),
        iobr(std::move(iobr_))
  {
    CPLOG(2, "%s iobs %p iobr %p"
      , __func__
      , static_cast<const void *>(&*iobs)
      , static_cast<const void *>(&*iobr)
    );
  }

  async_buffer_set_t()                           = delete;
  DELETE_COPY(async_buffer_set_t);

 public:
  virtual ~async_buffer_set_t() {}
  virtual int move_along(Connection_handler *c) = 0;
};

/* Nothing more than the two buffers. Used for async erase */
struct async_buffer_set_simple : public async_buffer_set_t {
  async_buffer_set_simple(unsigned debug_level_, iob_ptr &&iobs_, iob_ptr &&iobr_) noexcept
      : async_buffer_set_t(debug_level_, std::move(iobs_), std::move(iobr_))
  {
  }
  int move_along(Connection_handler *c) override
  {
    if (iobs) { /* check submission, clear and free on completion */
      if (c->test_completion(&*iobs) == false) {
        return E_BUSY;
      }
      iobs.reset(nullptr);
    }

    if (iobr) { /* check recv, clear and free on completion */
      if (c->test_completion(&*iobr) == false) {
        return E_BUSY;
      }

      const auto response_msg = c->msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, "ASYNC PUT");

      auto status = response_msg->get_status();
      iobr.reset(nullptr);
      return status;
    }
    else {
      throw API_exception("invalid async handle, task already completed?");
    }
  }
};

/* Is also an M, which is memory_registered_owned if the caller
 * has *not* registered memory, and is memory_registered_not_owned
 * if the caller *has* registered memory.
 */
template <typename M>
struct async_buffer_set_get_locate
    : public async_buffer_set_t
    , public M {
private:
  static constexpr const char *_cname = "async_buffer_set_get_locate";
  iob_ptr                      _iobrd;
  component::IMCAS::pool_t     _pool;
  std::uint64_t                _auth_id;
  void *                       _value;
  std::size_t &                _value_len;
  void *                       _desc[1];
  ::iovec                      _v[1];
  std::uint64_t                _addr;

 public:
  async_buffer_set_get_locate(unsigned debug_level_,
                              Registrar_memory_direct *rmd_,
                              iob_ptr &&               iobrd_,
                              iob_ptr &&               iobs_,
                              iob_ptr &&               iobr_,
                              component::IMCAS::pool_t pool_,
                              std::uint64_t            auth_id_,
                              void *                   value_,
                              std::size_t &            value_len_, // In: buffer size of value_. Out: actual KV store value size
                              Connection_handler       *c,
                              void *                   desc_,
                              std::uint64_t            addr_,
                              std::uint64_t            key_
    )
      : async_buffer_set_t(debug_level_, std::move(iobs_), std::move(iobr_))
      , M(rmd_,
          mcas::range<char *>(static_cast<char *>(value_), static_cast<char *>(value_) + value_len_)
              .round_inclusive(4096),
          desc_),
        _iobrd(std::move(iobrd_)),
        _pool(pool_),
        _auth_id(auth_id_),
        _value(value_),
        _value_len(value_len_),
        _desc{this->desc()},  // provided by M
        _v{::iovec{_value, _value_len}},
        _addr(addr_)
  {
    CPLOG(2, "%s: iobrd %p iobs2 %p iobr2 %p"
      , __func__
      , static_cast<const void *>(&*_iobrd)
      , static_cast<const void *>(&*iobs)
      , static_cast<const void *>(&*iobr)
    );
    /* reply have been received, with credentials for the DMA */

    CPLOG(2,
      "%s::%s post_read %p local (addr %p.%zx desc %p) <- (_addr 0x%zx, key 0x%zx)"
      , _cname
      , __func__
      , static_cast<const void *>(&*_iobrd)
      , _v[0].iov_base, _v[0].iov_len
      , _desc[0]
      , _addr, key_
    );
    c->post_read(std::begin(_v), std::end(_v), std::begin(_desc), _addr, key_, &*_iobrd);
    /* End */
  }
  DELETE_COPY(async_buffer_set_get_locate);
  int                  move_along(Connection_handler *c) override
  {
    if (_iobrd) {
      if (!c->test_completion(&*_iobrd)) {
        return E_BUSY;
      }
      /* What to do when DMA completes */
      CPLOG(2, "%s dma complete %p", __func__, static_cast<const void *>(&*_iobrd));
      _iobrd.reset(nullptr);
      /* DMA is complete. Issue GET_RELEASE */

      /* send release message */
      const auto msg = new (iobs->base())
          protocol::Message_IO_request(_auth_id, c->request_id(), _pool, protocol::OP_TYPE::OP_GET_RELEASE, _addr);

      c->post_recv(&*iobr);
      c->sync_inject_send(&*iobs, msg, __func__);
      /* End */
    }

    if ( iobr )
    {
      if ( ! c->test_completion(&*iobr) )
      {
        return E_BUSY;
      }
      /* What to do when second recv completes */
      const auto response_msg = c->msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, "ASYNC GET_RELEASE");
      auto status = response_msg->get_status();

      iobr.reset(nullptr);
      return status;
      /* End */
    }
    else {
      throw API_exception("invalid async handle, task already completed?");
    }
  }
};

template <typename M>
struct async_buffer_set_put_locate
    : public async_buffer_set_t
    , public M {
private:
  static constexpr const char *_cname = "async_buffer_set_put_locate";
  iob_ptr                      _iobrd;
  iob_ptr                      _iobs2;
  iob_ptr                      _iobr2;
  component::IMCAS::pool_t     _pool;
  std::uint64_t                _auth_id;
  const void *                 _value;
  std::size_t                  _value_len;
  void *                       _desc[1];
  ::iovec                      _v[1];
  std::uint64_t                _addr;

 public:
  async_buffer_set_put_locate(unsigned                 debug_level_,
                              Registrar_memory_direct *rmd_,
                              iob_ptr &&               iobs_,
                              iob_ptr &&               iobr_,
                              iob_ptr &&               iobrd_,
                              iob_ptr &&               iobs2_,
                              iob_ptr &&               iobr2_,
                              component::IMCAS::pool_t pool_,
                              std::uint64_t            auth_id_,
                              const void *             value_,
                              std::size_t              value_len_,
                              void *                   desc_)
      : async_buffer_set_t(debug_level_, std::move(iobs_), std::move(iobr_)),
        M(rmd_,
          mcas::range<char *>(static_cast<char *>(const_cast<void *>(value_)),
                              static_cast<char *>(const_cast<void *>(value_)) + value_len_)
              .round_inclusive(4096),
          desc_),
        _iobrd(std::move(iobrd_)),
        _iobs2(std::move(iobs2_)),
        _iobr2(std::move(iobr2_)),
        _pool{pool_},
        _auth_id{auth_id_},
        _value{value_},
        _value_len{value_len_},
        _desc{this->desc()}  // provided by M
        ,
        _v{::iovec{const_cast<void *>(_value), _value_len}},
        _addr{}
  {
      CPLOG(2, "%s: iobrd %p iobs2 %p iobr2 %p"
        , __func__
        , static_cast<const void *>(&*_iobrd)
        , static_cast<const void *>(&*_iobs2)
        , static_cast<const void *>(&*_iobr2)
      );
  }
  DELETE_COPY(async_buffer_set_put_locate);
  int                  move_along(Connection_handler *c) override
  {
    if (iobs) { /* check submission, clear and free on completion */
      if (c->test_completion(&*iobs) == false) {
        return E_BUSY;
      }
      /* What to do when first send completes */
      iobs.reset(nullptr);
      /* End */
    }

    if (iobr) { /* check recv, clear and free on completion */
      if (c->test_completion(&*iobr) == false) {
        return E_BUSY;
      }
      /* What to do when first recv completes */
      const auto response_msg = c->msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, "ASYNC PUT_LOCATE");
      auto status = response_msg->get_status();

      _addr    = response_msg->addr;
      auto key = response_msg->key;
      iobr.reset(nullptr);
      if (status != S_OK) {
        return status;
      }
      /* reply have been received, with credentials for the DMA */

      CPLOG(2,
        "%s post_write %p local (addr %p.%zx desc %p) -> (_addr 0x%zx, key 0x%zx)"
        , __func__
        , static_cast<const void *>(&*_iobrd)
        , _v[0].iov_base, _v[0].iov_len
        , _desc[0]
        , _addr, key
      );
      c->post_write(std::begin(_v), std::end(_v), std::begin(_desc), _addr, key, &*_iobrd);
      /* End */
    }

    if (_iobrd) {
      if (!c->test_completion(&*_iobrd)) {
        return E_BUSY;
      }
      /* What to do when DMA completes */
      CPLOG(2, "%s dma complete %p", __func__, static_cast<const void *>(&*_iobrd));
      _iobrd.reset(nullptr);
      /* DMA is complete. Issue PUT_RELEASE */

      /* send release message */
      const auto msg = new (_iobs2->base())
          protocol::Message_IO_request(_auth_id, c->request_id(), _pool, protocol::OP_TYPE::OP_PUT_RELEASE, _addr);

      c->post_recv(&*_iobr2);
      c->sync_inject_send(&*_iobs2, msg, __func__);
      /* End */
    }

    if (_iobr2) {
      if (!c->test_completion(&*_iobr2)) {
        return E_BUSY;
      }
      /* What to do when second recv completes */
      const auto response_msg = c->msg_recv<const mcas::protocol::Message_IO_response>(&*_iobr2, "ASYNC PUT_RELEASE");
      auto status = response_msg->get_status();

      _iobr2.reset(nullptr);
      return status;
      /* End */
    }
    else {
      throw API_exception("invalid async handle, task already completed?");
    }
  }
};

struct async_buffer_set_invoke : public async_buffer_set_t {
  std::vector<IMCAS::ADO_response> *out_ado_response;

 public:
  async_buffer_set_invoke(unsigned                          debug_level_,
                          iob_ptr &&                        iobs_,
                          iob_ptr &&                        iobr_,
                          std::vector<IMCAS::ADO_response> *out_ado_response_)
      : async_buffer_set_t(debug_level_, std::move(iobs_), std::move(iobr_)),
        out_ado_response(out_ado_response_)
  {
  }
  DELETE_COPY(async_buffer_set_invoke);
  int              move_along(Connection_handler *c) override
  {
    if (iobs) { /* check submission, clear and free on completion */
      if (c->test_completion(&*iobs) == false) {
        return E_BUSY;
      }
      iobs.reset(nullptr);
    }

    if (iobr) { /* check recv, clear and free on completion */
      if (c->test_completion(&*iobr) == false) {
        return E_BUSY;
      }
      const auto response_msg = c->msg_recv<const mcas::protocol::Message_ado_response>(&*iobr, __func__);
      assert(response_msg);
      auto status = response_msg->get_status();

      if (status == S_OK) {
        auto &out_response = *out_ado_response;
        out_response.clear();

        /* unmarshal responses */
        for (uint32_t i = 0; i < response_msg->get_response_count(); i++) {
          void *   out_data     = nullptr;
          size_t   out_data_len = 0;
          uint32_t out_layer_id = 0;
          response_msg->client_get_response(i, out_data, out_data_len, out_layer_id);

#ifdef DEBUG_NPC_RESPONSES
          PLOG("Response:");
          hexdump(out_data, out_data_len);
#endif
          out_response.emplace_back(out_data, out_data_len, out_layer_id);
        }
      }
      return status;
    }
    else {
      throw API_exception("invalid async handle, task already completed?");
    }
  }
};

template <typename M>
struct async_buffer_set_get_direct_offset
    : public async_buffer_set_t
    , public M {
private:
  using locate_element                               = protocol::Message_IO_response::locate_element;
  static constexpr const char *               _cname = "async_buffer_set_get_direct_offset";
  iob_ptr                                     _iobrd;
  iob_ptr                                     _iobs2;
  iob_ptr                                     _iobr2;
  component::IMCAS::pool_t                    _pool;
  std::uint64_t                               _auth_id;
  std::size_t                                 _offset;
  char *                                      _buffer;
  std::size_t &                               _length;
  std::uint64_t                               _key;
  void *                                      _desc[1];
  ::iovec                                     _v[1];
  std::vector<locate_element>                 _addr_list;
  std::vector<locate_element>::const_iterator _addr_cursor;

 public:
  async_buffer_set_get_direct_offset(unsigned                 debug_level_,
                                     Registrar_memory_direct *rmd_,
                                     iob_ptr &&               iobs_,
                                     iob_ptr &&               iobr_,
                                     iob_ptr &&               iobrd_,
                                     iob_ptr &&               iobs2_,
                                     iob_ptr &&               iobr2_,
                                     component::IMCAS::pool_t pool_,
                                     std::uint64_t            auth_id_,
                                     std::size_t              offset_,
                                     void *                   buffer_,
                                     std::size_t &            length_,
                                     void *                   desc_)
      : async_buffer_set_t(debug_level_, std::move(iobs_), std::move(iobr_)),
        M(rmd_,
          mcas::range<char *>(static_cast<char *>(buffer_), static_cast<char *>(buffer_) + length_)
              .round_inclusive(4096),
          desc_),
        _iobrd(std::move(iobrd_)),
        _iobs2(std::move(iobs2_)),
        _iobr2(std::move(iobr2_)),
        _pool(pool_),
        _auth_id{auth_id_},
        _offset{offset_},
        _buffer(static_cast<char *>(buffer_)),
        _length(length_),
        _key{},
        _desc{this->desc()}  // provided by M
        ,
        _v{},
        _addr_list{},
        _addr_cursor{}
  {
      CPLOG(2, "%s iobs2 %p iobr2 %p"
        , __func__
        , static_cast<const void *>(&*_iobs2)
        , static_cast<const void *>(&*_iobr2)
      );
  }
  DELETE_COPY(async_buffer_set_get_direct_offset);
  int                         move_along(Connection_handler *c) override
  {
    if (iobs) { /* check submission, clear and free on completion */
      if (c->test_completion(&*iobs) == false) {
        return E_BUSY;
      }
      /* What to do when first send completes */
      iobs.reset(nullptr);
      /* End */
    }

    if (iobr) { /* check recv, clear and free on completion */
      if (c->test_completion(&*iobr) == false) {
        return E_BUSY;
      }
      /* What to do when first recv completes */
      const auto response = c->msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, "ASYNC OP_LOCATE");
      auto status = response->get_status();
      {
        auto cursor = response->edata();
        _addr_list  = std::vector<locate_element>(cursor, cursor + response->element_count());

        CPLOG(2,
          "%s::%s: edata count %zu %p to %p"
          , _cname
          , __func__
          , response->element_count()
          , static_cast<const void *>(cursor)
          , static_cast<const void *>(cursor + response->element_count())
        );

        _length = 0;
        for (const auto &e : _addr_list) {
          _length += e.len;
          CPLOG(2, "%s::%s: addr 0x%" PRIx64 " len 0x%" PRIx64
            , _cname
            , __func__
            , e.addr, e.len);
        }
      }
      _key         = response->key;
      _addr_cursor = _addr_list.begin();

      iobr.reset(nullptr);

      return status == S_OK ? E_BUSY : status;
    }

    if ( _addr_list.empty() )
    {
      return S_OK;
    }

    if (_addr_cursor != _addr_list.end()) {
      if (_iobrd && !c->test_completion(&*_iobrd)) {
        return E_BUSY;
      }

      _iobrd = c->make_iob_ptr_read();
      CPLOG(2, "%s iobrd %p"
        , __func__
        , static_cast<const void *>(&*_iobrd)
      );

      /* reply have been received, with credentials for the DMA */
      _v[0] = ::iovec{_buffer, _addr_cursor->len};

      CPLOG(2,
        "%s::%s post_read %p local (addr %p.%zx desc %p) <- (_addr 0x%zx, key 0x%zx)"
        , _cname
        , __func__
        , static_cast<const void *>(&*_iobrd)
        , _v[0].iov_base, _v[0].iov_len
        , _desc[0]
        , _addr_cursor->addr, _key
      );
      c->post_read(std::begin(_v), std::end(_v), std::begin(_desc), _addr_cursor->addr, _key, &*_iobrd);
      _buffer += _addr_cursor->len;
      ++_addr_cursor;
      /* End */
    }

    if (_iobrd) {
      if (!c->test_completion(&*_iobrd)) {
        return E_BUSY;
      }
      /* What to do when DMA completes */
      /* DMA done. Might need another DMA */
      CPLOG(2, "%s::%s dma read complete %p"
        , _cname
        , __func__
        , static_cast<const void *>(&*_iobrd)
      );

      _iobrd.reset(nullptr);
      /* DMA is complete. Issue OP_RELEASE */

      /* send release message */
      const auto msg = new (_iobs2->base()) protocol::Message_IO_request(
          _auth_id, c->request_id(), _pool, protocol::OP_TYPE::OP_RELEASE, _offset, _length);

      c->post_recv(&*_iobr2);
      c->sync_inject_send(&*_iobs2, msg, __func__);
      /* End */
    }

    /* release in process, or not needed because length is 0 */
    if ( _iobr2 ) {
      if ( _iobr2 && ! c->test_completion(&*_iobr2) ) {
        return E_BUSY;
      }
      /* What to do when second recv completes */
      const auto response_msg = c->msg_recv<const mcas::protocol::Message_IO_response>(&*_iobr2, "OP_RELEASE");
      return response_msg->get_status();
      /* End */
    }
    else {
      throw API_exception("invalid async handle, task already completed?");
    }
  }
};

template <typename M>
struct async_buffer_set_put_direct_offset
    : public async_buffer_set_t
    , public M {
private:
  using locate_element                               = protocol::Message_IO_response::locate_element;
  static constexpr const char *               _cname = "async_buffer_set_put_direct_offset";
  iob_ptr                                     _iobrd;
  iob_ptr                                     _iobs2;
  iob_ptr                                     _iobr2;
  component::IMCAS::pool_t                    _pool;
  std::uint64_t                               _auth_id;
  std::size_t                                 _offset;
  const char *                                _buffer;
  std::size_t &                               _length;
  std::uint64_t                               _key;
  void *                                      _desc[1];
  ::iovec                                     _v[1];
  std::vector<locate_element>                 _addr_list;
  std::vector<locate_element>::const_iterator _addr_cursor;

 public:
  async_buffer_set_put_direct_offset(unsigned                 debug_level_,
                                     Registrar_memory_direct *rmd_,
                                     iob_ptr &&               iobs_,
                                     iob_ptr &&               iobr_,
                                     iob_ptr &&               iobrd_,
                                     iob_ptr &&               iobs2_,
                                     iob_ptr &&               iobr2_,
                                     component::IMCAS::pool_t pool_,
                                     std::uint64_t            auth_id_,
                                     std::size_t              offset_,
                                     const void *             buffer_,
                                     std::size_t &            length_,
                                     void *                   desc_)
      : async_buffer_set_t(debug_level_, std::move(iobs_), std::move(iobr_)),
        M(rmd_,
          mcas::range<char *>(static_cast<char *>(const_cast<void *>(buffer_)),
                              static_cast<char *>(const_cast<void *>(buffer_)) + length_)
              .round_inclusive(4096),
          desc_),
        _iobrd(std::move(iobrd_)),
        _iobs2(std::move(iobs2_)),
        _iobr2(std::move(iobr2_)),
        _pool(pool_),
        _auth_id{auth_id_},
        _offset{offset_},
        _buffer(static_cast<const char *>(buffer_)),
        _length(length_),
        _key{},
        _desc{this->desc()}  // provided by M
        ,
        _v{},
        _addr_list{},
        _addr_cursor{}
  {
      CPLOG(2, "%s iobs2 %p iobr2 %p"
        , __func__
        , static_cast<const void *>(&*_iobs2)
        , static_cast<const void *>(&*_iobr2)
      );
  }
  DELETE_COPY(async_buffer_set_put_direct_offset);
  int                         move_along(Connection_handler *c) override
  {
    if (iobs) { /* check submission, clear and free on completion */
      if (c->test_completion(&*iobs) == false) {
        return E_BUSY;
      }
      /* What to do when first send completes */
      iobs.reset(nullptr);
      /* End */
    }

    if (iobr) { /* check recv, clear and free on completion */
      if (c->test_completion(&*iobr) == false) {
        return E_BUSY;
      }
      /* What to do when first recv completes */
      const auto response = c->msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, "ASYNC OP_LOCATE");
      auto status = response->get_status();

      {
        auto cursor = response->edata();
        _addr_list  = std::vector<locate_element>(cursor, cursor + response->element_count());
        CPLOG(2,
          "%s::%s: edata count %zu %p to %p"
          , _cname
          , __func__
          , response->element_count()
          , static_cast<const void *>(cursor)
          , static_cast<const void *>(cursor + response->element_count())
        );
        _length = 0;
        for (const auto &e : _addr_list) {
          _length += e.len;
          CPLOG(2,
            "%s::%s: addr 0x%" PRIx64 " len 0x%" PRIx64
            , _cname
            , __func__
            , e.addr, e.len
          );
        }
      }

      _key         = response->key;
      _addr_cursor = _addr_list.begin();

      iobr.reset(nullptr);

      return status == S_OK ? E_BUSY : status;
    }

    if ( _addr_list.empty() )
    {
      return S_OK;
    }

    if (_addr_cursor != _addr_list.end()) {
      if (_iobrd && !c->test_completion(&*_iobrd)) {
        return E_BUSY;
      }

      _iobrd = c->make_iob_ptr_write();
      CPLOG(2, "%s iobrd %p"
        , __func__
        , static_cast<const void *>(&*_iobrd)
      );

      /* reply received, with credentials for the DMA */
      _v[0] = ::iovec{const_cast<char *>(_buffer), _addr_cursor->len};

      CPLOG(2,
        "%s::%s post_write %p local (addr %p.%zx desc %p) -> (_addr 0x%zx, key 0x%zx)"
        , _cname
        , __func__
        , static_cast<const void *>(&*_iobrd)
        , _v[0].iov_base, _v[0].iov_len
        , _desc[0]
        , _addr_cursor->addr
        , _key
      );

      c->post_write(std::begin(_v), std::end(_v), std::begin(_desc), _addr_cursor->addr, _key, &*_iobrd);
      _buffer += _addr_cursor->len;
      ++_addr_cursor;
      /* End */
    }

    if (_iobrd) {
      if (!c->test_completion(&*_iobrd)) {
        return E_BUSY;
      }
      /* What to do when DMA completes */
      /* DMA done. Might need another DMA */
      CPLOG(2
        , "%s::%s dma write complete %p"
        , _cname
        , __func__
        , static_cast<const void *>(&*_iobrd)
      );

      _iobrd.reset(nullptr);
      /* DMA is complete. Issue OP_RELEASE */

      /* send release message */
      const auto msg = new (_iobs2->base()) protocol::Message_IO_request(
          _auth_id, c->request_id(), _pool, protocol::OP_TYPE::OP_RELEASE_WITH_FLUSH, _offset, _length);

        c->post_recv(&*_iobr2);
        c->sync_inject_send(&*_iobs2, msg, __func__);
        /* End */
    }

    if ( _iobr2 ) {
      if ( ! c->test_completion(&*_iobr2) ) {
        return E_BUSY;
      }
      /* What to do when second recv completes */
      const auto response_msg = c->msg_recv<const mcas::protocol::Message_IO_response>(&*_iobr2, "OP_RELEASE_WITH_FLUSH,");
      return response_msg->get_status();
      /* End */
    }
    else {
      throw API_exception("invalid async handle, task already completed?");
    }
  }
};

Connection_handler::Connection_handler(unsigned                    debug_level_,
                                       Connection_base::Transport *connection,
                                       unsigned                    patience_)
    : Connection_base(debug_level_, connection, patience_),
#ifdef THREAD_SAFE_CLIENT
      _api_lock{},
#endif
      _exit{false},
      _request_id{0},
      _max_message_size{0},
      _max_inject_size(connection->max_inject_size()),
      _options()
{
  char *env = ::getenv("SHORT_CIRCUIT_BACKEND");
  if (env && env[0] == '1') {
    _options.short_circuit_backend = true;
  }
}

Connection_handler::~Connection_handler() { PLOG("%s: (%p)", __func__, static_cast<const void *>(this)); }

void Connection_handler::send_complete(void *param, buffer_t *iob)
{
  PLOG("%s param %p iob %p", __func__, param, static_cast<const void *>(iob));
}
void Connection_handler::recv_complete(void *param, buffer_t *iob)
{
  PLOG("%s param %p iob %p", __func__, param, static_cast<const void *>(iob));
}
void Connection_handler::write_complete(void *param, buffer_t *iob)
{
  PLOG("%s param %p iob %p", __func__, param, static_cast<const void *>(iob));
}
void Connection_handler::read_complete(void *param, buffer_t *iob)
{
  PLOG("%s param %p iob %p", __func__, param, static_cast<const void *>(iob));
}

Connection_handler::pool_t Connection_handler::open_pool(const std::string name,
                                                         const unsigned int  // flags
)
{
  API_LOCK();

  PMAJOR("open pool: %s", name.c_str());

  /* send pool request message */

  /* Use unique_ptr to ensure that the dynamically buffers are freed.
   * unique_ptr protects against the code forgetting to call free_buffer,
   * which it usually did when the function exited by a throw or a
   * non-terminal return.
   *
   */
  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();
  assert(iobs);
  assert(iobr);

  IKVStore::pool_t pool_id;

  assert(&*iobr != &*iobs);

  try {
    const auto msg =
        new (iobs->base()) mcas::protocol::Message_pool_request(iobs->length(), auth_id(), /* auth id */
                                                                request_id(), 0,           /* size */
                                                                0,                         /* expected obj count */
                                                                mcas::protocol::OP_OPEN, name, 0 /* flags */
        );

    /* The &* notation extracts a raw pointer form the "unique_ptr".
     * The difference is that the standard pointer does not imply
     * ownership; the function is simply "borrowing" the pointer.
     * Here, &*iobr is equivalent to iobr->get(). The choice is a
     * matter of style. &* uses two fewer tokens.
     */
    /* the sequence "post_recv, sync_inject_send, wait_for_completion"
     * is common enough that it probably deserves its own function.
     * But we may find that the entire single-exchange pattern as seen
     * in open_pool, create_pool and several others could be placed in
     * a single template function.
     */

    post_recv(&*iobr);
    sync_inject_send(&*iobs, msg, __func__);
    wait_for_completion(&*iobr); /* await response */

    const auto response_msg = msg_recv<const mcas::protocol::Message_pool_response>(&*iobr, __func__);

    pool_id = response_msg->pool_id;
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    pool_id = IKVStore::POOL_ERROR;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    pool_id = IKVStore::POOL_ERROR;
  }
  return pool_id;
}

Connection_handler::pool_t Connection_handler::create_pool(const std::string  name,
                                                           const size_t       size,
                                                           const unsigned int flags,
                                                           const uint64_t     expected_obj_count)
{
  API_LOCK();

  /* send pool request message */
  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();
  assert(iobs);
  assert(iobr);

  IKVStore::pool_t pool_id;

  try {
    const auto msg = new (iobs->base())
        protocol::Message_pool_request(iobs->length(), auth_id(), /* auth id */
                                       request_id(), size, expected_obj_count, mcas::protocol::OP_CREATE, name, flags);
    assert(msg->op());

    post_recv(&*iobr);
    sync_inject_send(&*iobs, msg, __func__);
    wait_for_completion(&*iobr);

    const auto response_msg = msg_recv<const mcas::protocol::Message_pool_response>(&*iobr, __func__);

    pool_id = response_msg->pool_id;
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    pool_id = IKVStore::POOL_ERROR;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    pool_id = IKVStore::POOL_ERROR;
  }

  /* Note: most request/response pairs return status. This one returns a pool_id
   * instead. */
  return pool_id;
}

status_t Connection_handler::close_pool(const pool_t pool)
{
  API_LOCK();
  /* send pool request message */
  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();
  const auto msg  = new (iobs->base())
      mcas::protocol::Message_pool_request(iobs->length(), auth_id(), request_id(), mcas::protocol::OP_CLOSE, pool);

  post_recv(&*iobr);
  sync_inject_send(&*iobs, msg, __func__);
  try {
    wait_for_completion(&*iobr);

    const auto response_msg = msg_recv<const mcas::protocol::Message_pool_response>(&*iobr, __func__);

    const auto status = response_msg->get_status();
    return status;
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    return E_FAIL;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    return E_FAIL;
  }
}

status_t Connection_handler::delete_pool(const std::string &name)

{
  if (name.empty()) return E_INVAL;

  API_LOCK();

  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();

  const auto msg = new (iobs->base()) mcas::protocol::Message_pool_request(iobs->length(), auth_id(), request_id(),
                                                                           0,  // size
                                                                           0,  // exp obj count
                                                                           mcas::protocol::OP_DELETE, name,
                                                                           0  // flags
  );

  post_recv(&*iobr);
  sync_inject_send(&*iobs, msg, __func__);
  try {
    wait_for_completion(&*iobr);

    const auto response_msg = msg_recv<const mcas::protocol::Message_pool_response>(&*iobr, __func__);

    return response_msg->get_status();
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    return E_FAIL;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    return E_FAIL;
  }
}

status_t Connection_handler::delete_pool(const IMCAS::pool_t pool)
{
  if (!pool) return E_INVAL;

  API_LOCK();

  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();

  const auto msg = new (iobs->base())
      mcas::protocol::Message_pool_request(iobs->length(), auth_id(), request_id(), mcas::protocol::OP_DELETE, pool);

  post_recv(&*iobr);
  sync_inject_send(&*iobs, msg, __func__);
  try {
    wait_for_completion(&*iobr);

    const auto response_msg = msg_recv<const mcas::protocol::Message_pool_response>(&*iobr, __func__);

    return response_msg->get_status();
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    return E_FAIL;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    return E_FAIL;
  }
}

status_t Connection_handler::configure_pool(const IMCAS::pool_t pool, const std::string &json)
{
  API_LOCK();

  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();

  if (!mcas::protocol::Message_IO_request::would_fit(json.length(), iobs->original_length())) {
    return IKVStore::E_TOO_LARGE;
  }

  const auto msg = new (iobs->base()) mcas::protocol::Message_IO_request(iobs->length(), auth_id(), request_id(), pool,
                                                                         mcas::protocol::OP_CONFIGURE,  // op
                                                                         json);

  post_recv(&*iobr);
  sync_inject_send(&*iobs, msg, __func__);
  try {
    wait_for_completion(&*iobr);
    const auto response_msg = msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, __func__);

    return response_msg->get_status();
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    return E_FAIL;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    return E_FAIL;
  }
}

/**
 * Memcpy version; both key and value are copied
 *
 */
status_t Connection_handler::put(const pool_t       pool,
                                 const std::string  key,
                                 const void *       value,
                                 const size_t       value_len,
                                 const unsigned int flags)
{
  if (value == nullptr || value_len == 0) return E_INVAL;
  return put(pool, key.c_str(), key.length(), value, value_len, flags);
}

status_t Connection_handler::put(const pool_t       pool,
                                 const void *       key,
                                 const size_t       key_len,
                                 const void *       value,
                                 const size_t       value_len,
                                 const unsigned int flags)
{
  API_LOCK();

  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();

  if (debug_level() > 1)
    PINF("put: %.*s (key_len=%lu) (value_len=%lu)", int(key_len), static_cast<const char *>(key), key_len, value_len);

  /* check key length */
  if (!mcas::protocol::Message_IO_request::would_fit(key_len + value_len, iobs->original_length())) {
    PWRN("mcas_client::%s value length (%lu) too long. Use put_direct.", __func__, value_len);
    return IKVStore::E_TOO_LARGE;
  }

  status_t status;

  try {
    const auto msg =
        new (iobs->base()) mcas::protocol::Message_IO_request(iobs->length(), auth_id(), request_id(), pool,
                                                              mcas::protocol::OP_PUT,  // op
                                                              key, key_len, value, value_len, flags);

    if (_options.short_circuit_backend) msg->add_scbe();

    iobs->set_length(msg->msg_len());

    post_recv(&*iobr);
    sync_send(&*iobs, msg, __func__); /* this will clean up iobs */
    wait_for_completion(&*iobr);

    const auto response_msg = msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, __func__);

    status = response_msg->get_status();
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    status = E_FAIL;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    status = E_FAIL;
  }

  return status;
}

auto Connection_handler::locate(const pool_t pool_, const std::size_t offset_, const std::size_t size_)
    -> std::tuple<uint64_t, std::vector<locate_element>>
{
  const auto iobr = make_iob_ptr_recv();
  const auto iobs = make_iob_ptr_send();

  /* send advance leader message */
  const auto msg = new (iobs->base())
      protocol::Message_IO_request(auth_id(), pool_, request_id(), protocol::OP_LOCATE, offset_, size_);

  post_recv(&*iobr);
  sync_inject_send(&*iobs, msg, __func__);
  /* wait for response from header before posting the value */
  wait_for_completion(&*iobr);

  const auto response = msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, __func__);

  if (response->get_status() != S_OK) {
    throw remote_fail(response->get_status());
  }

  auto                        cursor = response->edata();
  std::vector<locate_element> addr_list(cursor, cursor + response->element_count());

  return std::tuple<uint64_t, std::vector<locate_element>>(response->key, std::move(addr_list));
}

std::tuple<uint64_t, uint64_t, std::size_t> Connection_handler::get_locate(const pool_t   pool,
                                                                           const void *   key,
                                                                           const size_t   key_len,
                                                                           const unsigned flags)
{
  const auto iobr = make_iob_ptr_recv();
  const auto iobs = make_iob_ptr_send();

  /* send advance leader message */
  const auto msg = new (iobs->base()) protocol::Message_IO_request(iobs->length(), auth_id(), request_id(), pool,
                                                                   protocol::OP_GET_LOCATE, key, key_len, 0, flags);

  post_recv(&*iobr);
  sync_inject_send(&*iobs, msg, __func__);
  /* wait for response from header before posting the value */
  wait_for_completion(&*iobr);

  const auto response_msg = msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, __func__);

  if (response_msg->get_status() != S_OK) {
    throw remote_fail(msg->get_status());
  }
  return std::tuple<uint64_t, uint64_t, std::size_t>{response_msg->addr, response_msg->key,
                                                     response_msg->data_length()};
}

std::tuple<uint64_t, uint64_t> Connection_handler::put_locate(const pool_t   pool,
                                                              const void *   key,
                                                              const size_t   key_len,
                                                              const size_t   value_len,
                                                              const unsigned flags)
{
  const auto iobr = make_iob_ptr_recv();
  const auto iobs = make_iob_ptr_send();

  /* send advance leader message */
  const auto msg = new (iobs->base()) protocol::Message_IO_request(
      iobs->length(), auth_id(), request_id(), pool, protocol::OP_PUT_LOCATE, key, key_len, value_len, flags);

  post_recv(&*iobr);
  sync_inject_send(&*iobs, msg, __func__);
  /* wait for response from header before posting the value */
  wait_for_completion(&*iobr);

  const auto response_msg = msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, __func__);

  /* if response is not OK, don't follow with the value */
  if (response_msg->get_status() != S_OK) {
    throw remote_fail(msg->get_status());
  }

  return std::tuple<uint64_t, uint64_t>{response_msg->addr, response_msg->key};
}

IMCAS::async_handle_t Connection_handler::put_locate_async(const pool_t                        pool,
                                                           const void *                        key,
                                                           const size_t                        key_len,
                                                           const void *                        value,
                                                           const size_t                        value_len,
                                                           component::Registrar_memory_direct *rmd_,
                                                           void *const                         desc_,
                                                           const unsigned                      flags)
{
  auto iobr = make_iob_ptr_recv();
  auto iobs = make_iob_ptr_send();

  /* send locate message */
  const auto msg = new (iobs->base()) protocol::Message_IO_request(
      iobs->length(), auth_id(), request_id(), pool, protocol::OP_PUT_LOCATE, key, key_len, value_len, flags);
  iobs->set_length(msg->msg_len());

  post_recv(&*iobr);
  post_send(iobs->iov, iobs->iov + 1, iobs->desc, &*iobs, msg, __func__);

  /*
   * The entire put_locate protocol involves five completions at the client:
   *   send PUT_LOCATE request
   *   recv PUT_LOCATE response
   *   write DMA
   *   send PUT_RELEASE request
   *   recv PUT_RELEASE response
   */
  return desc_ ? static_cast<IMCAS::async_handle_t>(new async_buffer_set_put_locate<memory_registered_not_owned>(
                     debug_level(), rmd_, std::move(iobs), std::move(iobr), make_iob_ptr_write(), make_iob_ptr_send(),
                     make_iob_ptr_recv(), pool, auth_id(), value, value_len, desc_))
               : static_cast<IMCAS::async_handle_t>(new async_buffer_set_put_locate<memory_registered_owned>(
                     debug_level(), rmd_, std::move(iobs), std::move(iobr), make_iob_ptr_write(), make_iob_ptr_send(),
                     make_iob_ptr_recv(), pool, auth_id(), value, value_len, desc_));
}

IMCAS::async_handle_t Connection_handler::get_direct_offset_async(const pool_t                        pool_,
                                                                  const std::size_t                   offset_,
                                                                  void *const                         buffer_,
                                                                  std::size_t &                       len_,
                                                                  component::Registrar_memory_direct *rmd_,
                                                                  void *const                         desc_)
{
  auto iobr = make_iob_ptr_recv();
  auto iobs = make_iob_ptr_send();

  /* send advance leader message */
  const auto msg = new (iobs->base())
      protocol::Message_IO_request(auth_id(), request_id(), pool_, protocol::OP_LOCATE, offset_, len_);
  iobs->set_length(msg->msg_len());

  post_recv(&*iobr);
  post_send(iobs->iov, iobs->iov + 1, iobs->desc, &*iobs, msg, __func__);

  /*
   * The entire get_direct_offset protocol involves five completions at the
   * client: send LOCATE request recv LOCATE response read DMA send RELEASE
   * request recv RELEASE response
   */
  return desc_ ? static_cast<IMCAS::async_handle_t>(new async_buffer_set_get_direct_offset<memory_registered_not_owned>(
                     debug_level(), rmd_, std::move(iobs), std::move(iobr), iob_ptr(nullptr, this), make_iob_ptr_send(),
                     make_iob_ptr_recv(), pool_, auth_id(), offset_, buffer_, len_, desc_))
               : static_cast<IMCAS::async_handle_t>(new async_buffer_set_get_direct_offset<memory_registered_owned>(
                     debug_level(), rmd_, std::move(iobs), std::move(iobr), iob_ptr(nullptr, this), make_iob_ptr_send(),
                     make_iob_ptr_recv(), pool_, auth_id(), offset_, buffer_, len_, desc_));
}

IMCAS::async_handle_t Connection_handler::put_direct_offset_async(const pool_t                        pool_,
                                                                  const std::size_t                   offset_,
                                                                  const void *const                   buffer_,
                                                                  std::size_t &                       length_,
                                                                  component::Registrar_memory_direct *rmd_,
                                                                  void *const                         desc_)
{
  auto iobr = make_iob_ptr_recv();
  auto iobs = make_iob_ptr_send();

  /* send locate message */
  const auto msg = new (iobs->base())
      protocol::Message_IO_request(auth_id(), request_id(), pool_, protocol::OP_LOCATE, offset_, length_);
  iobs->set_length(msg->msg_len());

  post_recv(&*iobr);
  post_send(iobs->iov, iobs->iov + 1, iobs->desc, &*iobs, msg, __func__);

  /*
   * The entire get_direct_offset protocol involves five completions at the
   * client: send LOCATE request recv LOCATE response read DMA send RELEASE
   * request recv RELEASE response
   */
  return desc_ ? static_cast<IMCAS::async_handle_t>(new async_buffer_set_put_direct_offset<memory_registered_not_owned>(
                     debug_level(), rmd_, std::move(iobs), std::move(iobr), iob_ptr(nullptr, this), make_iob_ptr_send(),
                     make_iob_ptr_recv(), pool_, auth_id(), offset_, buffer_, length_, desc_))
               : static_cast<IMCAS::async_handle_t>(new async_buffer_set_put_direct_offset<memory_registered_owned>(
                     debug_level(), rmd_, std::move(iobs), std::move(iobr), iob_ptr(nullptr, this), make_iob_ptr_send(),
                     make_iob_ptr_recv(), pool_, auth_id(), offset_, buffer_, length_, desc_));
}

IMCAS::async_handle_t Connection_handler::get_locate_async( //
  const pool_t                        pool,
  const void *                        key,
  const size_t                        key_len,
  void *const                         value,
  size_t &                            value_len,
  component::Registrar_memory_direct *rmd_,
  void *const                         desc_,
  const unsigned                      flags)
{
  auto iobr = make_iob_ptr_recv();
  auto iobs = make_iob_ptr_send();
  const auto buffer_len = value_len;

  /* send advance leader message */
  const auto msg = new (iobs->base()) protocol::Message_IO_request(
      iobs->length(), auth_id(), request_id(), pool, protocol::OP_GET_LOCATE, key, key_len, value_len, flags);
  iobs->set_length(msg->msg_len());

  post_recv(&*iobr);
  post_send(iobs->iov, iobs->iov + 1, iobs->desc, &*iobs, msg, __func__);

  wait_for_completion(&*iobs);
  iobs.reset(nullptr);

  wait_for_completion(&*iobr);
  const auto response_msg = msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, "ASYNC GET_LOCATE");
  auto status = response_msg->get_status();
  if (status != S_OK) {
    throw remote_fail(status);
  }

  value_len = response_msg->data_length();
  auto addr = response_msg->addr;
  auto memory_key = response_msg->key;
  iobs.reset(nullptr);
  auto transfer_len = std::min(buffer_len, value_len);

  /*
   * The entire get_locate protocol involves five completions at the client:
   *   send GET_LOCATE request
   *   recv GET_LOCATE response
   *   read DMA
   *   send GET_RELEASE request
   *   recv GET_RELEASE response
   */
  return desc_ ? static_cast<IMCAS::async_handle_t>(new async_buffer_set_get_locate<memory_registered_not_owned>(
                     debug_level(), rmd_, make_iob_ptr_read(), make_iob_ptr_send(),
                     make_iob_ptr_recv(), pool, auth_id(), value, transfer_len, this, desc_, addr, memory_key))
               : static_cast<IMCAS::async_handle_t>(new async_buffer_set_get_locate<memory_registered_owned>(
                     debug_level(), rmd_, make_iob_ptr_read(), make_iob_ptr_send(),
                     make_iob_ptr_recv(), pool, auth_id(), value, transfer_len, this, desc_, addr, memory_key));
}

status_t Connection_handler::put_direct(const pool_t                              pool_,
                                        const void *const                         key_,
                                        const size_t                              key_len_,
                                        const void *const                         value_,
                                        const size_t                              value_len_,
                                        component::Registrar_memory_direct *const rmd_,
                                        const IMCAS::memory_handle_t              mem_handle_,
                                        const unsigned int                        flags_)
{
  component::IMCAS::async_handle_t async_handle = component::IMCAS::ASYNC_HANDLE_INIT;

  auto status = async_put_direct(pool_, key_, key_len_, value_, value_len_, async_handle, rmd_, mem_handle_, flags_);
  if (status == S_OK) {
    do {
      status = check_async_completion(async_handle);
    } while (status == E_BUSY);
  }
  return status;
}

status_t Connection_handler::async_put(const IMCAS::pool_t    pool,
                                       const void *           key,
                                       const size_t           key_len,
                                       const void *           value,
                                       const size_t           value_len,
                                       IMCAS::async_handle_t &out_handle,
                                       const unsigned int     flags)
{
  API_LOCK();

  if (debug_level() > 1)
    PINF("%s: %.*s (key_len=%lu) (value_len=%lu)", __func__, int(key_len), static_cast<const char *>(key), key_len,
         value_len);

  auto iobr = make_iob_ptr_recv();
  auto iobs = make_iob_ptr_send();

  /* check key length */
  if (!mcas::protocol::Message_IO_request::would_fit(key_len + value_len, iobs->original_length())) {
    PWRN("mcas_client::%s value length (%lu) too long. Use async_put_direct.", __func__, value_len);
    return IKVStore::E_TOO_LARGE;
  }

  try {
    const auto msg =
        new (iobs->base()) mcas::protocol::Message_IO_request(iobs->length(), auth_id(), request_id(), pool,
                                                              mcas::protocol::OP_PUT,  // op
                                                              key, key_len, value, value_len, flags);

    iobs->set_length(msg->msg_len());

    /* post both send and receive */
    post_recv(&*iobr);
    post_send(iobs->iov, iobs->iov + 1, iobs->desc, &*iobs, msg, __func__);

    out_handle = new async_buffer_set_simple(debug_level(), std::move(iobs), std::move(iobr));

    return S_OK;
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    throw Logic_exception("%s: network posting failed unexpectedly.", __func__);
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    throw Logic_exception("%s: network posting failed unexpectedly.", __func__);
  }

  return E_FAIL;
}

status_t Connection_handler::async_put_direct(const IMCAS::pool_t                        pool_,
                                              const void *const                          key_,
                                              const size_t                               key_len_,
                                              const void *const                          value_,
                                              const size_t                               value_len_,
                                              component::IMCAS::async_handle_t &         out_async_handle_,
                                              component::Registrar_memory_direct *       rmd_,
                                              const component::IKVStore::memory_handle_t mem_handle_,
                                              const unsigned int                         flags_)
{
  API_LOCK();

  assert(_max_message_size);

  if (pool_ == 0) {
    PWRN("%s: invalid pool identifier", __func__);
    return E_INVAL;
  }

  if (value_len_ && !value_) {
    PWRN("%s: bad parameter value=%p value_len=%zu", __func__, value_, value_len_);
    return E_BAD_PARAM;
  }

  try {
    auto iobr = make_iob_ptr_recv();
    auto iobs = make_iob_ptr_send();

    if (!mcas::protocol::Message_IO_request::would_fit(key_len_ + value_len_, iobs->original_length())) {
      /* check value is not too large for underlying transport */
      if (value_len_ > _max_message_size) {
        PWRN("%s: message size too large", __func__);
        return IKVStore::E_TOO_LARGE;
      }

      /* for large puts, where the receiver will not have
       * sufficient buffer space, we use a two-stage protocol */
      out_async_handle_ = put_locate_async(
          pool_, key_, key_len_, value_, value_len_, rmd_,
          mem_handle_ == IKVStore::HANDLE_NONE ? nullptr : static_cast<buffer_base *>(mem_handle_)->get_desc(), flags_);
    }
    else {
      CPLOG(1, "%s: key=(%.*s) key_len=%lu value=(%.20s...) value_len=%lu", __func__, int(key_len_),
        static_cast<const char *>(key_), key_len_, static_cast<const char *>(value_), value_len_);

      const auto msg =
          new (iobs->base()) mcas::protocol::Message_IO_request(iobs->length(), auth_id(), request_id(), pool_,
                                                                mcas::protocol::OP_PUT,  // op
                                                                key_, key_len_, value_len_, flags_);

      if (_options.short_circuit_backend) msg->add_scbe();

      iobs->set_length(msg->msg_len());

      post_recv(&*iobr);
      post_send(iobs->iov, iobs->iov + 1, iobs->desc, &*iobs, msg,
                __func__); /* send two concatentated buffers in single DMA */

      out_async_handle_ = new async_buffer_set_simple(debug_level(), std::move(iobs), std::move(iobr));
    }
    return S_OK;
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    return E_FAIL;
  }
  catch (const remote_fail &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    return e.status();
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    return E_FAIL;
  }
}

status_t Connection_handler::async_get_direct(const IMCAS::pool_t                        pool_,
                                              const void *const                          key_,
                                              const size_t                               key_len_,
                                              void *const                                value_,
                                              size_t &                                   value_len_,
                                              component::IMCAS::async_handle_t &         out_async_handle_,
                                              component::Registrar_memory_direct *       rmd_,
                                              const component::IKVStore::memory_handle_t mem_handle_,
                                              const unsigned int                         flags_)
{
  API_LOCK();

  if (value_len_ && !value_) {
    PWRN("%s: bad parameter value=%p value_len=%zu", __func__, value_, value_len_);
    return E_BAD_PARAM;
  }

  assert(_max_message_size);

  if (pool_ == 0) {
    PWRN("%s: invalid pool identifier", __func__);
    return E_INVAL;
  }

  try {
    auto iobr = make_iob_ptr_recv();
    auto iobs = make_iob_ptr_send();

    out_async_handle_ = get_locate_async(
        pool_, key_, key_len_, value_, value_len_, rmd_,
        mem_handle_ == IKVStore::HANDLE_NONE ? nullptr : static_cast<buffer_base *>(mem_handle_)->get_desc(), flags_);
    return S_OK;
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    return E_FAIL;
  }
  catch (const remote_fail &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    return e.status();
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    return E_FAIL;
  }
}

status_t Connection_handler::check_async_completion(IMCAS::async_handle_t &handle)
{
  API_LOCK();

  auto bptrs = static_cast<async_buffer_set_t *>(handle);
  assert(bptrs);

  int status = E_BUSY;
  try
  {
    status = bptrs->move_along(this);
    /* status will be one of
     * E_BUSY: the operattion is not finished, but the call to move_along may have
     * caused progress other: the operation has finished
     */
  }
  catch ( const Exception &e )
  {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    status = E_FAIL;
  }
  catch ( const std::exception &e )
  {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    status = E_FAIL;
  }

  if (status != E_BUSY) {
    delete bptrs;
  }

  return status;
}

status_t Connection_handler::get(const pool_t pool, const std::string &key, std::string &value)
{
  API_LOCK();

  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();
  assert(iobs);
  assert(iobr);

  status_t status;

  try {
    const auto msg =
        new (iobs->base()) mcas::protocol::Message_IO_request(iobs->length(), auth_id(), request_id(), pool,
                                                              mcas::protocol::OP_GET,  // op
                                                              key, "", 0);

    if (_options.short_circuit_backend) msg->add_scbe();

    post_recv(&*iobr);
    sync_inject_send(&*iobs, msg, __func__);
    wait_for_completion(&*iobr);

    const auto response_msg = msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, __func__);
#if 0
    /* NOTE: g++ might or might not be smart enough to skip the string argument evaluation if the
     *  debug_level() check in msg_recv_log will suppress use of the string.
     * If it is not, performance will suffer.
     */
    msg_recv_log(response_msg, __func__ + std::string(" ") + std::string(response_msg->data(), response_msg->data_length());
#endif
    status = response_msg->get_status();
    value.reserve(response_msg->data_length() + 1);
    value.insert(0, response_msg->cdata(), response_msg->data_length());
    assert(response_msg->data());
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    status = E_FAIL;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    status = E_FAIL;
  }

  return status;
}

status_t Connection_handler::get(const pool_t pool, const std::string &key, void *&value, size_t &value_len)
{
  API_LOCK();

  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();
  assert(iobs);
  assert(iobr);

  status_t status;

  try {
    const auto msg =
        new (iobs->base()) mcas::protocol::Message_IO_request(iobs->length(), auth_id(), request_id(), pool,
                                                              mcas::protocol::OP_GET,  // op
                                                              key.c_str(), key.length(), 0);

    /* indicate how much space has been allocated on this side. For
       get this is based on buffer size
    */
    msg->set_availabe_val_len_from_iob_len(iobs->original_length());

    if (_options.short_circuit_backend) msg->add_scbe();

    post_recv(&*iobr);
    sync_inject_send(&*iobs, msg, __func__);
    wait_for_completion(&*iobr); /* TODO; could we issue the recv and send together? */

    const auto response_msg = msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, __func__);

    if (response_msg->get_status() != S_OK) return response_msg->get_status();

    CPLOG(1, "%s: message value (%.*s)", __func__, int(response_msg->data_length()), response_msg->data());

    if (response_msg->is_set_twostage_bit()) {
      /* two-stage get */
      const auto data_len = response_msg->data_length() + 1;
      value               = ::aligned_alloc(MiB(2), data_len);
      if (value == nullptr) {
        throw std::bad_alloc();
      }
      madvise(value, data_len, MADV_HUGEPAGE);

      auto  region = make_memory_registered(value, data_len); /* we could have some pre-registered? */
      void *desc[] = {region.get_memory_descriptor()};

      ::iovec iov[]{{value, data_len - 1}};
      post_recv(std::begin(iov), std::end(iov), std::begin(desc), &iov[0]);

      /* synchronously wait for receive to complete */
      wait_for_completion(&iov);
      CPLOG(1, "%s Received value from two stage get", __func__);
    }
    else {
      /* copy off value from IO buffer */
      value = ::malloc(response_msg->data_length() + 1);
      if (value == nullptr) {
        throw std::bad_alloc();
      }
      value_len = response_msg->data_length();
      std::memcpy(value, response_msg->data(), response_msg->data_length());
      static_cast<char *>(value)[response_msg->data_length()] = '\0';
    }

    status = response_msg->get_status();
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    status = E_FAIL;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    status = E_FAIL;
  }

  return status;
}

status_t Connection_handler::get_direct(const pool_t                              pool_,
                                        const void *const                         key_,
                                        const size_t                              key_len_,
                                        void *const                               value_,
                                        size_t &                                  value_len_,
                                        component::Registrar_memory_direct *const rmd_,
                                        const IMCAS::memory_handle_t              mem_handle_)
{
  component::IMCAS::async_handle_t async_handle = component::IMCAS::ASYNC_HANDLE_INIT;

  auto status = async_get_direct(pool_, key_, key_len_, value_, value_len_, async_handle, rmd_, mem_handle_);
  if (status == S_OK) {
    do {
      status = check_async_completion(async_handle);
    } while (status == E_BUSY);
  }
  return status;
}

status_t Connection_handler::get_direct_offset(const pool_t                              pool_,
                                               const std::size_t                         offset_,
                                               std::size_t &                             length_,
                                               void *const                               buffer_,
                                               component::Registrar_memory_direct *const rmd_,
                                               const component::IMCAS::memory_handle_t   mem_handle_)
{
  component::IMCAS::async_handle_t async_handle = component::IMCAS::ASYNC_HANDLE_INIT;

  auto status = async_get_direct_offset(pool_, offset_, length_, buffer_, async_handle, rmd_, mem_handle_);
  if (status == S_OK) {
    do {
      status = check_async_completion(async_handle);
    } while (status == E_BUSY);
  }
  return status;
}

status_t Connection_handler::put_direct_offset(const pool_t                              pool_,
                                               const std::size_t                         offset_,
                                               std::size_t &                             size_,
                                               const void *const                         buffer_,
                                               component::Registrar_memory_direct *const rmd_,
                                               const component::IMCAS::memory_handle_t   mem_handle_)
{
  component::IMCAS::async_handle_t async_handle = component::IMCAS::ASYNC_HANDLE_INIT;

  auto status = async_put_direct_offset(pool_, offset_, size_, buffer_, async_handle, rmd_, mem_handle_);
  if (status == S_OK) {
    do {
      status = check_async_completion(async_handle);
    } while (status == E_BUSY);
  }
  return status;
}

status_t Connection_handler::async_get_direct_offset(const pool_t                              pool_,
                                                     const std::size_t                         offset_,
                                                     std::size_t &                             length_,
                                                     void *const                               buffer_,
                                                     IMCAS::async_handle_t &                   out_async_handle_,
                                                     component::Registrar_memory_direct *const rmd_,
                                                     const component::IMCAS::memory_handle_t   mem_handle_)
{
  if(length_ == 0)
    throw API_exception("%s: variant of get_direct_offset called with zero length", __func__);

  API_LOCK();

  if (length_ && !buffer_) {
    PWRN("%s: bad parameter buffer=%p size=%zu", __func__, buffer_, length_);
    return E_BAD_PARAM;
  }

  try {
    out_async_handle_ = get_direct_offset_async(
        pool_, offset_, buffer_, length_, rmd_,
        mem_handle_ == IMCAS::MEMORY_HANDLE_NONE ? nullptr : static_cast<buffer_base *>(mem_handle_)->get_desc());
    return S_OK;
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    return E_FAIL;
  }
  catch (const remote_fail &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    return e.status();
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    return E_FAIL;
  }
}

status_t Connection_handler::async_put_direct_offset(const pool_t                              pool_,
                                                     const std::size_t                         offset_,
                                                     std::size_t &                             length_,
                                                     const void *const                         buffer_,
                                                     IMCAS::async_handle_t &                   out_async_handle_,
                                                     component::Registrar_memory_direct *const rmd_,
                                                     const component::IMCAS::memory_handle_t   mem_handle_)
{
  if(length_ == 0)
    throw API_exception("%s: variant of put_direct_offset called with zero length", __func__);

  API_LOCK();

  if (length_ && !buffer_) {
    PWRN("%s: bad parameter buffer=%p size=%zu", __func__, buffer_, length_);
    return E_BAD_PARAM;
  }

  try {
    out_async_handle_ = put_direct_offset_async(
        pool_, offset_, buffer_, length_, rmd_,
        mem_handle_ == IMCAS::MEMORY_HANDLE_NONE ? nullptr : static_cast<buffer_base *>(mem_handle_)->get_desc());
    return S_OK;
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    throw Logic_exception("%s: network posting failed unexpectedly.", __func__);
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    throw Logic_exception("%s: network posting failed unexpectedly.", __func__);
  }
  return E_FAIL;
}

status_t Connection_handler::erase(const pool_t pool, const std::string &key)
{
  API_LOCK();

  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();
  assert(iobs);
  assert(iobr);

  status_t status;

  try {
    const auto msg = new (iobs->base()) mcas::protocol::Message_IO_request(
        iobs->length(), auth_id(), request_id(), pool, mcas::protocol::OP_ERASE, key.c_str(), key.length(), 0);

    post_recv(&*iobr);
    sync_inject_send(&*iobs, msg, __func__);
    wait_for_completion(&*iobr);

    const auto response_msg = msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, __func__);

    status = response_msg->get_status();
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    status = E_FAIL;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    status = E_FAIL;
  }

  return status;
}

status_t Connection_handler::async_erase(const IMCAS::pool_t    pool,
                                         const std::string &    key,
                                         IMCAS::async_handle_t &out_async_handle)
{
  API_LOCK();

  auto iobs = make_iob_ptr_send();
  auto iobr = make_iob_ptr_recv();

  assert(iobs);
  assert(iobr);

  try {
    const auto msg = new (iobs->base()) mcas::protocol::Message_IO_request(
        iobs->length(), auth_id(), request_id(), pool, mcas::protocol::OP_ERASE, key.c_str(), key.length(), 0);

    iobs->set_length(msg->msg_len());

    /* post both send and receive */
    post_recv(&*iobr);
    post_send(iobs->iov, iobs->iov + 1, iobs->desc, &*iobs, msg, __func__);

    out_async_handle = new async_buffer_set_simple(debug_level(), std::move(iobs), std::move(iobr));
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    throw Logic_exception("%s: network posting failed unexpectedly.", __func__);
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    throw Logic_exception("%s: network posting failed unexpectedly.", __func__);
  }

  return S_OK;
}

size_t Connection_handler::count(const pool_t pool)
{
  API_LOCK();

  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();
  assert(iobs);
  assert(iobr);

  try {
    const auto msg =
        new (iobs->base()) mcas::protocol::Message_INFO_request(auth_id(), IKVStore::Attribute::COUNT, pool);

    post_recv(&*iobr);
    sync_inject_send(&*iobs, msg, msg->base_message_size(), __func__);
    wait_for_completion(&*iobr);

    const auto response_msg = msg_recv<const mcas::protocol::Message_INFO_response>(&*iobr, __func__);

    return response_msg->value();
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    return 0;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    return 0;
  }
}

status_t Connection_handler::get_attribute(const IKVStore::pool_t    pool,
                                           const IKVStore::Attribute attr,
                                           std::vector<uint64_t> &   out_attr,
                                           const std::string *       key)
{
  API_LOCK();

  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();
  assert(iobs);
  assert(iobr);

  status_t status;

  try {
    const auto msg = new (iobs->base()) mcas::protocol::Message_INFO_request(auth_id(), attr, pool);

    if (key) msg->set_key(iobs->length(), *key);

    post_recv(&*iobr);
    sync_inject_send(&*iobs, msg, msg->message_size(), __func__);

    wait_for_completion(&*iobr);
    const auto response_msg = msg_recv<const mcas::protocol::Message_INFO_response>(&*iobr, __func__);

    out_attr.clear();
    out_attr.push_back(response_msg->value());
    status = response_msg->get_status();
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    status = E_FAIL;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    status = E_FAIL;
  }
  return status;
}

status_t Connection_handler::get_statistics(IMCAS::Shard_stats &out_stats)
{
  API_LOCK();

  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();
  assert(iobs);
  assert(iobr);

  status_t status;

  try {
    const auto msg =
        new (iobs->base()) mcas::protocol::Message_INFO_request(auth_id(), mcas::protocol::INFO_TYPE_GET_STATS, 0);

    post_recv(&*iobr);
    sync_inject_send(&*iobs, msg, msg->message_size(), __func__);

    wait_for_completion(&*iobr);
    const auto response_msg = msg_recv<const mcas::protocol::Message_stats>(&*iobr, __func__);

    status = response_msg->get_status();
#pragma GCC diagnostic push
#if defined __clang__ || 9 <= __GNUC__
#pragma GCC diagnostic ignored "-Waddress-of-packed-member"
#endif
    out_stats = response_msg->stats;
#pragma GCC diagnostic pop
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    status = E_FAIL;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    status = E_FAIL;
  }

  return status;
}

status_t Connection_handler::find(const IMCAS::pool_t pool,
                                  const std::string & key_expression,
                                  const offset_t      offset,
                                  offset_t &          out_matched_offset,
                                  std::string &       out_matched_key)
{
  API_LOCK();

  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();
  assert(iobs);
  assert(iobr);

  status_t status;

  try {
    const auto msg =
        new (iobs->base()) mcas::protocol::Message_INFO_request(auth_id(), mcas::protocol::INFO_TYPE_FIND_KEY, pool);
    msg->offset = offset;

    msg->set_key(iobs->length(), key_expression);

    post_recv(&*iobr);
    sync_inject_send(&*iobs, msg, msg->message_size(), __func__);

    wait_for_completion(&*iobr);
    const auto response_msg = msg_recv<const mcas::protocol::Message_INFO_response>(&*iobr, "FIND");

    status = response_msg->get_status();

    if (status == S_OK) {
      out_matched_key    = response_msg->c_str();
      out_matched_offset = response_msg->offset;
    }
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    status = E_FAIL;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    status = E_FAIL;
  }
  return status;
}

status_t Connection_handler::invoke_ado(const IKVStore::pool_t            pool,
                                        const std::string &               key,
                                        const void *                      request,
                                        const size_t                      request_len,
                                        const unsigned int                flags,
                                        std::vector<IMCAS::ADO_response> &out_response,
                                        const size_t                      value_size)
{
  API_LOCK();

  const auto iobs = make_iob_ptr_send();
  assert(iobs);

  try {
    const auto msg = new (iobs->base()) mcas::protocol::Message_ado_request(
        iobs->length(), auth_id(), request_id(), pool, key, request, request_len, flags, value_size);
    iobs->set_length(msg->message_size());

    if (flags & IMCAS::ADO_FLAG_ASYNC) {
      sync_send(&*iobs, msg, __func__);
      /* do not wait for response */
      return S_OK;
    }

    const auto iobr = make_iob_ptr_recv();
    assert(iobr);

    post_recv(&*iobr);
    sync_send(&*iobs, msg, __func__);
    wait_for_completion(&*iobr); /* wait for response */

    const auto response_msg = msg_recv<const mcas::protocol::Message_ado_response>(&*iobr, __func__);

    status_t status = response_msg->get_status();

    out_response.clear();

    if (status == S_OK) {
      /* unmarshall responses */
      for (uint32_t i = 0; i < response_msg->get_response_count(); i++) {
        void *   out_data     = nullptr;
        size_t   out_data_len = 0;
        uint32_t out_layer_id = 0;
        response_msg->client_get_response(i, out_data, out_data_len, out_layer_id);

#if defined DEBUG_NPC_RESPONSES
        PLOG("%s: Response:", __func__);
        hexdump(out_data, out_data_len);
#endif
        out_response.emplace_back(out_data, out_data_len, out_layer_id);
      }
    }
    else {
      if (response_msg->get_response_count() > 0) {
        void *   err_msg      = nullptr;
        size_t   err_msg_len  = 0;
        uint32_t out_layer_id = 0;
        response_msg->client_get_response(0, err_msg, err_msg_len, out_layer_id);
        PLOG("%s:%u ADO response status %d %.*s", __FILE__, __LINE__, status, int(err_msg_len),
             static_cast<const char *>(err_msg));
        ::free(err_msg);
      }
    }

    return status;
  }
  catch (const Exception &e) {
    PLOG("%s:%u ADO response Exception %s", __FILE__, __LINE__, e.cause());
    return E_FAIL;
  }
  catch (const std::exception &e) {
    PLOG("%s:%u ADO response exception %s", __FILE__, __LINE__, e.what());
    return E_FAIL;
  }
}

status_t Connection_handler::invoke_ado_async(const component::IMCAS::pool_t               pool,
                                              const std::string &                          key,
                                              const void *                                 request,
                                              const size_t                                 request_len,
                                              const component::IMCAS::ado_flags_t          flags,
                                              std::vector<component::IMCAS::ADO_response> &out_response,
                                              component::IMCAS::async_handle_t &           out_async_handle,
                                              const size_t                                 value_size)
{
  API_LOCK();

  auto iobs = make_iob_ptr_send();
  auto iobr = make_iob_ptr_recv();

  assert(iobs);
  assert(iobr);

  try {
    const auto msg = new (iobs->base()) mcas::protocol::Message_ado_request(
        iobs->length(), auth_id(), request_id(), pool, key, request, request_len, flags, value_size);
    iobs->set_length(msg->message_size());

    post_recv(&*iobr);
    post_send(&*iobs, msg, __func__);

    out_async_handle = new async_buffer_set_invoke(debug_level(), std::move(iobs), std::move(iobr), &out_response);

    return S_OK;
  }
  catch (const Exception &e) {
    PLOG("%s:%u ADO response Exception %s", __FILE__, __LINE__, e.cause());
    return E_FAIL;
  }
  catch (const std::exception &e) {
    PLOG("%s:%u ADO response exception %s", __FILE__, __LINE__, e.what());
    return E_FAIL;
  }
  catch (...) {
    return E_FAIL;
  }
}

status_t Connection_handler::invoke_put_ado(const IKVStore::pool_t            pool,
                                            const std::string &               key,
                                            const void *                      request,
                                            const size_t                      request_len,
                                            const void *                      value,
                                            const size_t                      value_len,
                                            const size_t                      root_len,
                                            const unsigned int                flags,
                                            std::vector<IMCAS::ADO_response> &out_response)
{
  API_LOCK();

  if (request_len == 0) return E_INVAL;

  const auto iobs = make_iob_ptr_send();
  assert(iobs);

  out_response.clear();

  status_t status;

  try {
    const auto msg = new (iobs->base()) mcas::protocol::Message_put_ado_request(
        iobs->length(), auth_id(), request_id(), pool, key, request, request_len, value, value_len, root_len, flags);

    iobs->set_length(msg->message_size());

    if (flags & IMCAS::ADO_FLAG_ASYNC) {
      sync_send(&*iobs, msg, __func__);
      /* do not wait for response */
      return S_OK;
    }

    const auto iobr = make_iob_ptr_recv();
    assert(iobr);

    post_recv(&*iobr);
    sync_send(&*iobs, msg, __func__);
    wait_for_completion(&*iobr); /* wait for response */

    const auto response_msg = msg_recv<const mcas::protocol::Message_ado_response>(&*iobr, __func__);

    status = response_msg->get_status();

    if (status == S_OK) {
      out_response.clear();

      /* unmarshal responses */
      for (uint32_t i = 0; i < response_msg->get_response_count(); i++) {
        void *   out_data     = nullptr;
        size_t   out_data_len = 0;
        uint32_t out_layer_id = 0;
        response_msg->client_get_response(i, out_data, out_data_len, out_layer_id);

#ifdef DEBUG_NPC_RESPONSES
        if (out_data_len > 0) {
          PLOG("Response:", __func__);
          hexdump(out_data, out_data_len);
        }
        else {
          PLOG("Response (inline): %p", __func__, out_data);
        }
#endif

        out_response.emplace_back(out_data, out_data_len, out_layer_id);
      }
    }
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    status = E_FAIL;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    status = E_FAIL;
  }

  return status;
}

auto Connection_handler::make_iob_ptr(buffer_t::completion_t completion_) -> iob_ptr
{
  return iob_ptr(allocate(completion_), this);
}

auto Connection_handler::make_iob_ptr_recv() -> iob_ptr
{
  return make_iob_ptr(recv_complete);
}
auto Connection_handler::make_iob_ptr_send() -> iob_ptr
{
  return make_iob_ptr(send_complete);
}
auto Connection_handler::make_iob_ptr_write() -> iob_ptr
{
  return make_iob_ptr(write_complete);
}
auto Connection_handler::make_iob_ptr_read() -> iob_ptr
{
  return make_iob_ptr(read_complete);
}

int Connection_handler::tick()
{
  using namespace mcas::protocol;

  switch (_state) {
    case INITIALIZE: {
      set_state(HANDSHAKE_SEND);
      break;
    }
    case HANDSHAKE_SEND: {

      const auto iob = make_iob_ptr_send();
      auto       msg = new (iob->base()) mcas::protocol::Message_handshake(auth_id(), 1);
      msg->set_status(S_OK);
      iob->set_length(msg->msg_len());
      post_send(iob->iov, iob->iov + 1, iob->desc, &*iob, msg, __func__);

      try {
        wait_for_completion(&*iob);
      }
      catch (...) {
        PERR("%s %s handshake send failed", __FILE__, __func__);
        set_state(STOPPED);
      }

      static int sent = 0;
      sent++;
      PMAJOR(">>> Sent handshake (%d)", sent);

      set_state(HANDSHAKE_GET_RESPONSE);
      break;
    }
    case HANDSHAKE_GET_RESPONSE: {
      const auto iobr = make_iob_ptr_recv();
      post_recv(iobr->iov, iobr->iov + 1, iobr->desc, &*iobr);

      try {
        wait_for_completion(&*iobr);
        const auto response_msg = msg_recv<const mcas::protocol::Message_handshake_reply>(&*iobr, "handshake");
        (void)response_msg;
      }
      catch (...) {
        PERR("%s %s handshake response failed", __FILE__, __func__);
        set_state(STOPPED);
      }

      static int recv = 0;
      recv++;


      set_state(READY);

      _max_message_size = max_message_size(); /* from fabric component */
      break;
    }
    case READY: {
      return 0;
      break;
    }
    case SHUTDOWN: {
      const auto iobs = make_iob_ptr_send();
      auto       msg  = new (iobs->base()) mcas::protocol::Message_close_session(reinterpret_cast<uint64_t>(this));

      iobs->set_length(msg->msg_len());
      post_send(iobs->iov, iobs->iov + 1, iobs->desc, &*iobs, msg, __func__);

      // server-side may have disappeared
      // try {
      //   wait_for_completion(&*iobs);
      // }
      // catch (...) {
      // }

      set_state(STOPPED);
      PLOG("Connection_handler::%s: connection %p shutdown.", __func__, static_cast<const void *>(this));
      return 0;
    }
    case STOPPED: {
      assert(0);
      return 0;
    }
  }  // end switch

  return 1;
}

}  // namespace client
}  // namespace mcas
