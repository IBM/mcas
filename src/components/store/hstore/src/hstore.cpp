/*
   Copyright [2017-2021] [IBM Corporation]
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

#include "hstore.h"

#include "clean_align.h"
#include "hop_hash.h"
#include "is_locked.h"
#include "key_not_found.h"
#include "logging.h"
#include "perishable.h"
#include "persist_atomic_controller.h"
#include "persist_fixed_string.h"
#include "pool_path.h"

#include "hstore_nupm_types.h"
#include "persister_nupm.h"

#include <common/errors.h>
#include <common/exceptions.h>
#include <common/perf/tm.h>
#include <common/utils.h>

#include <city.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <tbb/scalable_allocator.h> /* scalable_free */
#pragma GCC diagnostic pop

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstring> /* strerror, memcmp, memcpy */
#include <memory> /* unique_ptr */
#include <new>
#include <map> /* session set */
#include <mutex> /* thread safe use of libpmempool/obj */
#include <set>
#include <stdexcept> /* domain_error */

template<typename T>
  struct type_number;

template<> struct type_number<char> { static constexpr uint64_t value = 2; };

#if HEAP_RECONSTITUTE /* reconstituting allocator */
template<> struct type_number<impl::mod_control> { static constexpr std::uint64_t value = 4; };
#endif /* USE_CC_HEAP */

/* globals */

thread_local std::map<void *, hstore::open_pool_type *> tls_cache = {};

/* forced because pool_t is an integral tye, not a pointer */
void *to_ptr(component::IKVStore::pool_t p) { return reinterpret_cast<void *>(p); }
component::IKVStore::pool_t to_pool_t(void *v) { return reinterpret_cast<component::IKVStore::pool_t>(v); }

auto hstore::locate_session(const pool_t p) -> open_pool_type *
{
  auto *const v = to_ptr(p);
  auto it = tls_cache.find(v);
  if ( it == tls_cache.end() )
  {
    std::unique_lock<std::mutex> sessions_lk(_pools_mutex);
    auto ps = _pools.find(v);
    if ( ps == _pools.end() )
    {
      return nullptr;
    }
    it = tls_cache.emplace(v, ps->second.get()).first;
  }
  return it->second;
}

auto hstore::move_pool(const pool_t p) -> std::shared_ptr<open_pool_type>
{
  auto *const v = to_ptr(p);

  std::unique_lock<std::mutex> sessions_lk(_pools_mutex);
  auto ps = _pools.find(v);
  if ( ps == _pools.end() )
  {
    std::ostringstream o;
    o << "invalid pool identifier " << v;
    throw std::invalid_argument(o.str());
  }

  tls_cache.erase(v);
  auto s2 = ps->second;
  _pools.erase(ps);
  return s2;
}

hstore::hstore(
	unsigned debug_level_
#if HEAP_MM
	, const string_view mm_plugin_path_
#endif
	, const string_view owner_
	, const string_view name_
	, std::unique_ptr<dax_manager> &&mgr_
	)
  : common::log_source(debug_level_)
  , _pool_manager(std::make_shared<pm_type>(
		debug_level()
#if HEAP_MM
		, mm_plugin_path_
#endif
		, owner_
		, name_
		, std::move(mgr_))
	)
  , _pools_mutex{}
  , _pools{}
{
#if HEAP_MM
  PLOG("hstore plugin path(%s)", mm_plugin_path_.data());
#endif
}

hstore::~hstore()
{
}

auto hstore::thread_safety() const -> int
{
  return thread_model;
}

int hstore::get_capability(const Capability cap) const
{
  switch (cap)
  {
  case Capability::POOL_DELETE_CHECK: /*< checks if pool is open before allowing delete */
    return false;
  case Capability::RWLOCK_PER_POOL:   /*< pools are locked with RW-lock */
    return false;
  case Capability::POOL_THREAD_SAFE:  /*< pools can be shared across multiple client threads */
    return is_thread_safe;
  default:
    return -1;
  }
}

#include "session.h"

auto hstore::create_pool(const std::string & name_,
                         const std::size_t size_,
                         flags_t flags_,
                         const uint64_t expected_obj_count_,
                         Addr base_addr_unused) -> pool_t
try
{
  CPLOG(1, PREFIX "pool_name=%s size %zu", LOCATION, name_.c_str(), size_);
  try
  {
    _pool_manager->pool_create_check(size_);
  }
  catch ( const std::exception &e )
  {
    CPLOG(0, "%s: %s", __func__, e.what());
    return pool_t(E_FAIL);
  }

  auto path = pool_path(name_);

  auto rac = _pool_manager->pool_create_1(path, size_);
  auto s =
    std::static_pointer_cast<session_type>(
      std::shared_ptr<open_pool_type>(
        _pool_manager->pool_create_2(
          AK_INSTANCE
          rac
          , flags_ & ~(FLAGS_CREATE_ONLY|FLAGS_SET_SIZE)
          , expected_obj_count_
        ).release()
      )
    );

  std::unique_lock<std::mutex> sessions_lk(_pools_mutex);
  _pools.emplace(::base(rac.address_map().front()), s);

  return to_pool_t(::base(rac.address_map().front()));
}
catch ( const pool_error &e )
{
  return e.value() != int(pool_ec::region_fail) || flags_ & FLAGS_CREATE_ONLY
    ? static_cast<IKVStore::pool_t>(POOL_ERROR)
    : open_pool(name_, flags_ & ~FLAGS_SET_SIZE, base_addr_unused)
    ;
}
catch ( const std::bad_alloc &e )
{
  CPLOG(0, "%s: %s", __func__, e.what());
  return POOL_ERROR; // E_TOO_LARGE incorrect type
}

auto hstore::open_pool(const std::string &name_,
                       flags_t flags,
                       Addr) -> pool_t
{
  auto path = pool_path(name_);
  try {
    auto v = _pool_manager->pool_open_1(path);

    std::unique_lock<std::mutex> sessions_lk(_pools_mutex);
    /* open pools are indexed by the base address of their first (contiguous) segment */
    if ( v.address_map().empty() )
    {
      return POOL_ERROR; /* pool not found (by name) */
    }
    else
    {
      auto it = _pools.find(::base(v.address_map().front()));
      if ( it != _pools.end() )
      {
        /* already have a session, make a copy of the pointer */
        _pools.emplace(::base(v.address_map().front()), it->second);
      }
      else
      {
        /* no session yet, create one */
        auto s = _pool_manager->pool_open_2(AK_INSTANCE v, flags);
        /* explicit conversion to shared_ptr fpr g++ 5 */
        _pools.emplace(::base(v.address_map().front()), std::shared_ptr<open_pool_type>(s.release()));
      }
      return to_pool_t(::base(v.address_map().front()));
    }
  }
  catch( const pool_error &e ) {
    CPLOG(0, "%s: %s", __func__, e.message().c_str());
    return POOL_ERROR;
  }
  catch( const std::invalid_argument &e ) {
    CPLOG(0, "%s: %s", __func__, e.what());
    return POOL_ERROR;
  }
  catch ( const std::bad_alloc &e )
  {
    CPLOG(0, "%s: %s", __func__, e.what());
    return POOL_ERROR; // E_TOO_LARGE incorrect type
  }
}

status_t hstore::close_pool(const pool_t p)
{
  std::string path;
  try
  {
    auto pool = move_pool(p);
    CPLOG(1, PREFIX "closed pool (%" PRIxIKVSTORE_POOL_T ")", LOCATION, p);
    _pool_manager->pool_close_check(path);
  }
  catch ( const std::runtime_error &e )  {
    CPLOG(0, "%s: %s", __func__, e.what());
    return E_POOL_NOT_FOUND;
  }
  catch ( const std::invalid_argument &e )  {
    CPLOG(0, "%s: %s", __func__, e.what());
    return E_INVAL;
  }

  return S_OK;
}

status_t hstore::delete_pool(const std::string& name_)
{
  auto path = pool_path(name_);

  try {
    _pool_manager->pool_delete(path);
  }
  catch ( const std::runtime_error &e )  {
    CPLOG(0, "%s: %s", __func__, e.what());
    return E_POOL_NOT_FOUND;
  }
  catch ( const std::invalid_argument &e )  {
    CPLOG(0, "%s: %s", __func__, e.what());
    return E_INVAL;
  }

  CPLOG(1, PREFIX "pool deleted: %s", LOCATION, name_.c_str());
  return S_OK;
}

auto hstore::grow_pool( //
  const pool_t pool,
  const std::size_t increment_size,
  std::size_t& reconfigured_size ) -> status_t
{
  const auto session = static_cast<session_type *>(locate_session(pool));
  if ( ! session )
  {
    return E_POOL_NOT_FOUND;
  }
  try
  {
    reconfigured_size = session->pool_grow(_pool_manager->get_dax_manager(), increment_size);
  }
  catch ( const std::bad_alloc &e )
  {
    CPLOG(0, "%s: %s", __func__, e.what());
    return E_TOO_LARGE; /* would be E_NO_MEM, if it were in the interface */
  }
  return S_OK;
}

auto hstore::put(const pool_t pool,
                 const std::string &key,
                 const void * value,
                 const std::size_t value_len,
                 flags_t flags) -> status_t
{
  TM_ROOT()
  CPLOG(
    1
    , PREFIX "(key=%s) (value=%.*s)"
    , LOCATION
    , key.c_str()
    , int(value_len)
    , static_cast<const char*>(value)
  );

  /* Strangely, zero is not allowed as a value length */
  if ( value_len == 0 )
  {
    return E_BAD_PARAM;
  }

  if ( (flags & ~FLAGS_DONT_STOMP) != 0 )
  {
    return E_BAD_PARAM;
  }
  if ( value == nullptr )
  {
    return E_BAD_PARAM;
  }

  const auto session = static_cast<session_type *>(locate_session(pool));

  if ( session )
  {
    try
    {
      TM_SCOPE(insert_or_update)
      auto it = session->insert(AK_INSTANCE TM_REF key, value, value_len);

      TM_SCOPE(update)
      return
        it.second                  ? S_OK
        : flags & FLAGS_DONT_STOMP ? int(E_KEY_EXISTS)
        : (
            session->update_by_issue_41(
              AK_INSTANCE
              TM_REF
              key
              , value
              , value_len
              , std::get<0>(it.first->second).data()
              , std::get<0>(it.first->second).size())
            , S_OK
          )
        ;
    }
    catch ( const std::bad_alloc &e )
    {
      CPLOG(0, "%s: %s", __func__, e.what());
      return E_TOO_LARGE; /* would be E_NO_MEM, if it were in the interface */
    }
    catch ( const std::invalid_argument &e )
    {
      CPLOG(0, "%s: %s", __func__, e.what());
      return E_NOT_SUPPORTED;
    }
    catch ( const impl::is_locked &e )
    {
      CPLOG(0, "%s: %s", __func__, e.what());
      return E_LOCKED; /* ... and is locked, so cannot be updated */
    }
  }
  else
  {
    return E_POOL_NOT_FOUND;
  }
}

auto hstore::get_pool_regions(const pool_t pool, nupm::region_descriptor & out_regions) -> status_t
{
  const auto session = static_cast<session_type *>(locate_session(pool));
  if ( ! session )
  {
    return E_POOL_NOT_FOUND;
  }
  out_regions = _pool_manager->pool_get_regions(session->handle());
  return S_OK;
}

auto hstore::put_direct(const pool_t pool,
                        const std::string& key,
                        const void * value,
                        const std::size_t value_len,
                        memory_handle_t,
                        flags_t flags) -> status_t
{
  return put(pool, key, value, value_len, flags);
}

auto hstore::get(const pool_t pool,
                 const std::string &key,
                 void*& out_value,
                 std::size_t& out_value_len) -> status_t
{
  TM_ROOT()
  const auto session = static_cast<const session_type *>(locate_session(pool));
  if ( ! session )
  {
    return E_POOL_NOT_FOUND;
  }

  try
  {
    /* Although not documented, assume that non-zero
     * out_value implies that out_value_len holds
     * the buffer's size.
     */
    if ( out_value )
    {
      auto buffer_size = out_value_len;
      out_value_len = session->get(TM_REF key, out_value, buffer_size);
      /*
       * It might be reasonable to
       *  a) fill the buffer and/or
       *  b) return the necessary size in out_value_len,
       * but neither action is documented, so we do not.
       */
      if ( buffer_size < out_value_len )
      {
        return E_INSUFFICIENT_BUFFER;
      }
    }
    else
    {
      try
      {
        auto r = session->get_alloc(key);
        out_value = std::get<0>(r);
        out_value_len = std::get<1>(r);
      }
      catch ( const std::bad_alloc &e )
      {
        CPLOG(0, "%s: %s", __func__, e.what());
        return E_TOO_LARGE; /* would be E_NO_MEM, if it were in the interface */
      }
    }
    return S_OK;
  }
  catch ( const impl::key_not_found &e )
  {
    CPLOG(0, "%s: %s", __func__, e.what());
    return E_KEY_NOT_FOUND;
  }
}

auto hstore::get_direct(const pool_t pool,
                        const std::string & key,
                        void* out_value,
                        std::size_t& out_value_len,
                        memory_handle_t) -> status_t
{
  TM_ROOT()
  const auto session = static_cast<const session_type *>(locate_session(pool));
  if ( ! session )
  {
    return E_POOL_NOT_FOUND;
  }

  try
  {
    const auto buffer_size = out_value_len;
    out_value_len = session->get(TM_REF key, out_value, buffer_size);
    if ( buffer_size < out_value_len )
    {
      return E_INSUFFICIENT_BUFFER;
    }
    return S_OK;
  }
  catch ( const impl::key_not_found &e )
  {
    CPLOG(0, "%s: %s", __func__, e.what());
    return E_KEY_NOT_FOUND;
  }
}

auto hstore::get_attribute(
  const pool_t pool,
  const Attribute attr,
  std::vector<uint64_t>& out_attr,
  const std::string* key) -> status_t
{
  out_attr.clear();

  const auto session = static_cast<const session_type *>(locate_session(pool));
  if ( ! session )
  {
    return E_POOL_NOT_FOUND;
  }

  switch ( attr )
  {
  case MEMORY_TYPE:
    {
      out_attr.push_back(MEMORY_TYPE_PMEM_DEVDAX);
      return S_OK;
    }
  case VALUE_LEN:
    if ( ! key )
    {
      return E_BAD_PARAM;
    }
    try
    {
      /* interface does not say what we do to the out_attr vector;
       * push_back is at least non-destructive.
       */
      out_attr.push_back(session->get_value_len(*key));
      return S_OK;
    }
    catch ( const impl::key_not_found &e )
    {
      CPLOG(0, "%s: %s", __func__, e.what());
      return E_KEY_NOT_FOUND;
    }
    catch ( const std::bad_alloc &e )
    {
      CPLOG(0, "%s: %s", __func__, e.what());
      return E_TOO_LARGE; /* would be E_NO_MEM, if it were in the interface */
    }
    break;
  case AUTO_HASHTABLE_EXPANSION:
    try
    {
      out_attr.push_back(session->get_auto_resize());
      return S_OK;
    }
    catch ( const std::bad_alloc &e )
    {
      CPLOG(0, "%s: %s", __func__, e.what());
      return E_TOO_LARGE; /* would be E_NO_MEM, if it were in the interface */
    }
    break;
  case PERCENT_USED:
    out_attr.push_back(session->percent_used());
    return S_OK;
    break;
#if ENABLE_TIMESTAMPS
  case IKVStore::Attribute::WRITE_EPOCH_TIME:
    if ( ! key )
    {
      return E_BAD_PARAM;
    }
    try
    {
      out_attr.push_back(session->get_write_epoch_time(*key));
      return S_OK;
    }
    catch ( const impl::key_not_found &e )
    {
      CPLOG(0, "%s: %s", __func__, e.what());
      return E_KEY_NOT_FOUND;
    }
    break;
#endif
  default:
    return E_NOT_SUPPORTED;
  }
  assert(nullptr == "missing return");
}

auto hstore::set_attribute(
  const pool_t pool,
  const Attribute attr
  , const std::vector<uint64_t> & value
  , const std::string *) -> status_t
{
  auto session = static_cast<session_type *>(locate_session(pool));
  if ( ! session )
  {
    return E_POOL_NOT_FOUND;
  }
  switch ( attr )
  {
  case AUTO_HASHTABLE_EXPANSION:
    if ( value.size() < 1 )
    {
      return E_BAD_PARAM;
    }
    {
      session->set_auto_resize(bool(value[0]));
      return S_OK;
    }
  default:
    return E_NOT_SUPPORTED;
  }
  assert(nullptr == "missing return");
}

auto hstore::resize_value(
  const pool_t pool
  , const std::string &key
  , const std::size_t new_value_len
  , const std::size_t alignment
) -> status_t
{
  TM_ROOT()
  const auto session = static_cast<session_type *>(locate_session(pool));
  try
  {
    return
      session
      ? ( session->resize_mapped(AK_INSTANCE TM_REF key, new_value_len, clean_align(alignment)), S_OK )
      : E_FAIL
      ;
  }
  /* how might this fail? Out of memory, key not found, not locked, read locked */
  catch ( const std::invalid_argument &e )
  {
    CPLOG(0, "%s: %s", __func__, e.what());
    return E_BAD_ALIGNMENT; /* bad alignment, probably */
  }
  catch ( const std::bad_alloc &e )
  {
    CPLOG(0, "%s: %s", __func__, e.what());
    return E_TOO_LARGE; /* would be E_NO_MEM, if it were in the interface */
  }
  catch ( const impl::key_not_found &e )
  {
    CPLOG(0, "%s: %s", __func__, e.what());
    return E_KEY_NOT_FOUND; /* key not found */
  }
  catch ( const impl::is_locked &e )
  {
    CPLOG(0, "%s: %s", __func__, e.what());
    return E_LOCKED; /* could not get unique lock (?) */
  }
}

auto hstore::lock(
  const pool_t pool
  , const std::string &key
  , lock_type_t type
  , void *& out_value
  , std::size_t & out_value_len
  , std::size_t alignment
  , key_t& out_key
  , const char ** out_key_ptr
) -> status_t
try
{
  TM_ROOT()
  const auto session = static_cast<session_type *>(locate_session(pool));
  if(!session) return E_FAIL;
	/* 0 probably means that alignment is a don't care, which is the same as alignment 1 */
	if ( alignment == 0 ) { alignment = 1; }
#if 0 /* As with mapstore, allocator (not hstore) deals with non-2^n alignments */
	if ( ( alignment & (alignment - 1) ) != 0  ) { return E_BAD_ALIGNMENT; }
#endif
  auto r = session->lock(AK_INSTANCE TM_REF key, type, out_value, out_value_len, alignment);

  out_key = r.key;
  if ( out_key_ptr )
  {
    *out_key_ptr = r.key_ptr;
  }
  /* If lock valid, safe to provide access to the key */
  if ( r.key != KEY_NONE )
  {
    out_value = r.value;
    out_value_len = r.value_len;
  }

  switch ( r.state )
  {
  case lock_result::e_state::created:
    /* Returns undocumented "E_LOCKED" if lock not held */
    return r.key == KEY_NONE ? E_LOCKED : S_OK_CREATED;
  case lock_result::e_state::not_created:
    return E_KEY_NOT_FOUND;
  case lock_result::e_state::extant:
    /* Returns undocumented "E_LOCKED" if lock not held */
    return r.key == KEY_NONE ? E_LOCKED : S_OK;
  case lock_result::e_state::creation_failed:
    /* should not happen. */
    return E_KEY_EXISTS;
#if 0 /* not a separate error */
  case lock_result::e_state::misaligned:
    /* should not happen. */
    return E_MISALIGNED;
#endif
  }
  return E_KEY_NOT_FOUND;
}
catch ( const std::bad_alloc &e )
{
  CPLOG(0, "%s: %s", __func__, e.what());
  return E_TOO_LARGE;
}

auto hstore::unlock(const pool_t pool,
                    key_t key_,
                    unlock_flags_t flags_) -> status_t
{
  TM_ROOT()
  /* NOTE: if flags & UNLOCK_FLAGS_PMFLUSH, only flush if the lock held
     is a write lock
  */
  const auto session = static_cast<session_type *>(locate_session(pool));
  return
    session
    ? session->unlock_indefinite(TM_REF key_, flags_)
    : E_POOL_NOT_FOUND
    ;
}

auto hstore::erase(const pool_t pool,
                   const std::string &key
                   ) -> status_t
{
  TM_ROOT()
  const auto session = static_cast<session_type *>(locate_session(pool));
  return session
    ? session->erase(TM_REF key)
    : E_POOL_NOT_FOUND
    ;
}

std::size_t hstore::count(const pool_t pool)
{
  const auto session = static_cast<session_type *>(locate_session(pool));
  if ( ! session )
  {
    return std::size_t(E_POOL_NOT_FOUND);
  }

  return session->count();
}

void hstore::debug(const pool_t, const unsigned cmd, const uint64_t arg)
{
  switch ( cmd )
    {
    case 0:
      perishable::enable(bool(arg));
      break;
    case 1:
      perishable::reset(arg);
      break;
    case 2:
      {
      }
      break;
    default:
      break;
    };
}

auto hstore::map(
                 pool_t pool,
                 std::function
                 <
                   int(const void * key, std::size_t key_len,
                       const void * val, std::size_t val_len)
                 > f_
                 ) -> status_t
{
  const auto session = static_cast<session_type *>(locate_session(pool));

  return session
    ? ( session->map(f_), S_OK )
    : int(E_POOL_NOT_FOUND)
    ;
}

auto hstore::map(
  pool_t pool_,
  std::function<
    int(
      const void * key,
      std::size_t key_len,
      const void * value,
      std::size_t value_len,
      common::tsc_time_t timestamp
    )
  > f_,
  common::epoch_time_t t_begin_,
  common::epoch_time_t t_end_
) -> status_t
{
  const auto session = static_cast<session_type *>(locate_session(pool_));

  return session
    ? ( session->map(f_, t_begin_, t_end_) ? S_OK : E_NOT_SUPPORTED )
    : int(E_POOL_NOT_FOUND)
    ;
}

auto hstore::map_keys(
                 pool_t pool,
                 std::function
                 <
                   int(const std::string &key)
                 > f_
                 ) -> status_t
{
  const auto session = static_cast<session_type *>(locate_session(pool));

  return session
    ? ( session->map([&f_] (const void * key, std::size_t key_len,
                            const void *, std::size_t) -> int
                     {
                       f_(std::string(static_cast<const char*>(key), key_len));
                       return 0;
                     }), S_OK )
    : int(E_POOL_NOT_FOUND)
    ;
}

auto hstore::free_memory(void * p) -> status_t
{
  scalable_free(p);
  return S_OK;
}

auto hstore::atomic_update(
    const pool_t pool
    , const std::string& key
    , const std::vector<IKVStore::Operation *> &op_vector
    , const bool take_lock) -> status_t
try
{
  TM_ROOT(hs_atomic_update)
  using op_it_type = std::vector<IKVStore::Operation *>::const_iterator;
  const auto update_method =
    take_lock
    ? &session_type::lock_and_atomic_update<op_it_type>
    : &session_type::atomic_update<op_it_type>
    ;
  const auto session = static_cast<session_type *>(locate_session(pool));
  return
    session
    ? ( (session->*update_method)(AK_INSTANCE TM_REF key, op_vector.begin(), op_vector.end()), S_OK )
    : int(E_POOL_NOT_FOUND)
    ;
}
catch ( const std::bad_alloc &e )
{
  CPLOG(0, "%s: %s", __func__, e.what());
  return E_TOO_LARGE; /* would be E_NO_MEM, if it were in the interface */
}
catch ( const std::invalid_argument &e )
{
  CPLOG(0, "%s: %s", __func__, e.what());
  return E_NOT_SUPPORTED;
}
catch ( const impl::key_not_found &e )
{
  CPLOG(0, "%s: %s", __func__, e.what());
  return E_KEY_NOT_FOUND;
}
catch ( const impl::is_locked &e )
{
  CPLOG(0, "%s: %s", __func__, e.what());
  return E_LOCKED; /* ... is locked, so cannot be updated */
}
catch ( const std::system_error &e )
{
  CPLOG(0, "%s: %s", __func__, e.what());
  return E_FAIL;
}

auto hstore::swap_keys(
  const pool_t pool
  , const std::string key0
  , const std::string key1
) -> status_t
try
{
  TM_ROOT()
  const auto session = static_cast<session_type *>(locate_session(pool));
  return
    session
    ? session->swap_keys(AK_INSTANCE TM_REF key0, key1)
    : int(E_POOL_NOT_FOUND)
    ;
}
catch ( const std::bad_alloc &e )
{
  CPLOG(0, "%s: %s", __func__, e.what());
  return E_TOO_LARGE; /* would be E_NO_MEM, if it were in the interface */
}

auto hstore::allocate_pool_memory(
  const pool_t pool,
  const size_t size,
  const size_t alignment,
  void * & out_addr
) -> status_t
try
{
  const auto session = static_cast<session_type *>(locate_session(pool));
  return
    session
    ? ( out_addr = session->allocate_memory(AK_INSTANCE size, clean_align(alignment, sizeof(void *))), S_OK )
    : int(E_POOL_NOT_FOUND)
    ;
}
catch ( const std::invalid_argument &e )
{
  CPLOG(0, "%s: %s", __func__, e.what());
  return E_BAD_ALIGNMENT; /* ... probably */
}
catch ( const std::bad_alloc &e )
{
  CPLOG(0, "%s: %s", __func__, e.what());
  return E_TOO_LARGE; /* would be E_NO_MEM, if it were in the interface */
}

auto hstore::free_pool_memory(
  const pool_t pool,
  const void* const addr,
  const size_t size
) -> status_t
try
{
  const auto session = static_cast<session_type *>(locate_session(pool));
  return
    session
    ? ( session->free_memory(addr, size), S_OK )
    : int(E_POOL_NOT_FOUND)
    ;
}
catch ( const API_exception &e ) /* bad pointer */
{
  CPLOG(0, "%s: %s", __func__, e.cause());
  return E_INVAL;
}
catch ( const std::exception &e )
{
  CPLOG(0, "%s: %s", __func__, e.what());
  return E_FAIL;
}

auto hstore::flush_pool_memory(
  const pool_t pool,
  const void* const addr,
  const size_t size
) -> status_t
try
{
  const auto session = static_cast<session_type *>(locate_session(pool));
  return
    session
    ? ( session->flush_memory(addr, size), S_OK )
    : int(E_POOL_NOT_FOUND)
    ;
}
catch ( const API_exception &e ) /* bad pointer */
{
  CPLOG(0, "%s: %s", __func__, e.cause());
  return E_INVAL;
}
catch ( const std::exception &e )
{
  CPLOG(0, "%s: %s", __func__, e.what());
  return E_FAIL;
}

auto hstore::open_pool_iterator(pool_t pool) -> pool_iterator_t
{
  auto session = static_cast<session_type *>(locate_session(pool));
  return
    session
    ? session->open_iterator()
    : nullptr
    ;
}

status_t hstore::deref_pool_iterator(
  const pool_t pool
  , pool_iterator_t iter
  , const common::epoch_time_t t_begin
  , const common::epoch_time_t t_end
  , pool_reference_t & ref
  , bool & time_match
  , bool increment
)
{
  const auto session = static_cast<session_type *>(locate_session(pool));
  return
    session
    ? session->deref_iterator(
        iter
        , t_begin
        , t_end
        , ref
        , time_match
        , increment
      )
    : E_INVAL
    ;
}

status_t  hstore::close_pool_iterator(
  const pool_t pool
  , pool_iterator_t iter
)
{
  auto session = static_cast<session_type *>(locate_session(pool));
  return
    session
    ? session->close_iterator(iter)
    : E_INVAL
    ;
}

