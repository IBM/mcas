#ifndef __CCPM_IMMUTABLE_ALLOCATOR_H__
#define __CCPM_IMMUTABLE_ALLOCATOR_H__

#include <libpmem.h>
#include <api/ado_itf.h>
#include <common/utils.h>
#include <common/dump_utils.h>
#include <common/logging.h>
#include <exception>
#include <atomic>
#include <ccpm/interfaces.h>
#include <EASTL/tracker.h>


namespace ccpm
{
class Immutable_allocator_base : public IHeap
{
	const uint32_t MAGIC = 0xF005BA11;
	const unsigned debug_level = 0;

private:

  struct Immutable_slab {
    uint32_t              magic;
    uint32_t              resvd;
    byte *                slab_end;
    Immutable_allocator_base * linked_slab;
    std::atomic<byte *>   next_free;
  } * _root __attribute__((packed));

  static_assert(sizeof(Immutable_slab) == 32, "bad size Immutable_slab structure");
  bool _rebuilt = false;
public:

  // _root is not initialized
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
  Immutable_allocator_base(region_vector_t regions,
                           ccpm::ownership_callback_t callback,
                           bool force_init)
  {
    (void)debug_level; // unused
    reconstitute(regions, callback, force_init);
    assert(_root);
  }

  explicit Immutable_allocator_base(::iovec& region)
  {
    region_vector_t regions;
    regions.push_back(region);
    reconstitute(regions, nullptr, true /* this is new - force initialization */);
    assert(_root);
  }
#pragma GCC diagnostic pop

  

  Immutable_allocator_base(Immutable_allocator_base& src) : _root(src._root)
  {
    assert(_root);
  }

  Immutable_allocator_base& operator=(const Immutable_allocator_base& src)
  {
    _root = src._root;
    assert(_root);
    return *this;
  }

  virtual ~Immutable_allocator_base() {
  }

  /**
   * @brief      Force full slab persistence
   */
  void persist() {
    pmem_persist(_root, std::size_t(reinterpret_cast<byte*>(_root) - _root->slab_end));
  }

  /* Reconstitute allocator from existing memory (ctor)
   * @param buffer Buffer
   * @param size Size of buffer in bytes
   * @param callback Callback to check ownership
   * @param force_init Iff true force reset of allocator
   *
   *
   * @return : True iff re-initialization took place
   **/
  bool reconstitute(const region_vector_t &regions,
                    ccpm::ownership_callback_t callback,
                    const bool force_init) override
  {
    if(regions.size() == 0)
      throw std::invalid_argument("invalid regions parameter");
    

    void * buffer = regions[0].iov_base;
    size_t size = regions[0].iov_len;
    assert(size > 0);
    assert(buffer);

    _root = static_cast<Immutable_slab*>(buffer);

    if(force_init || _root->magic != MAGIC) {

      if(!force_init) {
        PWRN("Immutable_allocator_base: detected corruption, re-initializing");
      }

      if(size <= sizeof(struct Immutable_slab))
        throw std::bad_alloc();

      _root->magic = MAGIC;
      _root->slab_end = static_cast<byte*>(buffer) + size;
      _root->next_free = static_cast<byte*>(buffer) + sizeof(struct Immutable_slab);
      if(regions.size() > 1) {
        auto copy_regions = regions;
        copy_regions.erase(copy_regions.begin());
        /* recurse through regions */
        _root->linked_slab = new Immutable_allocator_base(copy_regions, callback, force_init);
      }
      else {
        _root->linked_slab = nullptr; /* no more linked regions */
      }
      _root->resvd = 0;

      pmem_persist(_root, sizeof(*_root));
      _rebuilt = true;
    }
    else {
      assert(is_valid());
    }

    if(_root->next_free.load() > _root->slab_end)
      throw std::logic_error("next_free exceeds slab end");

    machine_checks();

    /* check that the memory is accessible */
    //touch_pages(buffer, _root->slab_end - static_cast<byte*>(buffer));

    return _rebuilt;
  }

  void * first_element() {
    return reinterpret_cast<byte*>(_root) + sizeof(struct Immutable_slab);
  }


  status_t allocate(void* & ptr, std::size_t bytes, std::size_t alignment = 0) override
  {
    if(alignment > 0) throw API_exception("alignment not supported");
    ptr = allocate(bytes);
    return S_OK;
  }

  status_t free(void* &, std::size_t) override
  {
    PERR("free called on immutable allocator");
    asm("int3");
    return E_NOT_SUPPORTED;
  }

  status_t remaining(std::size_t& out_size) const override
  {
    out_size = std::size_t(_root->slab_end - reinterpret_cast<byte*>(_root->next_free.load()));
    return S_OK;
  }

  bool rebuilt() const { return _rebuilt; }

  /**
   * @brief      Allocate memory
   *
   * @param[in]  size  Size to allocate in bytes
   *
   * @return     Pointer to newly allocated region
   */
  void * allocate(const size_t size)
  {
    assert(_root);
    assert(is_valid());

		if(size == 0)
			throw std::invalid_argument("size parameter cannnot be 0");

    /* round up to 8 aligned */
    auto rsize = round_up(size, 8);

    byte * nptr;
    byte * updated_nptr;

    nptr = _root->next_free.load(); //std::memory_order_relaxed);
    assert(nptr <= _root->slab_end);
    updated_nptr = nptr + rsize;

    if(updated_nptr > _root->slab_end) { /* out of memory condition */
      if(_root->linked_slab) {
        return _root->linked_slab->allocate(size);
      }
      throw std::bad_alloc();
    }

    while(!_root->next_free.compare_exchange_weak(nptr,
                                                  updated_nptr,
                                                  std::memory_order_release,
                                                  std::memory_order_relaxed));

    pmem_flush(&_root->next_free, sizeof(_root->next_free));
    assert(check_aligned(nptr, 8));
    return nptr;
  }

  /**
   * @brief      Expand the slab allocator (creates and links in new allocator)
   *
   * @param[in]  region  New memory region
   */
  void expand(::iovec region) {
    if(_root->linked_slab) {
      _root->linked_slab->expand(region);
      return;
    }
    
    /* link in a new allocator */
    _root->linked_slab = new Immutable_allocator_base(region); 
    pmem_persist(_root->linked_slab, sizeof(_root->linked_slab));
  }

  inline size_t remaining_size() const { return std::size_t(_root->slab_end - _root->next_free); }

  bool is_valid() const { return _root->magic == MAGIC; }

  void dump_info() const {
    PINF("-- Immutable Allocator --");
    PINF("magic    : %X", _root->magic);
    PINF("range    : %p-%p (size %lu)", static_cast<const void *>(_root), static_cast<const void *>(_root->slab_end), _root->slab_end - reinterpret_cast<byte*>(_root));
    PINF("next free: %p", static_cast<const void *>(_root->next_free.load()));
    PINF("free     : %lu/%lu",
         _root->slab_end - reinterpret_cast<byte*>(_root->next_free.load()),
         _root->slab_end - reinterpret_cast<byte*>(_root));
    PINF("linked   : %p", reinterpret_cast<void*>(_root->linked_slab));
    PINF("-------------------------");
    if(_root->linked_slab)
      _root->linked_slab->dump_info();
  }


private:
  void machine_checks() {
    if(pmem_has_hw_drain() || pmem_has_auto_flush())
      throw std::logic_error("machine check failed");
  }
};


/* wrapper for std C++ containers */
template<class T>
class Immutable_allocator
  : private Immutable_allocator_base
//    public std::allocator_traits<T>
{
public:
#if 0
  explicit Immutable_allocator(void * base, const size_t size, const bool force_init) noexcept
    : Immutable_allocator_base(base, size, nullptr, force_init) {
    assert(base);
    assert(size);
  };
#endif
  explicit Immutable_allocator(region_vector_t& regions, const bool force_init) noexcept 
    : Immutable_allocator_base(regions, nullptr, force_init) {
  };

  T* allocate(std::size_t n=1) {
    return static_cast<T*>(Immutable_allocator_base::allocate(n * sizeof(T)));
  }
  void deallocate(T*, std::size_t) noexcept {
    PWRN("deallocate on Immutable_allocator does nothing.");
  }
  void dump_info() {
    Immutable_allocator_base::dump_info();
  }
};

// class Keyed_memory_allocator : public ccpm::Immutable_allocator_base
// {
// public:
//   static constexpr size_t base_size = 4096;

//   explicit Keyed_memory_allocator(const std::string& name,
//                                   uint64_t work_key,
//                                   component::IADO_plugin::Callback_table& cb,
//                                   bool force_init = false) :
//     ccpm::Immutable_allocator_base({init(name, work_key, force_init, cb), base_size}, nullptr, force_init),
//     _cb(cb)
//   {
//     assert(_root);
//     assert(_root_len > 0);
//     pmem_persist(this, sizeof(Keyed_memory_allocator));
//   }

//   void * init(const std::string& name,
//               const uint64_t work_key,
//               const bool force_init,
//               component::IADO_plugin::Callback_table& cb)
//   {
//     status_t rc;

//     if(force_init) {
//       cb.erase_key_func(work_key, name);
//       _root_len = base_size;
//       rc = cb.create_key_func(work_key,
//                               name,
//                               _root_len,
//                               component::IADO_plugin::FLAGS_PERMANENT_LOCK,
//                               _root);

//       if(rc != S_OK) throw Logic_exception("could not create allocator key");
//       PLOG("created new allocator (_root=%p, _root_len=%lu)", _root, _root_len);
//     }
//     else {
//       /* get existing allocator */
//       rc = cb.open_key_func(work_key,
//                             name,
//                             component::IADO_plugin::FLAGS_PERMANENT_LOCK,
//                             _root,
//                             _root_len);
//       if(rc != S_OK) throw Logic_exception("could not open allocator key");
//       if(_root_len != base_size) throw Logic_exception("reopened root incorrect size");
//       PLOG("opened existing allocator (rc=%d, _root=%p, _root_len=%lu)", rc, _root, _root_len);
//     }
//     assert(_root);
//     assert(_root_len);
//     PLOG("init returning (_root=%p, _root_len=%lu)", _root, _root_len);
//     return _root;
//   }

// private:
//   component::IADO_plugin::Callback_table& _cb;
//   void * _root;
//   size_t _root_len;
// };

class EASTL_immutable_allocator
  : public eastl::DummyTracker
{
  using Base = ccpm::Immutable_allocator_base;
public:
  using tracker_type = eastl::DummyTracker;
  explicit EASTL_immutable_allocator(const char* = NULL) : _base() { assert(0); }
  EASTL_immutable_allocator(Base * inner_allocator) : _base(inner_allocator) { assert(_base->is_valid()); };
  EASTL_immutable_allocator(const EASTL_immutable_allocator& allocator) : _base(allocator._base) { assert(_base->is_valid()); };
  EASTL_immutable_allocator(const EASTL_immutable_allocator&, const char*) : _base() { assert(0); }

  EASTL_immutable_allocator& operator=(const EASTL_immutable_allocator& src) { _base = src._base; return *this; }

  void* allocate(size_t n, int flags = 0) {
    assert(flags == 0);
    assert(_base);
    return _base->allocate(n);
  }

  void* allocate(size_t /*n*/, size_t /*alignment*/, size_t, int = 0) { assert(0); return NULL; }
  void  deallocate(void*, size_t)                 { assert(0); }

  const char* get_name() const      { return ""; }
  void        set_name(const char*) { }

private:
  Base * _base;
};

inline bool operator==(const EASTL_immutable_allocator&, const EASTL_immutable_allocator&) { return true;  }
inline bool operator!=(const EASTL_immutable_allocator&, const EASTL_immutable_allocator&) { return false; }


// class EASTL_keyed_memory_allocator
// {
// public:
//   explicit EASTL_keyed_memory_allocator(const char* = NULL) { assert(0); }
//   EASTL_keyed_memory_allocator(Keyed_memory_allocator * inner_allocator) : _ia(inner_allocator) {}
//   EASTL_keyed_memory_allocator(const EASTL_keyed_memory_allocator& allocator) {
//     _ia = allocator._ia;
//   }
//   EASTL_keyed_memory_allocator(const EASTL_keyed_memory_allocator&, const char*) { assert(0); }

//   EASTL_keyed_memory_allocator& operator=(const EASTL_keyed_memory_allocator&) { return *this; }

//   void* allocate(size_t n, int  = 0) { assert(_ia); return _ia->allocate(n); }
//   void* allocate(size_t /*n*/, size_t /*alignment*/, size_t, int = 0) { assert(0); return NULL; }
//   void  deallocate(void*, size_t)                 { assert(0); }

//   const char* get_name() const      { return ""; }
//   void        set_name(const char*) { }
//   void set_inner_allocator(Keyed_memory_allocator * inner_allocator) { _ia = inner_allocator;
//   }
// private:
//   Keyed_memory_allocator * _ia;
// };

// inline bool operator==(const EASTL_keyed_memory_allocator&, const EASTL_keyed_memory_allocator&) { return true;  }
// inline bool operator!=(const EASTL_keyed_memory_allocator&, const EASTL_keyed_memory_allocator&) { return false; }



}

#endif // __CCPM_IMMUTABLE_ALLOCATOR_H__
