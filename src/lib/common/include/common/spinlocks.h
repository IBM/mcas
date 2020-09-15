/*
   eXokernel Development Kit (XDK)

   Samsung Research America Copyright (C) 2013

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.

   As a special exception, if you link the code in this file with
   files compiled with a GNU compiler to produce an executable,
   that does not cause the resulting executable to be covered by
   the GNU Lesser General Public License.  This exception does not
   however invalidate any other reasons why the executable file
   might be covered by the GNU Lesser General Public License.
   This exception applies to code released by its copyright holders
   in files containing the exception.
*/

/*
  Authors:
  Copyright (C) 2013, Daniel G. Waddington <d.waddington@samsung.com>
*/

#ifndef __COMMON_SPIN_LOCKS_H__
#define __COMMON_SPIN_LOCKS_H__

#include <common/common.h>
#include <common/cpu.h>
#include <common/logging.h>
#include <common/types.h>
#include <common/utils.h>

#include "atomic.h"
#include "errors.h"

#include <pthread.h>

#define INLINE inline __attribute__((always_inline))

/* Compile read-write barrier */
#define xdk_barrier() asm volatile("" : : : "memory")

#define cmpxchg(P, O, N) __sync_val_compare_and_swap((P), (O), (N))
#define atomic_xadd(P, V) __sync_fetch_and_add((P), (V))
#define atomic_inc(P) __sync_add_and_fetch((P), 1)
#define atomic_dec(P) __sync_add_and_fetch((P), -1)
#define atomic_add(P, V) __sync_add_and_fetch((P), (V))
#define atomic_set_bit(P, V) __sync_or_and_fetch((P), 1 << (V))
#define atomic_clear_bit(P, V) __sync_and_and_fetch((P), ~(1 << (V)))

static inline void *xchg(void *ptr, void *x) {
#ifdef __amd64__
  __asm__ __volatile__("xchgq %0,%1"
                       : "=r"(x)
                       : "m"(*static_cast<volatile long long *>(ptr)),
                         "0"(reinterpret_cast<unsigned long long>(x))
                       : "memory");
#elif defined(__i386__)
  return (void *) __atomic_exchange_n((unsigned long *) ptr,
                                      (unsigned long *) x, __ATOMIC_RELAXED);
#elif defined(__arm__)
  return (void *) __atomic_exchange_n((unsigned long *) ptr,
                                      (unsigned long *) x, __ATOMIC_RELAXED);
#endif

  return x;
}

namespace common
{
/**
 * Ticket lock spin-lock.  This type of lock is used in the Linux kernel.  It
 * is
 * fast and fair, but may not scale to large number of cores.  The performance
 * of
 * this lock can collapse (see Non-scalable Lock ar Dangerous, Boyd-Wickizer
 * et al.)
 *
 */
class Ticket_lock {
 private:
  struct tl_s
  {
      unsigned short ticket; /* little endian order */
      unsigned short users;
  } __attribute__((packed));
  union {
    unsigned u;
    tl_s s;
  };

 public:
  Ticket_lock() : u(0) {}

  INLINE void lock() {
#pragma GCC diagnostic push
#if ( defined __clang_major__ && 4 <= __clang_major__ ) || 9 <= __GNUC__
#pragma GCC diagnostic ignored "-Waddress-of-packed-member"
#endif
    unsigned short me = atomic_xadd(&s.users, 1);
#pragma GCC diagnostic pop
    while (s.ticket != me) cpu_relax();
  }

  INLINE void unlock() {
    xdk_barrier();
    s.ticket++;
  }

  int trylock() {
    auto me = s.users;
    auto menew = me + 1;
    unsigned cmp = (static_cast<unsigned>(me) << 16) + me;
    unsigned cmpnew = (static_cast<unsigned>(menew) << 16) + me;

#pragma GCC diagnostic push
#if ( defined __clang_major__ && 4 <= __clang_major__ ) || 9 <= __GNUC__
#pragma GCC diagnostic ignored "-Waddress-of-packed-member"
#endif
    if (cmpxchg(&u, cmp, cmpnew) == cmp) return 0;
#pragma GCC diagnostic pop

    return E_BUSY;
  }

  int lockable() {
    xdk_barrier();
    return (s.ticket == s.users);
  }
} __attribute__((packed));

/**
 * Basic spinlock.  This lock is not fair or truly scalable.
 *
 *
 * @return
 */
class Spin_lock {
 private:
  volatile atomic_t _l alignas(sizeof(atomic_t));

  enum {
    UNLOCKED = 0,
    LOCKED = 1,
  };

 public:
  Spin_lock() : _l(UNLOCKED)
  {
  }

  /**
   * Take lock
   *
   */
  INLINE void lock() {
    while (!__sync_bool_compare_and_swap(&_l, UNLOCKED, LOCKED)) {
      while (_l) cpu_relax(); /* unsafe spin to help reduce coherency traffic */
    }
  }

  // void sleep_lock () {
  //   while (!__sync_bool_compare_and_swap(&_l, UNLOCKED, LOCKED)) {
  //     cpu_relax();
  //   }
  // }

  INLINE void unlock() { _l = UNLOCKED; }

  /**
   * Try to take lock.  Do not block.
   *
   *
   * @return
   */
  INLINE bool try_lock() {
    return __sync_bool_compare_and_swap(&_l, UNLOCKED, LOCKED);
  }

} __attribute__((packed));


/**
 * Basic spinlock.  This lock is not fair or truly scalable.
 *
 *
 * @return
 */
class Spin_lock_padded {
 private:
  /* pad to x2 cache line size */
  byte _padding0[2 * CACHE_LINE_SIZE];
  volatile atomic_t _l alignas(sizeof(atomic_t));
  byte _padding1[2 * CACHE_LINE_SIZE - sizeof(atomic_t)];

  enum {
    UNLOCKED = 0,
    LOCKED = 1,
  };

 public:
  Spin_lock_padded() : _l(UNLOCKED)
  {
    (void)_padding0; // unused
    (void)_padding1; // unused
  }

  /**
   * Take lock
   *
   */
  INLINE void lock() {
    while (!__sync_bool_compare_and_swap(&_l, UNLOCKED, LOCKED)) {
      while (_l) cpu_relax(); /* unsafe spin to help reduce coherency traffic */
    }
  }

  // void sleep_lock () {
  //   while (!__sync_bool_compare_and_swap(&_l, UNLOCKED, LOCKED)) {
  //     cpu_relax();
  //   }
  // }

  INLINE void unlock() { _l = UNLOCKED; }

  /**
   * Try to take lock.  Do not block.
   *
   *
   * @return
   */
  INLINE bool try_lock() {
    return __sync_bool_compare_and_swap(&_l, UNLOCKED, LOCKED);
  }

} __attribute__((packed));

/**
 * Reentrant locks can be locked multiple times by the same thread.  This
 * is useful when using lock guards on nest/recursive code.
 *
 */
template <class T>
class Reentrant_lock_tmpl : public T {
 private:
  /* owner field can only be written by the lock holder,
       but can be ready by many
    */
  pthread_t _owner alignas(sizeof(pthread_t));
  volatile unsigned _ref_count;

 public:
  Reentrant_lock_tmpl() : _ref_count(0) {}

  INLINE void lock() {
    if (_owner ==
        pthread_self()) { /* made atomic read because we aligned above */
      _ref_count++;
      return;
    }
    this->T::lock();
    _owner = pthread_self();
    assert(_ref_count == 0);
  }

  INLINE status_t unlock() {
    /* call on unlock without owner owning the lock! */
    if (_owner != pthread_self()) return E_INVAL;

    if (_ref_count > 0)
      _ref_count--;
    else
      this->T::unlock();

    return S_OK;
  }
};

#undef INLINE

/**
 * Lock guard class for auto lock management
 *
 * @param lock Lock instance
 *
 */
template <class T>
class Lock_guard_tmpl : public T {
 private:
  T &_lock;

 public:
  Lock_guard_tmpl(T &lock) : _lock(lock) { _lock.lock(); }
  ~Lock_guard_tmpl() { _lock.unlock(); }
};

/*
 * Typedef helpers and defaults
 */
typedef Reentrant_lock_tmpl<Spin_lock> Reentrant_spin_lock;
typedef Lock_guard_tmpl<Reentrant_spin_lock> Reentrant_lock_guard;
typedef Lock_guard_tmpl<Spin_lock> Spin_lock_guard;
typedef Lock_guard_tmpl<Ticket_lock> Ticket_lock_guard;
}  // namespace common

#endif  // __COMMON_SPIN_LOCKS_H__
