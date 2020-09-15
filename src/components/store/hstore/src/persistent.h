/*
   Copyright [2017-2019] [IBM Corporation]
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


#ifndef MCAS_HSTORE_PERISTENT_H
#define MCAS_HSTORE_PERISTENT_H

#include "perishable.h"
#include "test_flags.h" /* TEST_HSTORE_PERISHABLE */
#include <atomic>

template <typename T>
	struct persistent
	{
	private:
		T _v;
	public:
		using value_type = T;
		persistent()
			: _v((perishable::tick(), T()))
		{
		}
		persistent(const persistent &other)
			: _v(perishable::tick() ? other._v : T())
		{
		}

		persistent(const T &t)
			: _v(perishable::tick() ? t : T())
		{
		}
		persistent<T> &operator=(const persistent &other)
		{
			if ( perishable::tick() )
			{
				_v = other._v;
			}
			return *this;
		};
		persistent<T> &operator=(const T &t)
		{
			if ( perishable::tick() )
			{
				_v = t;
			}
			return *this;
		};
		persistent<T> &operator&=(const T &t)
		{
			if ( perishable::tick() )
			{
				_v &= t;
			}
			return *this;
		};
		persistent<T> &operator|=(const T &t)
		{
			if ( perishable::tick() )
			{
				_v |= t;
			}
			return *this;
		};
		operator T() const
		{
			return _v;
		}
		T load() const
		{
			return _v;
		}
		persistent<T> &operator++()
		{
			if ( perishable::tick() )
			{
				++_v;
			}
			return *this;
		}
		persistent<T> &operator--()
		{
			if ( perishable::tick() )
			{
				--_v;
			}
			return *this;
		}
		auto &operator*() const
		{
			return *_v;
		}
		T operator->() const
		{
			return _v;
		}

		const T &ref() const
		{
			return _v;
		}
	};

template <typename T>
	struct persistent_traits
	{
		using value_type = T;
	};

template <typename T>
	struct persistent_traits<persistent<T>>
	{
		using value_type = typename persistent<T>::value_type;
	};

template <typename T>
	struct persistent_atomic
	{
	private:
		std::atomic<T> _v;
		static_assert(sizeof _v <= 8, "persistent store is too large");
	public:
		persistent_atomic()
			: _v((perishable::tick(), T{}))
		{
		}
		persistent_atomic(const persistent_atomic &other)
			: _v(perishable::tick() ? other._v.load() : T())
		{
		}
		persistent_atomic(const T &t)
			: _v(perishable::tick() ? t : T())
		{
		}
		persistent_atomic<T> &operator=(const persistent_atomic &other)
		{
			if ( perishable::tick() )
			{
				_v = other._v.load();
			}
			return *this;
		};
		persistent_atomic<T> &operator=(const T &t)
		{
			if ( perishable::tick() )
			{
				_v = t;
			}
			return *this;
		};
		persistent_atomic<T> &operator+=(const T &t)
		{
			if ( perishable::tick() )
			{
				_v += t;
			}
			return *this;
		};
		persistent_atomic<T> &operator-=(const T &t)
		{
			if ( perishable::tick() )
			{
				_v -= t;
			}
			return *this;
		};
		persistent_atomic<T> &operator&=(const T &t)
		{
			if ( perishable::tick() )
			{
				_v &= t;
			}
			return *this;
		};
		persistent_atomic<T> &operator|=(const T &t)
		{
			if ( perishable::tick() )
			{
				_v |= t;
			}
			return *this;
		};
		operator T() const
		{
			return _v;
		}
		persistent_atomic<T> &operator++()
		{
			if ( perishable::tick() )
			{
				++_v;
			}
			return *this;
		};
		persistent_atomic<T> &operator--()
		{
			if ( perishable::tick() )
			{
				--_v;
			}
			return *this;
		};
		auto &operator*() const
		{
			return *_v;
		};
		T operator->() const
		{
			return _v;
		};
	};


#if TEST_HSTORE_PERISHABLE
static constexpr bool perishable_testing = true;
template <typename T>
	using persistent_t = persistent<T>;
template <typename T>
	using persistent_atomic_t = persistent_atomic<T>;
/* Return a reference to the true object underlying a persistent<T>.
 * Dangerous in that the caller can use the reference in such a
 * way as to defeat the persistence testing. Make it a bit less dangerous
 * by returning a *const* ref.
 */
template <typename T>
	const T & persistent_ref(const persistent<T> &p_)
	{
		return  p_.ref();
	}
template <typename T>
	T persistent_load(const persistent<T> &p_)
	{
		return  p_.load();
	}
#else
static constexpr bool perishable_testing = false;
template <typename T>
	using persistent_t = T;
template <typename T>
	using persistent_atomic_t = T;
template <typename T>
	const T & persistent_ref(T &p_)
	{
		return p_;
	}
template <typename T>
	T persistent_load(T &p_)
	{
		return p_;
	}
#endif

#endif
