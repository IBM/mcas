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

#ifndef NUPM_ALLOCATOR_RA_H
#define NUPM_ALLOCATOR_RA_H

#include "deallocator_ra.h"

#include "bad_alloc.h"
#include "mr_traits.h"

#include <cstddef> /* size_t, ptrdiff_t */

namespace nupm
{
template <typename T, typename MR>
class allocator_adaptor;

template <typename MR>
class allocator_adaptor<void, MR>
	: public deallocator_adaptor<void, MR>
{
public:
	using deallocator_type = deallocator_adaptor<void, MR>;
	using typename deallocator_type::const_pointer;
	using typename deallocator_type::pointer;
	using typename deallocator_type::value_type;

	template <typename U>
	struct rebind
	{
		using other = allocator_adaptor<U, MR>;
	};
};

template <typename T, typename MR>
class allocator_adaptor
	: public deallocator_adaptor<T, MR>
{
public:
	using deallocator_type = deallocator_adaptor<T, MR>;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
	using typename deallocator_type::const_pointer;
	using typename deallocator_type::pointer;
	using reference = T &;
	using const_reference = const T &;
	using typename deallocator_type::value_type; // = T;

	template <typename U>
	struct rebind
	{
		using other = allocator_adaptor<U, MR>;
	};

	allocator_adaptor(MR &pmr_) noexcept
		: deallocator_type(pmr_)
	{
	}

	allocator_adaptor(const allocator_adaptor &a_) noexcept = default;

	template <typename U>
	allocator_adaptor(const allocator_adaptor<U, MR> &a_) noexcept
		: allocator_adaptor(*a_.pmr())
	{
	}

	allocator_adaptor &operator=(const allocator_adaptor &a_) = delete;

#if 0
			/* deprecated in C++20 */
			pointer address(reference x) const noexcept
			{
				return pointer(&x);
			}
			const_pointer address(const_reference x) const noexcept
			{
				return pointer(&x);
			}
#endif
	auto allocate(size_type s,
                typename allocator_adaptor<void, MR>::const_pointer /* hint */ = typename allocator_adaptor<void, MR>::const_pointer{},
                const char * = nullptr) -> pointer
	{
		unsigned numa_node = 0;
		auto ptr = mr_traits<MR>::allocate(this->pmr(), numa_node, s * sizeof(T), alignof(T));
		if (ptr == 0)
			throw bad_alloc(0, s, sizeof(T));

		return reinterpret_cast<pointer>(ptr);
	}

	/* Rca_LB expectation (non-standard) */
	auto allocate(size_type s, size_type alignment) -> pointer
	{
		int numa_node = 0;
		auto ptr = this->pmr()->alloc(s * sizeof(T), numa_node, alignment);
		if (ptr == 0)
			throw bad_alloc(0, s, sizeof(T));

		return reinterpret_cast<pointer>(ptr);
	}

	/* EASTL expectation (non-standard) */
	auto allocate(size_type s, size_type alignment, size_type offset) -> pointer
	{
		assert(offset == 0);
		auto ptr = this->pmr()->alloc(s * sizeof(T), alignment);
		if (ptr == 0)
			throw bad_alloc(0, s, sizeof(T));

		return reinterpret_cast<pointer>(ptr);
	}
};

template <typename T>
using allocator_ra = allocator_adaptor<T, core::AVL_range_allocator>;
} // namespace nupm
#endif
