/*
   Copyright [2019,2021] [IBM Corporation]
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

#ifndef __CCPM_CRASH_CONSISTENT_ALLOCATOR_H__
#define __CCPM_CRASH_CONSISTENT_ALLOCATOR_H__

#include <ccpm/interfaces.h>
#include <common/exceptions.h>
#include <gsl/pointers>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace ccpm
{
	struct area_top;

	struct cca
		: public IHeap_expandable
	{
		using byte_span = common::byte_span;
		using persist_type = gsl::not_null<persister *>;
	private:
		using top_vec_t = std::vector<std::unique_ptr<area_top>>;
		top_vec_t _top;
		top_vec_t::difference_type _last_top_allocate;
		top_vec_t::difference_type _last_top_free;
		persist_type _persist;

		void init(
			region_span regions
			, ownership_callback_t resolver
			, bool force_init
		);
	public:
		explicit cca(persist_type persist, region_span regions, ownership_callback_t resolver);

		explicit cca(persist_type persist, region_span regions);

		explicit cca(persist_type persist);

		~cca();

		cca(const cca &) = delete;
		const cca& operator=(const cca &) = delete;

		bool reconstitute(
			region_span regions
			, ownership_callback_t resolver
			, bool force_init
		) override;

		status_t allocate(
			void * & ptr_
			, std::size_t bytes_
			, std::size_t alignment_
		) override;

    void * allocate(std::size_t bytes_, std::size_t alignment_ = 1) {
      void * ptr = nullptr;
      if(allocate(ptr, bytes_, alignment_) != 0)
        throw General_exception("ccpm::cca::allocate failed");
      return ptr;
    }

    void * allocate_root(std::size_t bytes_, std::size_t alignment_ = 8) {
      void * ptr = nullptr;
      if(allocate(ptr, bytes_, alignment_) != 0)
        throw General_exception("ccpm::cca::allocate failed");
      set_root(common::make_byte_span(ptr, bytes_));
      return ptr;
    }

		status_t free(
			void * & ptr_
			, std::size_t bytes_
		) override;

		void add_regions(
			region_span
		) override;

		bool includes(
			const void *addr
		) const override;

		status_t remaining(
			std::size_t & out_size_
		) const override;

		region_vector_t get_regions() const override;

    void set_root(
      byte_span
    );

    byte_span get_root() const;

		void print(std::ostream &, const std::string &title = "cca") const;
	};
}

#endif
