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

#include <ccpm/log.h>

#include <common/pointer_cast.h>
#include <gsl/pointers>
#include <cstring>
#include <iostream>
#include <new> // bad_alloc

#define PERSIST(pf, x) ((pf)->persist(common::make_byte_span(&(x), sizeof (x))))


struct element
{
	enum class tag { DATA, ALLOC, FREE };

	tag _tag;
	void *_original_address;
	std::size_t _length;
	char *_saved_address;

	void report(const char *what) const
	{
#if 0
		void *a = _original_address;
		std::cerr << this << " " << what << " ";

		switch ( _tag )
		{
		case tag::DATA:
			std::cerr << "data";
			break;
		case tag::ALLOC:
			std::cerr << "alloc";
			break;
		case tag::FREE:
			std::cerr << "free";
			break;
		}

		std::cerr << " " << _length << " bytes at " << a;

		switch ( _tag )
		{
		case tag::DATA:
			switch ( _length )
			{
			case 4:
				std::cerr << ":" << *static_cast<int *>(a);
				break;
			case 8:
				std::cerr << ":" << *static_cast<void **>(a);
				break;
			default:
				break;
			}
			break;
		case tag::ALLOC:
			break;
		case tag::FREE:
			break;
		}
		std::cerr << "\n";
#else
		(void)what; // unused
#endif
	}

public:
	explicit element(void *original_address_, std::size_t length_, char *saved_address_)
		: _tag(tag::DATA)
		, _original_address(original_address_)
		, _length(length_)
		, _saved_address(saved_address_)
	{
		/* saved data */
		report("log");
	}
	explicit element(tag t, void *&address_, std::size_t length_, char *saved_address_)
		: _tag(t)
		, _original_address(address_)
		, _length(length_)
		, _saved_address(saved_address_)
	{
		/* saved data */
		report("log");
	}
	void commit(ccpm::IHeap_expandable *heap_)
	{
		report("commit");
		switch ( _tag )
		{
		case tag::DATA:
			break;
		case tag::ALLOC:
			break;
		case tag::FREE:
			heap_->free(_original_address, _length);
			break;
		}
	}
	void rollback(ccpm::IHeap_expandable *heap_)
	{
		report("rollback");
		switch ( _tag )
		{
		case tag::DATA:
			std::memcpy(_original_address, _saved_address, _length);
			break;
		case tag::ALLOC:
			heap_->free(_original_address, _length);
			break;
		case tag::FREE:
			break;
		}
	}
};

namespace ccpm
{
	struct block_header
	{
		using persister_type = gsl::not_null<ccpm::persister *>;
	private:
		/* A block_header is followed by "block space". A variable number of "elements"
		 * grow up from the end of the block header, and "data space" (for saved bytes
		 * to be used in a rollback) grow down from the end of the block header.
		 *  [bh|elements ... ->  | free space  |  <- ...  data spaces]
		 *  ^  ^                 ^             ^                     ^
		 *  |  |                 |             |                     data_space_end
		 *  |  |                 |             data_space_current
		 *  |  |                 element_count
		 *  |  this+1
		 *  this
		 */
		/* The previously allocate block (or nullptr_t, if none */
		block_header *_previous;
		/*
		 * Space currently used for elements
		 */
		std::size_t _element_count;
		/*
		 * The "heap" (Space allocated for block space immediately follows the block_header)
		 */

		const char *data_space_current() const { return _element_count ? (element_back()->_saved_address) : _data_space_end; }
		char *data_space_current() { return _element_count ? (element_back()->_saved_address) : _data_space_end; }
		char *_data_space_end;
		element *element_first() {
			return common::pointer_cast<element>(this+1);
		}
		element *element_last() {
			return common::pointer_cast<element>(this+1) + _element_count;
		}
		const element *element_last() const
		{
			return common::pointer_cast<const element>(this+1) + _element_count;
		}
		const element *element_back() const
		{
			return element_last() - 1;
		}
		element *element_back() {
			return element_last() - 1;
		}

	public:
		explicit block_header(block_header *ptr, std::size_t free_size)
			: _previous(std::move(ptr))
			, _element_count(0)
			, _data_space_end(common::pointer_cast<char>(this+1) + free_size)
		{
		}
		block_header(const block_header &) = delete;
		block_header &operator=(const block_header &) = delete;
		block_header *previous() const
		{
			return _previous;
		}
		std::size_t size() const
		{
			return std::size_t(_data_space_end - common::pointer_cast<const char>(this));
		}
		void rollback(persister_type persist_, IHeap_expandable *heap_)
		{
			while ( _element_count != 0 )
			{
				element_back()->rollback(heap_);
				--_element_count;
				PERSIST(persist_, *this);
			}
		}
		void commit(persister_type persist_, IHeap_expandable *heap_)
		{
			while ( _element_count != 0 )
			{
				element_back()->commit(heap_);
				--_element_count;
				PERSIST(persist_, *this);
			}
		}
		bool fits_data(std::size_t size) const
		{
			/* An element will fit if its size is 0 (save is elided) or there is
			 * room enough room to store an element and all the data.
			 */
			return size == 0 || common::pointer_cast<const char>(element_last() + 1) <= data_space_current() - size;
		}
		bool fits_alloc() const
		{
			/* An element will fit if or there is room enough room to store an element
			 */
			return common::pointer_cast<const char>(element_last() + 1) <= data_space_current();
		}

		void add(persister_type persist_, char *begin, std::size_t size)
		{
			if ( 0 != size )
			{
				/* save the data */
				auto dst = data_space_current() - size;
				std::memcpy(dst, begin, size);
				persist_->persist(common::make_byte_span(dst, size));
				/* save the element */
				new (element_last()) element(begin, size, dst);
				PERSIST(persist_, *element_last());
				/* modifies _data_space_current() and element_last(), in one operation */
				++_element_count;
				PERSIST(persist_, *this);
			}
			else
			{
				/* 0-length save, ignore it. */
			}
		}

		void allocated(persister_type persist_, void *&p, std::size_t size)
		{
			if ( 0 != size )
			{
				/* save the element */
				new (element_last()) element(element::tag::ALLOC, p, size, data_space_current());
				PERSIST(persist_, *element_last());
				/* modifies _data_space_current() and element_last(), in one operation */
				++_element_count;
				PERSIST(persist_, *this);
			}
			else
			{
				/* 0-length allocataion, ignore it. */
			}
		}

		void freed(persister_type persist_, void *&p, std::size_t size)
		{
			if ( 0 != size )
			{
				/* save the element */
				new (element_last()) element(element::tag::FREE, p, size, data_space_current());
				PERSIST(persist_, *element_last());
				/* modifies _data_space_current() and element_last(), in one operation */
				++_element_count;
				PERSIST(persist_, *this);
			}
			else
			{
				/* 0-length free, ignore it. */
			}
		}
	};

	void log::clear_top()
	{
		void *r = _root;
		auto s = _root->size();
		_root = _root->previous();
		_mr->free(r, s);
	}

	log::log(persister_type persister_, heap_type mr_)
		: _persister(persister_)
		, _mr(mr_)
		, _root(nullptr)
	{
	}

	log::~log()
	{
		commit();
	}

	constexpr std::size_t log::min_log_extend; //  = std::max(sizeof(block_header), std::size_t(65536U));

	void log::extend(std::size_t size)
	{
		const auto min_log_extend_safe = std::max(sizeof(block_header), std::size_t(65536U));
		void *p = nullptr;
		auto block_size = std::max(sizeof(element) + size, min_log_extend_safe - sizeof(block_header)); /* the bare minumum: one element + space for the data */

		_mr->allocate(p, sizeof(block_header) + block_size, sizeof(void *));
		if ( ! p )
		{
			throw bad_alloc_log();
		}
		_root = new (p) block_header(_root, block_size);
	}

	/*
	 * Add a region to the log.
	 */
	void log::add(void *begin, std::size_t size)
	{
		if ( ! _root || ! _root->fits_data(size) )
		{
			extend(size);
		}

		_root->add(_persister, static_cast<char *>(begin), size);
	}

	/*
	 * Add an allocation to the log.
	 */
	void log::allocated(void *&pl, std::size_t size)
	{
		if ( ! _root || ! _root->fits_alloc() )
		{
			extend(size);
		}

		_root->allocated(_persister, pl, size);
	}

	/*
	 * Add a deallocation to the log.
	 */
	void log::freed(void *&pl, std::size_t size)
	{
		if ( ! _root || ! _root->fits_alloc() )
		{
			extend(size);
		}

		_root->freed(_persister, pl, size);
		pl = nullptr;
	}
	/*
	 * hide all previoud add commands
	 */
	void log::commit()
	{
		while ( _root )
		{
			_root->commit(_persister, _mr);
			clear_top();
		}
	}
	/*
	 * Restore all data areas added after initialization (or the most recent clear)
	 * to the values present at their last add command.
	 */
	void log::rollback()
	{
		while ( _root )
		{
			_root->rollback(_persister, _mr);
			clear_top();
		}
	}
}
