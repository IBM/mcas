/*
   Copyright [2019] [IBM Corporation]
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

#ifndef _REGION_MOFIFICATIONS_
#define _REGION_MOFIFICATIONS_

#if 0 /* Collision: EASTL and common/types both define uint128_t */
#include <common/types.h> /* offset_t */
#else
#include <cstddef>
using offset_t = std::size_t;
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

#include <boost/icl/interval_map.hpp>
#include <boost/icl/right_open_interval.hpp>
#include <boost/iterator/transform_iterator.hpp>

#pragma GCC diagnostic pop

namespace nupm
{

  extern thread_local bool tracker_active;

  class Region_modifications
    : private
        boost::icl::interval_map<
          const void *
	  , char
	  , boost::icl::partial_absorber
	  , ICL_COMPARE_INSTANCE(ICL_COMPARE_DEFAULT, const void *)
          , ICL_COMBINE_INSTANCE(boost::icl::inplace_identity, char)
	  , ICL_SECTION_INSTANCE(boost::icl::inter_section, char)
	  , boost::icl::right_open_interval<const void *>
	>
  {
    using base =
      boost::icl::interval_map<
        const void *
        , char
        , boost::icl::partial_absorber
        , ICL_COMPARE_INSTANCE(ICL_COMPARE_DEFAULT, const void *)
        , ICL_COMBINE_INSTANCE(boost::icl::inplace_identity, char)
        , ICL_SECTION_INSTANCE(boost::icl::inter_section, char)
        , boost::icl::right_open_interval<const void *>
      >
      ;

  public:
    void add(const void *p, std::size_t s, char tag = 'w') // w => raw
    {
      auto pc = static_cast<const char *>(p);
      insert(base::value_type(boost::icl::right_open_interval<const void *>(pc, pc + s), tag));
    }

    template <typename T>
      void add(const T &v)
      {
        this->add(&v, sizeof v);
      }

    /* get_region - iterate space of ALL modified regions collected - return region size or zero for end */

    auto begin()
    {
        auto choose_first = std::bind(&base::value_type::first, std::placeholders::_1);
	return boost::make_transform_iterator(base::begin(), choose_first);
    }
    auto end()
    {
        auto choose_first = std::bind(&base::value_type::first, std::placeholders::_1);
	return boost::make_transform_iterator(base::end(), choose_first);
    }

    /* clear region tracking */

    using base::clear;
    Region_modifications()
    {
      tracker_active = true;
    }
    ~Region_modifications()
    {
      tracker_active = false;
    }
  };

  extern thread_local Region_modifications tls_modifications;

  /* add region */
  void region_tracker_add(const void * p, size_t p_len, char tag = 'w');
  void region_tracker_coalesce_across_TLS();

  /* offset_t is uint64_t. Presume that it is meant to be an index into the
   * interval set.
   * If examining all intervals, better to use
   *   for ( auto e : tls_modifications ) ...
   */ 
  std::size_t region_tracker_get_region(offset_t offset, const void*& p);
  void region_tracker_clear();
}

#endif
