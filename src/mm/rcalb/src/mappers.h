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


/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __MAPPERS_H__
#define __MAPPERS_H__

#include <common/types.h> /* addr_t */
#include <common/utils.h> /* KiB, MiB, MB, GB, round_down */
#include <boost/icl/split_interval_map.hpp>
#include <cstddef> /* size_t */

inline static unsigned get_log2_bin(size_t a)
{
  unsigned fsmsb = unsigned(((sizeof(size_t) * 8) - unsigned(__builtin_clzl(a))));
  if ((addr_t(1) << (fsmsb - 1)) == a) fsmsb--;
  return fsmsb;
}

/* Two types of regions:
 *   large-object regions contain a single object
 *   small-object regions contain multiple objects
 */
class Large_and_small_bucket_mapper {
  friend class Region_map;

protected:
  /* Objects not exceeding 8K size are allocated in 1MB pools.
   * Objects exceding 8K are allocated singly.
   */
  static constexpr size_t L0_MAX_SMALL_OBJECT_SIZE = KiB(256);
  static constexpr size_t L0_REGION_SIZE           = MiB(1);

 public:
  Large_and_small_bucket_mapper()
      : _large_object_bucket_index(get_log2_bin(L0_MAX_SMALL_OBJECT_SIZE) + 1)
  {
  }

  unsigned bucket(size_t object_size)
  {
    if (object_size <= L0_MAX_SMALL_OBJECT_SIZE)
      return get_log2_bin(object_size);

    /* All "large" objects are allocated in a single bucket, regardless of size.
     * Therefore, size maps to bucket index, but bucket index does not map to a size.
     */
    return boost::numeric_cast<unsigned>(_large_object_bucket_index);
  }

  void *base(void *addr, size_t object_size)
  {
    if (object_size <= L0_MAX_SMALL_OBJECT_SIZE)
      return round_down(addr, L0_REGION_SIZE);

    return addr;
  }

  size_t region_size(size_t object_size)
  {
    if ( object_size <= L0_MAX_SMALL_OBJECT_SIZE )
      return L0_REGION_SIZE;
    else
      return object_size;
  }

  size_t rounded_up_object_size(size_t object_size)
  {
    if (object_size <= L0_MAX_SMALL_OBJECT_SIZE)
      return (1UL << get_log2_bin(object_size));
    else
      return object_size;
  }

  bool could_exist_in_region(size_t object_size)
  {
    return object_size <= L0_MAX_SMALL_OBJECT_SIZE;
  }

 private:
  const size_t _large_object_bucket_index; /* index for bucket of large objects */
};

/** 
 * Basic log2 mapper; mainly for testing
 * 
 */
class Log2_bucket_mapper {
  static constexpr size_t REGION_SIZE = MB(1);

 public:
  unsigned bucket(size_t object_size) { return get_log2_bin(object_size); }

  void *base(
    void *addr
    , size_t // size unused

  )
/* Magic value GB(1). Why? */
 { return round_down(addr, GB(1)); }

  size_t region_size(
    size_t // object_size unused
  ) { return REGION_SIZE; }

  size_t rounded_up_object_size(size_t size)
  {
    size_t rup = 1UL << get_log2_bin(size);
    assert(rup >= size);
    return rup;
  }

  bool could_exist_in_region(
    size_t // object_size unused
  ) { return true; }
};

//using Bucket_mapper = Log2_bucket_mapper;
using Bucket_mapper = Large_and_small_bucket_mapper;


#endif
