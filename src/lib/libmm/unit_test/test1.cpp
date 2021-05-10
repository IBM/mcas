/*
  Copyright [2021] [IBM Corporation]
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

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <malloc.h>
#include <common/logging.h>
#include <common/errors.h>
#include <common/utils.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

class Libmm_test : public ::testing::Test {
protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

#include "mm_plugin_itf.h"

TEST_F(Libmm_test, MM_plugin_wrapper)
{
  std::string plugin_path = ::getenv("PLUGIN");
  PLOG("loading plugin: %s", plugin_path.c_str());
  MM_plugin_wrapper mm(plugin_path);

  size_t slab_size = MiB(32);
  auto slab_memory = mmap(reinterpret_cast<void*>(0xAA00000000),
                          slab_size,
                          PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                          -1, 0);

  ASSERT_TRUE(mm.add_managed_region(slab_memory, slab_size) == S_OK);
  
  void * p = nullptr;
  ASSERT_TRUE(mm.allocate(1022, &p) == S_OK);
  ASSERT_TRUE(mm.deallocate(p, 1022) == S_OK);  
  ASSERT_TRUE(munmap(slab_memory, slab_size) == 0);
}


TEST_F(Libmm_test, MM_plugin_cxx_allocator)
{
  std::string plugin_path = ::getenv("PLUGIN");
  PLOG("loading plugin: %s", plugin_path.c_str());
  MM_plugin_wrapper mm(plugin_path);

  size_t slab_size = MiB(32);
  auto slab_memory = mmap(reinterpret_cast<void*>(0xAA00000000),
                          slab_size,
                          PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                          -1, 0);

  ASSERT_TRUE(mm.add_managed_region(slab_memory, slab_size) == S_OK);

  MM_plugin_cxx_allocator<uint64_t> cxxalloc(mm);
  std::vector<uint64_t,MM_plugin_cxx_allocator<uint64_t>> v(cxxalloc);

  PLOG("adding to vector...");
  for(unsigned i=0;i<1024;i++)
    v.push_back(i);
  PLOG("done adding to vector.");
}


/** 
 * Use this with testing ADO
 * 
 */
int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  auto r = RUN_ALL_TESTS();
  return r;
}
