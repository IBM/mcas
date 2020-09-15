/*
  Copyright [2017-2020] [IBM Corporation]
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

#include <api/interfaces.h>
#include <ccpm/immutable_allocator.h>
#include <common/cycles.h>
#include <common/logging.h>
#include <common/utils.h>
#include <libpmem.h>
#include <boost/numeric/conversion/cast.hpp>
#include <cstring>
#include "testing.h"

status_t ADO_testing_plugin::register_mapped_memory(void *shard_vaddr, void *local_vaddr, size_t len)
{
  PLOG("ADO_testing_plugin: register_mapped_memory (%p, %p, %lu)", shard_vaddr, local_vaddr, len);
  /* we would need a mapping if we are not using the same virtual
     addresses as the Shard process */
  return S_OK;
}

#define ASSERT_OK(X, Y) \
  if (X != S_OK) {      \
    PERR(Y);            \
    return E_FAIL;      \
  }

#define ASSERT_TRUE(X, Y) \
  if (!(X)) {             \
    PERR(Y);              \
    return E_FAIL;        \
  }

#define ASSERT_FALSE(X, Y) \
  if (X) {                 \
    PERR(Y);               \
    return E_FAIL;         \
  }

void ADO_testing_plugin::launch_event(const uint64_t     auth_id,
                                      const std::string &pool_name,
                                      const size_t       pool_size,
                                      const unsigned int pool_flags,
                                      const unsigned int memory_type,
                                      const size_t       expected_obj_count,
                                      const std::vector<std::string>& params)
{
  PMAJOR("ADO_testing_plugin: launch event (auth_id=%lu, %s, %lu, %u, %u, %lu)",
         auth_id, pool_name.c_str(), pool_size,
         pool_flags, memory_type, expected_obj_count);

  for(auto& p : params)
    PMAJOR("param: %s", p.c_str());
}

void ADO_testing_plugin::notify_op_event(component::ADO_op op)
{
  PMAJOR("ADO_testing_plugin: op event (%s)", to_str(op).c_str());
}

status_t ADO_testing_plugin::do_work(uint64_t                     work_key,
                                     const char *                 key_addr,
                                     size_t                       key_len,
                                     IADO_plugin::value_space_t & values,
                                     const void *                 in_work_request, /* don't use iovec because of non-const */
                                     const size_t                 in_work_request_len,
                                     bool                         new_root,
                                     response_buffer_vector_t &   response_buffers)
{
  (void)new_root; // unused
  auto value = values[0].ptr;
  auto value_len = values[0].len;
  auto detached_value = values[1].ptr;
  auto detached_value_len = values[1].len;

  status_t rc = S_OK;

  ASSERT_TRUE(value != nullptr, "ADO_testing_plugin: bad do_work parameter");
  ASSERT_TRUE(value_len != 0, "ADO_testing_plugin: value_len is 0");
  ASSERT_TRUE(key_addr != nullptr, "ADO_testing_plugin: bad do_work parameter");
  ASSERT_TRUE(key_len != 0, "ADO_testing_plugin: bad do_work parameter");

  std::string k(key_addr, key_len);

  if (k == "BasicInvokeAdo") {
    PLOG("ADO_testing_plugin: running BasicInvokeAdo (key=%s)", k.c_str());
    std::string work(static_cast<const char *>(in_work_request), in_work_request_len);
    PLOG("ADO_testing_plugin: work=(%s)", work.c_str());
    ASSERT_TRUE(work == "RUN!TEST-BasicInvokeAdo", "ADO_testing_plugin: unexpected message");

    memset(value, 0xe, value_len);
    PLOG("value=%p value_len=%lu", value, value_len);
    ASSERT_TRUE(value_len == 4096, "ADO_testing_plugin: value bad length");

    /* resize value */
    void *new_addr = nullptr;
    rc             = cb_resize_value(work_key, std::string(key_addr, key_len), KB(8), new_addr);
    ASSERT_TRUE(rc == S_OK, "ADO_testing_plugin: resize_value callback failed");
    PLOG("value after resize=%p", new_addr);

    /* check old content is still there */
    void *tmp = malloc(value_len);
    if ( tmp == nullptr )
    {
      throw std::bad_alloc();
    }
    memset(tmp, 0xe, value_len);

    PLOG("ADO_testing_plugin: content after resize OK.");
    memset(new_addr, 0xf, KB(8));

    /* create new key-value pair */
    void *                     new_value_addr = nullptr;
    const char *               new_key_addr   = nullptr;
    component::IKVStore::key_t key_handle;
    rc = cb_create_key(work_key, "newKey", 12, true, new_value_addr, &new_key_addr, &key_handle);
    memset(new_value_addr, 'N', 12);
    PLOG("new key created at %p", new_value_addr);
    ASSERT_TRUE(new_value_addr && rc == S_OK, "ADO_testing_plugin: create_key callback failed");
    ASSERT_TRUE(new_key_addr != nullptr, "ADO_testing_plugin: key ptr is null");
    ASSERT_TRUE(strncmp(new_key_addr, "newKey", 6) == 0, "ADO_testing_plugin: key ptr invalid");
    PLOG("key ptr verified OK!");

    /* allocate value memory */
    PLOG("calling cb_allocate_pool_memory");
    void * new_value_memory = nullptr;
    size_t s                = 1200;
    rc                      = cb_allocate_pool_memory(s, 8 /* alignment */, new_value_memory);
    ASSERT_TRUE(new_value_memory != nullptr, "cb_allocate_pool_memory return nullptr");
    PLOG("new value: %p rc=%d", new_value_memory, rc);
    rc = cb_free_pool_memory(s, new_value_memory);
    ASSERT_TRUE(rc == S_OK, "free_pool_memory callback failed");
    PLOG("free OK");

    /* get pool info */
    std::string info;
    rc = cb_get_pool_info(info);
    ASSERT_TRUE(rc == S_OK, "get_pool_info callback failed");
    PINF("pool info:(%s)", info.c_str());
  }
  else if (k == "AdoKeyReference") {
    void *                     new_value_addr = nullptr;
    const char *               new_key_addr   = nullptr;
    component::IKVStore::key_t key_handle, key_handle2;
    ASSERT_OK(
        cb_create_key(work_key, "akrKey", 256, FLAGS_NO_IMPLICIT_UNLOCK, new_value_addr, &new_key_addr, &key_handle),
        "ADO_testing_plugin: create key failed");

    ASSERT_TRUE(new_value_addr, "ADO_testing_plugin: bad result from create key");
    ASSERT_TRUE(new_key_addr, "ADO_testing_plugin: bad result from create key");

    memset(new_value_addr, 'N', 256);
    ASSERT_TRUE(cb_create_key(work_key, "akrKey", 256, true, new_value_addr, &new_key_addr, &key_handle2) == E_LOCKED,
                "ADO_testing_plugin: create key should be locked already");

    ASSERT_TRUE(key_handle != nullptr, "ADO_testing_plugin: invalid key_handle");
    ASSERT_OK(cb_unlock(work_key, key_handle), "ADO_testing_plugin: unlock callback failed");
    rc = S_OK;
  }
  else if (k == "BasicInvokePutAdo") {
    std::string val(reinterpret_cast<char *>(value), value_len);
    PLOG("ADO_testing_plugin: (key=%.*s,value=%s)", int(key_len), key_addr, val.c_str());
    std::string request(reinterpret_cast<const char *>(in_work_request), in_work_request_len);
    PLOG("ADO_testing_plugin: (request=%s)", request.c_str());

    ASSERT_TRUE("RUN!TEST-BasicInvokePutAdo" == request, "ADO_testing_plugin: invoke_put_ado failed");
    ASSERT_TRUE("VALUE_TO_PUT" == val, "ADO_testing_plugin: invoke_put_ado failed");
  }
  else if (k == "InvokeAdoCreateOnDemand") {
    void * src  = nullptr;
    void * dst  = nullptr;
    size_t size = KB(64);
    rc          = cb_allocate_pool_memory(size, 4096 /* alignment */, src);
    rc |= cb_allocate_pool_memory(size, 4096 /* alignment */, dst);
    ASSERT_TRUE(rc == S_OK, "InvokeAdoCreateOnDemand: failed");
    memset(src, 0, size);
    memset(dst, 0xA, size);
    auto start = rdtsc();
    memcpy(dst, src, size);
    pmem_persist(dst, size);
    auto stop = rdtsc();
    PLOG("%lu cycles for %lu (%f usec)", stop - start, size, common::cycles_to_usec(stop - start));
  }
  else if (k == "FindKeyCallback") {
    std::string matched_key;
    offset_t    matched_pos = 0;

    void *new_value_addr = nullptr;
    ASSERT_OK(cb_create_key(work_key, "aKey", 12, true, new_value_addr, nullptr, nullptr),
              "ADO_testing_plugin: create key failed");
    ASSERT_OK(cb_create_key(work_key, "bKey", 13, true, new_value_addr, nullptr, nullptr),
              "ADO_testing_plugin: create key failed");
    ASSERT_OK(cb_create_key(work_key, "cKey", 14, true, new_value_addr, nullptr, nullptr),
              "ADO_testing_plugin: create key failed");

    PLOG("issuing find key (pos=%lu)..", matched_pos);
    while ((rc = cb_find_key(".*",
                             matched_pos,                          // begin pos
                             component::IKVIndex::FIND_TYPE_NEXT,  // find type
                             matched_pos, matched_key)) == S_OK) {
      PLOG("match: (%s) @ %lu", matched_key.c_str(), matched_pos);
      matched_pos++;
      ASSERT_OK(rc, "ADO_testing_plugin: find_key failed");
    }
    rc = S_OK;  // reset RC
  }
  else if (k == "BasicAllocatePoolMemory") {
    void * result = nullptr;
    void * result2 = nullptr;
    size_t size   = KB(4);

    rc = cb_allocate_pool_memory(size, 4096 /* alignment */, result);
    ASSERT_TRUE(rc == S_OK, "BasicAllocatePoolMemory::allocate_pool_memory failed");
    PLOG("allocated %p", result);

    rc = cb_allocate_pool_memory(size, 4096 /* alignment */, result2);
    ASSERT_TRUE(rc == S_OK, "BasicAllocatePoolMemory::allocate_pool_memory failed");
    PLOG("allocated %p", result2);

    ASSERT_TRUE(result != result2, "BasicAllocatePoolMemory::allocate_pool_memory repeat allocation failed");

    ASSERT_OK(cb_free_pool_memory(size, result), "BasicAllocatePoolMemory::allocate_pool_memory freed failed");
    ASSERT_OK(cb_free_pool_memory(size, result2), "BasicAllocatePoolMemory::allocate_pool_memory freed failed");

  }
  else if (k == "BasicDetachedMemory") {
    ASSERT_TRUE(detached_value != nullptr, "BasicDetachedMemory::invalid detached ptr");

    /* free detached memory */
    rc = cb_free_pool_memory(detached_value_len, detached_value);
    ASSERT_TRUE(rc == S_OK, "BasicDetachedMemory::free_pool_memory failed");
  }
  else if (k == "GetReferenceVector") {
    component::IADO_plugin::Reference_vector v;
    rc = cb_get_reference_vector(0, 0, v);
    PMAJOR("rc (count=%lu, val=%p, val_len=%lu", v.count(), static_cast<const void *>(v.ref_array()),
           v.value_memory_size());

    ASSERT_TRUE(rc == S_OK, "get_reference_vector failed");
    ASSERT_TRUE(v.count() == 4, "GetReferenceVector:: unexpected vector size");

    auto ref = v.ref_array();
    for (size_t i = 0; i < v.count(); i++) {
      PLOG("kv: key=(%p, %lu) val=(%p, %lu)", ref[i].key, ref[i].key_len, ref[i].value, ref[i].value_len);
      ASSERT_FALSE(ref[i].key == nullptr, "GetReferenceVector::bad reference vector");
      ASSERT_TRUE(ref[i].key_len > 0, "GetReferenceVector::bad reference vector");
      ASSERT_FALSE(ref[i].value == nullptr, "GetReferenceVector::bad reference vector");
      ASSERT_TRUE(ref[i].value_len > 0, "GetReferenceVector::bad reference vector");
    }

    for (size_t i = 0; i < v.count(); i++) {
      PLOG("kv: key=(%s) val=(%s)", reinterpret_cast<char *>(ref[i].key), reinterpret_cast<char *>(ref[i].value));
    }

    rc = cb_free_pool_memory(v.value_memory_size(), v.value_memory());

    ASSERT_TRUE(rc == S_OK, "GetReferenceVector::free_pool_memory failed");
  }
  else if (k == "Iterator") {
    component::IKVStore::pool_iterator_t  iterator = nullptr;
    component::IKVStore::pool_reference_t r;
    while ((rc = cb_iterate(0, 0, iterator, r)) == S_OK) {
      PLOG("Iterator: ref (%.*s,%.*s,%s)", int(r.key_len), static_cast<const char*>(r.key),
           int(r.value_len), static_cast<const char *>(r.value), r.timestamp.str().c_str());
    }
    if (rc == E_NOT_IMPL) PLOG("Component does not support iterator.");

    ASSERT_TRUE(rc == E_NOT_IMPL || rc == E_OUT_OF_BOUNDS, "iterator failed");
    rc = S_OK;
  }
  else if (k == "IteratorTS") {
    std::string request(static_cast<const char *>(in_work_request), in_work_request_len);
    common::epoch_time_t ts(std::stoull(request));

    PLOG("IteratorTS: requested begin timestamp: %lu seconds", ts.seconds());
    unsigned cnt = 0;
    {
      component::IKVStore::pool_iterator_t  iterator = nullptr;
      component::IKVStore::pool_reference_t r;
      while ((rc = cb_iterate(ts, 0, iterator, r)) == S_OK) {
        PLOG("Iterator: ref (%.*s,%.*s,%s)", int(r.key_len), static_cast<const char*>(r.key),
             int(r.value_len), static_cast<const char *>(r.value), r.timestamp.str().c_str());
        cnt++;
      }
      if (rc == E_NOT_IMPL) PLOG("Component does not support iterator.");

      /* return count */
      uint64_t * rb = reinterpret_cast<uint64_t *>(malloc(sizeof(uint64_t)));
      if ( rb == nullptr )
      {
        throw std::bad_alloc();
      }
      *rb = cnt;
      //      *rb memcpy(rb, text.data(), rb_len);
      response_buffers.emplace_back(rb, sizeof(uint64_t), response_buffer_t::alloc_type_malloc{});
    }

    rc = S_OK;
  }
  else if (k == "Erase") {
    PLOG("performing self erase");
    rc = S_ERASE_TARGET;
  }
  else if (k.find("BasicAdoResponse") == 0) {

    /* return the key in the ADO response */
    const std::string text(key_addr, key_len);

    auto rb_len = text.length();
    void * rb = malloc(rb_len);
    if ( rb == nullptr )
    {
      throw std::bad_alloc();
    }
    memcpy(rb, text.data(), rb_len);
    response_buffers.emplace_back(rb, rb_len, response_buffer_t::alloc_type_malloc{});
    rc = S_OK;
  }
  else if (k == "GetReferenceVectorByTime") {
    component::IADO_plugin::Reference_vector v;
    rc = cb_get_reference_vector(0, 0, v);
    PMAJOR("rc (count=%lu, val=%p, val_len=%lu", v.count(), static_cast<const void *>(v.ref_array()),
           v.value_memory_size());

    ASSERT_TRUE(rc == S_OK, "get_reference_vector failed");
    ASSERT_TRUE(v.count() == 21, "get_reference_vector failed");

    {
      auto ref = v.ref_array();
      for (size_t i = 0; i < v.count(); i++) {
        PLOG("kv: key=(%p, %lu) val=(%p, %lu)", ref[i].key, ref[i].key_len, ref[i].value, ref[i].value_len);
        ASSERT_FALSE(ref[i].key == nullptr, "GetReferenceVector::bad reference vector");
        ASSERT_TRUE(ref[i].key_len > 0, "GetReferenceVector::bad reference vector");
        ASSERT_FALSE(ref[i].value == nullptr, "GetReferenceVector::bad reference vector");
        ASSERT_TRUE(ref[i].value_len > 0, "GetReferenceVector::bad reference vector");
      }
    }

    rc = cb_free_pool_memory(v.value_memory_size(), v.value_memory());
    ASSERT_TRUE(rc == S_OK, "GetReferenceVector::free_pool_memory failed");

    wmb();
    /* now do by time retrieval */
    {
      sleep(3);
      auto now = common::epoch_now();
      PLOG("epoch now: %s", now.str().c_str());
      wmb();
      sleep(1);
      {
        void *new_value_addr = nullptr;
        ASSERT_OK(cb_create_key(work_key, "aKey", 7, true, new_value_addr, nullptr, nullptr),
                  "create key failed GetReferenceVectorByTime");
        strcpy(static_cast<char *>(new_value_addr), "Hello!");
      }

      wmb();
      {
        void *new_value_addr = nullptr;
        ASSERT_OK(cb_create_key(work_key, "anotherKey", 7, true, new_value_addr, nullptr, nullptr),
                  "create key failed GetReferenceVectorByTime");
        strcpy(static_cast<char *>(new_value_addr), "World!");
      }

      wmb();
      rc = cb_get_reference_vector(now, 0, v);

      PMAJOR("by-time rc (count=%lu, val=%p, val_len=%lu", v.count(),
             static_cast<const void *>(v.ref_array()), v.value_memory_size());

      ASSERT_TRUE(v.count() == 2, "bad vector count GetReferenceVectorByTime");

      {
        auto ref = v.ref_array();
        for (size_t i = 0; i < v.count(); i++) {
          PLOG("kv: key=(%.*s) val=(%.*s)", int(ref[i].key_len),
               static_cast<const char *>(ref[i].key), int(ref[i].value_len),
               static_cast<const char *>(ref[i].value));
          ASSERT_FALSE(ref[i].key == nullptr,
                       "GetReferenceVector::bad reference vector");
          ASSERT_TRUE(ref[i].key_len > 0,
                      "GetReferenceVector::bad reference vector");
          ASSERT_FALSE(ref[i].value == nullptr,
                       "GetReferenceVector::bad reference vector");
          ASSERT_TRUE(ref[i].value_len > 0,
                      "GetReferenceVector::bad reference vector");
        }
      }

      rc = cb_free_pool_memory(v.value_memory_size(), v.value_memory());
      ASSERT_TRUE(rc == S_OK, "GetReferenceVector::free_pool_memory failed");
    }
  }
  else {
    /* this gets run for ado-perf performance test */
    //    pmem_memset(value, 0x1, value_len, 0);
    rc = S_OK;
  }

  if(option_DEBUG)
    PMAJOR("%s %s length %zu return %d", __FILE__, k.c_str(), key_len, rc);

  return rc;
}

status_t ADO_testing_plugin::shutdown()
{
  /* here you would put graceful shutdown code if any */
  return S_OK;
}

/**
 * Factory-less entry point
 *
 */
extern "C" void *factory_createInstance(component::uuid_t interface_iid)
{
  PLOG("instantiating ADO_testing_plugin");
  if (interface_iid == interface::ado_plugin)
    return static_cast<void *>(new ADO_testing_plugin());
  else
    return NULL;
}

#undef RESET_STATE
