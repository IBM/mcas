#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <mcas_api_wrapper.h>

#define KB(X) (X << 10)
#define MB(X) (X << 20)
#define GB(X) ((1ULL << 30) * X)
#define TB(X) ((1ULL << 40) * X)

int main(int argc, char* argv[])
{
  if(argc != 3) {
    printf("libmcasapi-test <server-url> <device>\n");
    return 0;
  }

  /* open session */
  mcas_session_t session = mcas_open_session_ex(argv[1], /* server */
                                                argv[2], /* net device */
                                                2,
                                                30);  

  /* create a pool */
  mcas_pool_t pool;
  assert(mcas_create_pool(session, "myPool", MB(64), 0, &pool) == 0);

  assert(mcas_put(pool, "someKey", "someValue", 0) == 0);
  assert(mcas_put(pool, "someOtherKey", "someOtherValue", 0) == 0);

  {
    void * v;
    size_t vlen = 0;
    assert(mcas_get(pool, "someKey", &v, &vlen) == 0);
    assert(vlen > 0);
    assert(mcas_free_memory(session, v) == 0);
  }

  printf("count: %lu\n", mcas_count(pool));

  {
    uint64_t * p = NULL;
    size_t p_count = 0;
    assert(mcas_get_attribute(pool,
                              NULL,
                              ATTR_COUNT,
                              &p,
                              &p_count) == 0);
    assert(p_count == 1);
    assert(p[0] == 2);
    printf("count-->: %lu\n", p[0]);
    free(p);
  }
  

  /* erase key */
  assert(mcas_erase(pool, "someKey") == 0);
  
  /* allocate some memory */
  void * ptr = aligned_alloc(4096, MB(2));
  mcas_memory_handle_t mr;
  memset(ptr, 0xf, MB(2));
  assert(mcas_register_direct_memory(session, ptr, MB(2), &mr) == 0);


  /* perform direct transfers */
  assert(mcas_put_direct_ex(pool,
                            "myBigKey",
                            ptr,
                            MB(2),
                            mr,
                            0) == 0);

  {
    size_t s = MB(2);
    assert(mcas_get_direct_ex(pool,
                              "myBigKey",
                              ptr,
                              &s,
                              mr) == 0);
  }


  /* clean up memory */
  assert(mcas_unregister_direct_memory(session, mr) == 0);
  free(ptr);
  assert(mcas_close_pool(pool) == 0);

  /* delete pool */
  assert(mcas_delete_pool(session, "myPool") == 0);
  
  /* close session */
  assert(mcas_close_session(session) == 0);

  return 0;
}
