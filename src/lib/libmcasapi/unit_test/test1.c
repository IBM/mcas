#include <stdio.h>
#include <assert.h>

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
  mcas_pool_t pool = mcas_create_pool(session, "myPool", MB(64), 0);

  assert(mcas_close_pool(session, pool) == 0);

  /* close session */
  assert(mcas_close_session(session) == 0);

  return 0;
}
