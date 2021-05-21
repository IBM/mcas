#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include <vector>
//#define TEST_LOAD

int main()
{
#ifdef TEST_LOAD
  static const char * PLUGIN_PATH = "dist/lib/libmm-plugin-jemalloc.so";
  void * mod = dlopen(PLUGIN_PATH, RTLD_NOW | RTLD_DEEPBIND);
  if(!mod) printf("Error: %s\n", dlerror());
#endif
  
  printf("> Test prog.\n");

  {
    printf("Performing 1MB mallocs ...\n");
    std::vector<void*> v(10);
    for(unsigned i=0;i<10;i++) {
      v.push_back(malloc(1024*1024));
    }

    printf("Freeing previous allocs ...\n");
    for(unsigned i=0;i<10;i++) {
      free(v[i]);
    }
  }

  {
    printf("Performing random 8-64B mallocs ...\n");
    static size_t count = 10000;
    std::vector<void*> v(count);
    for(unsigned i=0;i<count;i++) {
      v.push_back(malloc(8+(rand() % 56)));
    }

    printf("Freeing previous allocs ...\n");
    for(unsigned i=0;i<10;i++) {
      free(v[i]);
    }
  }
  
  
  {
    printf("Performing large malloc ...\n");
    size_t s = 1024 * 4096; // 4MiB
    printf("> allocating %lu bytes\n", s);
    void * p = malloc(s);
    printf("> result of malloc: p=%p\n", p);
    memset(p, 0, s);
    free(p);
  }

  {
    printf("Performing calloc ...\n");
    void * p = calloc(32, 128);
    printf(">result of calloc: p=%p\n", p);
    free(p);
  }
  

  printf("> Done.\n");

  return 0;
}
