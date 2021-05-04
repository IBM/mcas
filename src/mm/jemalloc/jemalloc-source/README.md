git clone git@github.com:jemalloc/jemalloc.git

./configure --prefix=${HOME}/mcas/build/src/mm/jemalloc-dist --disable-prof-gcc --enable-prof-libunwind --disable-initial-exec-tls --with-jemalloc-prefix=jel_

#--without-export 
#--with-mangling=malloc:__jemalloc_malloc,free:__jemalloc_free 

make build_lib_static
