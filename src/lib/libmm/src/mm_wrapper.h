#ifndef __MM_WRAPPER_H__
#define __MM_WRAPPER_H__

#include <stdlib.h>

using malloc_function_t = void* (*)(size_t);
using calloc_function_t = void* (*)(size_t, size_t);
using free_function_t = void (*)(void *ptr);
using aligned_alloc_function_t =  void* (*)(size_t alignment, size_t size);
using realloc_function_t = void * (*)(void *ptr, size_t size);
using memalign_function_t = void * (*)(size_t alignment, size_t size);
using vfprintf_function_t = int (*) (FILE *, const char *format, va_list ap);
using puts_function_t = int (*)(const char *s);
using fputs_function_t = int (*)(const char *s, FILE * stream);
using malloc_usable_size_function_t = size_t (*) (void *ptr);

#endif // __MM_WRAPPER_H__
