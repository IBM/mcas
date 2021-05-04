#ifndef __SAFE_PRINT_H__
#define __SAFE_PRINT_H__

#include <stdarg.h>
#include <string.h>
#include <unistd.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

void SAFE_PRINT(const char * format, ...) __attribute__((format(printf, 1, 2)));

#ifdef DEBUG
inline void SAFE_PRINT(const char * format, ...)
{
  static const size_t m_max_buffer = 512;
  va_list args;
  va_start(args, format);
  char buffer[m_max_buffer];
  char formatb[m_max_buffer];
  sprintf(formatb, "%s%s%s\n", NORMAL_CYAN, format, RESET);
  vsnprintf(buffer, m_max_buffer, formatb, args);
  va_end(args);
  write(1, buffer, strlen(buffer));
}
#else
inline void SAFE_PRINT(const char * format, ...)
{
}
#endif

#pragma GCC diagnostic pop
#endif
