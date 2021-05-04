#include <common/logging.h>

#define PREFIX "MM-PLUGIN-JEMALLOC:"
void PPLOG(const char * format, ...)
{
  static const size_t m_max_buffer = 512;
  va_list args;
  va_start(args, format);
  char buffer[m_max_buffer];
  char formatb[m_max_buffer];
  sprintf(formatb, "%s%s%s%s\n", NORMAL_CYAN, PREFIX, format, RESET);
  vsnprintf(buffer, m_max_buffer, formatb, args);
  va_end(args);
  //  write(1, buffer, strlen(buffer));
  printf("%s",buffer);
}

void PPERR(const char * format, ...)
{
  static const size_t m_max_buffer = 512;
  va_list args;
  va_start(args, format);
  char buffer[m_max_buffer];
  char formatb[m_max_buffer];
  sprintf(formatb, "%s%s%s%s\n", BRIGHT_RED, "error:", format, RESET);
  vsnprintf(buffer, m_max_buffer, formatb, args);
  va_end(args);
  //  write(1, buffer, strlen(buffer));
  printf("%s",buffer);
}


void PPNOTICE(const char * format, ...)
{
  static const size_t m_max_buffer = 512;
  va_list args;
  va_start(args, format);
  char buffer[m_max_buffer];
  char formatb[m_max_buffer];
  sprintf(formatb, "%s%s%s%s\n", BRIGHT_CYAN, PREFIX, format, RESET);
  vsnprintf(buffer, m_max_buffer, formatb, args);
  va_end(args);
  //  write(1, buffer, strlen(buffer));
  printf("%s",buffer);
}
