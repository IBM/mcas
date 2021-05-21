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

#include <common/logging.h>

#define PREFIX "MM-PLUGIN-JEMALLOC:"

#ifdef DEBUG
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
#else
inline void PPLOG(const char * , ...) {}
#endif

#ifdef DEBUG
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
#else
inline void PPERR(const char * , ...){}
#endif

#ifdef DEBUG
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
#else
inline void PPNOTICE(const char * , ...) {}
#endif
