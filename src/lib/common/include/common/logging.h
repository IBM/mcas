/*
   eXokernel Development Kit (XDK)

   Samsung Research America Copyright (C) 2013
   IBM Corporation 2019

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.

   As a special exception, if you link the code in this file with
   files compiled with a GNU compiler to produce an executable,
   that does not cause the resulting executable to be covered by
   the GNU Lesser General Public License.  This exception does not
   however invalidate any other reasons why the executable file
   might be covered by the GNU Lesser General Public License.
   This exception applies to code released by its copyright holders
   in files containing the exception.
*/

/*
  Author(s):
  Copyright (C) 2016,2019, Daniel G. Waddington <daniel.waddington@ibm.com>
  Copyright (C) 2014, Daniel G. Waddington <daniel.waddington@acm.org>
*/

#ifndef __COMMON_LOGGING_H__
#define __COMMON_LOGGING_H__

#if defined(__cplusplus)
#include <cstdio>
#include <cstdarg>

/* optional: print a timestamp */
#define LOG_PRINT_TIMESTAMP 0

#if LOG_PRINT_TIMESTAMP
#include <chrono>

namespace
{
  double log_ts()
  {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }
}

#define LOG_TS_FMT " %f"
#define LOG_TS_ARG log_ts(),
#else
#define LOG_TS_FMT ""
#define LOG_TS_ARG
#endif

namespace common
{
  struct log_source
  {
  private:
	unsigned _debug_level;
  public:
    ~log_source() {}
    explicit log_source(const unsigned debug_level_)
      : _debug_level(debug_level_)
    {}
    unsigned debug_level() const { return _debug_level; }
    log_source(const log_source &other) = default;
  };
  /* g++ formast checking wants an argument formatted with %p to be a void *,
   * or at least not a char *. Function to turn any pointer into an type
   * acceptable to %p/
   */
  template <typename T> const void *p_fmt(const T t) { return static_cast<const void *>(t); }
}

#endif

#define NORMAL_CYAN "\033[36m"
#define NORMAL_MAGENTA "\033[35m"
#define NORMAL_BLUE "\033[34m"
#define NORMAL_YELLOW "\033[33m"
#define NORMAL_GREEN "\033[32m"
#define NORMAL_RED "\033[31m"

#define BRIGHT "\033[1m"
#define NORMAL_XDK "\033[0m"
#define RESET "\033[0m"

#define BRIGHT_CYAN "\033[1m\033[36m"
#define BRIGHT_MAGENTA "\033[1m\033[35m"
#define BRIGHT_BLUE "\033[1m\033[34m"
#define BRIGHT_YELLOW "\033[1m\033[33m"
#define BRIGHT_GREEN "\033[1m\033[32m"
#define BRIGHT_RED "\033[1m\033[31m"

#define WHITE_ON_RED "\033[41m"
#define WHITE_ON_GREEN "\033[42m"
#define WHITE_ON_YELLOW "\033[43m"
#define WHITE_ON_BLUE "\033[44m"
#define WHITE_ON_MAGENTA "\033[44m"

#define ESC_LOG NORMAL_GREEN
#define ESC_DBG NORMAL_YELLOW
#define ESC_INF NORMAL_CYAN
#define ESC_WRN NORMAL_RED
#define ESC_ERR BRIGHT_RED
#define ESC_END "\033[0m"

void pr_info(const char * format, ...) __attribute__((format(printf, 1, 2)));

inline void pr_info(const char * format, ...)
{
#ifdef CONFIG_DEBUG
  static constexpr size_t m_max_buffer = 512;
  va_list args;
  va_start(args, format);
  char buffer[m_max_buffer];
  vsnprintf(buffer, m_max_buffer, format, args);
  va_end(args);
  fprintf(stderr, "%s[LOG]:" LOG_TS_FMT " %s %s\n", ESC_LOG, LOG_TS_ARG buffer, ESC_END);
#else
  (void)format;
#endif
}

void pr_error(const char * format, ...) __attribute__((format(printf, 1, 2)));
inline void pr_error(const char * format, ...)
{
#ifdef CONFIG_DEBUG
  static constexpr size_t m_max_buffer = 512;
  va_list args;
  va_start(args, format);
  char buffer[m_max_buffer];
  vsnprintf(buffer, m_max_buffer, format, args);
  va_end(args);
  fprintf(stderr, "%s[LOG]:" LOG_TS_FMT " %s %s\n", ESC_ERR, LOG_TS_ARG buffer, ESC_END);
#else
  (void)format;
#endif
}

void PLOG(const char * format, ...) __attribute__((format(printf, 1, 2)));
inline void PLOG(const char * format, ...)
{
#ifdef CONFIG_DEBUG
  static constexpr size_t m_max_buffer = 512;
  va_list args;
  va_start(args, format);
  char buffer[m_max_buffer];
  vsnprintf(buffer, m_max_buffer, format, args);
  va_end(args);
  fprintf(stderr, "%s[LOG]:" LOG_TS_FMT " %s %s\n", ESC_LOG, LOG_TS_ARG buffer, ESC_END);
#else
  (void)format;
#endif
}

void PDBG(const char * format, ...) __attribute__((format(printf, 1, 2)));
inline void PDBG(const char * format, ...)
{
#ifdef CONFIG_DEBUG
  static constexpr size_t m_max_buffer = 512;
  va_list args;
  va_start(args, format);
  char buffer[m_max_buffer];
  vsnprintf(buffer, m_max_buffer, format, args);
  va_end(args);
  fprintf(stderr, "%s[DBG]:" LOG_TS_FMT " %s %s\n", ESC_DBG, LOG_TS_ARG buffer, ESC_END);
#else
  (void)format;
#endif
}

void PINF(const char * format, ...) __attribute__((format(printf, 1, 2)));
inline void PINF(const char * format, ...)
{
#ifdef CONFIG_DEBUG
  static constexpr size_t m_max_buffer = 512;
  va_list args;
  va_start(args, format);
  char buffer[m_max_buffer];
  vsnprintf(buffer, m_max_buffer, format, args);
  va_end(args);
  fprintf(stderr, "%s %s %s\n", ESC_INF, buffer, ESC_END);
#else
  (void)format;
#endif
}

void PWRN(const char * format, ...) __attribute__((format(printf, 1, 2)));
inline void PWRN(const char * format, ...)
{
#ifdef CONFIG_DEBUG
  static constexpr size_t m_max_buffer = 512;
  va_list args;
  va_start(args, format);
  char buffer[m_max_buffer];
  vsnprintf(buffer, m_max_buffer, format, args);
  va_end(args);
  fprintf(stderr, "%s[WRN]:" LOG_TS_FMT " %s %s\n", ESC_WRN, LOG_TS_ARG buffer, ESC_END);
#else
  (void)format;
#endif
}

void PERR(const char * format, ...) __attribute__((format(printf, 1, 2)));
inline void PERR(const char * format, ...)
{
#ifdef CONFIG_DEBUG
  static constexpr size_t m_max_buffer = 512;
  va_list args;
  va_start(args, format);
  char buffer[m_max_buffer];
  vsnprintf(buffer, m_max_buffer, format, args);
  va_end(args);
  fprintf(stderr, "%sError:" LOG_TS_FMT " %s %s\n", ESC_ERR, LOG_TS_ARG buffer, ESC_END);
#else
  (void)format;
#endif
}

void PEXCEP(const char * format, ...) __attribute__((format(printf, 1, 2)));
inline void PEXCEP(const char * format, ...)
{
#ifdef CONFIG_DEBUG
  static constexpr size_t m_max_buffer = 512;
  va_list args;
  va_start(args, format);
  char buffer[m_max_buffer];
  vsnprintf(buffer, m_max_buffer, format, args);
  va_end(args);
  fprintf(stderr, "%sException:" LOG_TS_FMT " %s %s\n", ESC_ERR, LOG_TS_ARG buffer, ESC_END);
#else
  (void)format;
#endif
}

void PNOTICE(const char * format, ...) __attribute__((format(printf, 1, 2)));
inline void PNOTICE(const char * format, ...)
{
#ifdef CONFIG_DEBUG
  static constexpr size_t m_max_buffer = 512;
  va_list args;
  va_start(args, format);
  char buffer[m_max_buffer];
  vsnprintf(buffer, m_max_buffer, format, args);
  va_end(args);
  fprintf(stderr, "%sNOTICE:" LOG_TS_FMT " %s %s\n", BRIGHT_RED, LOG_TS_ARG buffer, ESC_END);
#else
  (void)format;
#endif
}

void PMAJOR(const char * format, ...) __attribute__((format(printf, 1, 2)));
inline void PMAJOR(const char * format, ...)
{
#ifdef CONFIG_DEBUG
  static constexpr size_t m_max_buffer = 512;
  va_list args;
  va_start(args, format);
  char buffer[m_max_buffer];
  vsnprintf(buffer, m_max_buffer, format, args);
  va_end(args);
  fprintf(stderr, "%s[+]" LOG_TS_FMT " %s %s\n", NORMAL_BLUE, LOG_TS_ARG buffer, ESC_END);
#else
  (void)format;
#endif
}

void PLOG2(const char *color, const char * format, ...) __attribute__((format(printf, 2, 3)));
inline void PLOG2(const char * color, const char * format, ...)
{
#ifdef CONFIG_DEBUG
  static constexpr size_t m_max_buffer = 512;
  va_list args;
  va_start(args, format);
  char buffer[m_max_buffer];
  vsnprintf(buffer, m_max_buffer, format, args);
  va_end(args);
  fprintf(stderr, "%s[+]" LOG_TS_FMT " %s %s\n", color, LOG_TS_ARG buffer, ESC_END);
#else
  (void) format;
  (void) color;
#endif
}

/* one-line conditional PLOG */

// clang-format off

#define GET_MACRO(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,NAME,...) NAME

#define CPLOG_10(level,format,p0,p1,p2,p3,p4,p5,p6,p7) if ( (level) < this->debug_level() ) { PLOG(format,p0,p1,p2,p3,p4,p5,p6,p7); }
#define CPLOG_9(level,format,p0,p1,p2,p3,p4,p5,p6) if ( (level) < this->debug_level() ) { PLOG(format,p0,p1,p2,p3,p4,p5,p6); }
#define CPLOG_8(level,format,p0,p1,p2,p3,p4,p5) if ( (level) < this->debug_level() ) { PLOG(format,p0,p1,p2,p3,p4,p5); }
#define CPLOG_7(level,format,p0,p1,p2,p3,p4) if ( (level) < this->debug_level() ) { PLOG(format,p0,p1,p2,p3,p4); }
#define CPLOG_6(level,format,p0,p1,p2,p3) if ( (level) < this->debug_level() ) { PLOG(format,p0,p1,p2,p3); }
#define CPLOG_5(level,format,p0,p1,p2) if ( (level) < this->debug_level() ) { PLOG(format,p0,p1,p2); }
#define CPLOG_4(level,format,p0,p1) if ( (level) < this->debug_level() ) { PLOG(format,p0,p1); }
#define CPLOG_3(level,format,p0) if ( (level) < this->debug_level() ) { PLOG(format,p0); }
#define CPLOG_2(level,msg) if ( (level) < this->debug_level() ) { PLOG(msg); }
#define CPLOG_1(level)
#define CPLOG(...) GET_MACRO(__VA_ARGS__,CPLOG_10,CPLOG_9,CPLOG_8,CPLOG_7,CPLOG_6,CPLOG_5,CPLOG_4,CPLOG_3,CPLOG_2,CPLOG_1,CPLOG_0)(__VA_ARGS__)

#define CPINF_10(level,format,p0,p1,p2,p3,p4,p5,p6,p7) if ( (level) < this->debug_level() ) { PINF(format,p0,p1,p2,p3,p4,p5,p6,p7); }
#define CPINF_9(level,format,p0,p1,p2,p3,p4,p5,p6) if ( (level) < this->debug_level() ) { PINF(format,p0,p1,p2,p3,p4,p5,p6); }
#define CPINF_8(level,format,p0,p1,p2,p3,p4,p5) if ( (level) < this->debug_level() ) { PINF(format,p0,p1,p2,p3,p4,p5); }
#define CPINF_7(level,format,p0,p1,p2,p3,p4) if ( (level) < this->debug_level() ) { PINF(format,p0,p1,p2,p3,p4); }
#define CPINF_6(level,format,p0,p1,p2,p3) if ( (level) < this->debug_level() ) { PINF(format,p0,p1,p2,p3); }
#define CPINF_5(level,format,p0,p1,p2) if ( (level) < this->debug_level() ) { PINF(format,p0,p1,p2); }
#define CPINF_4(level,format,p0,p1) if ( (level) < this->debug_level() ) { PINF(format,p0,p1); }
#define CPINF_3(level,format,p0) if ( (level) < this->debug_level() ) { PINF(format,p0); }
#define CPINF_2(level,msg) if ( (level) < this->debug_level() ) { PINF(msg); }
#define CPINF_1(level)
#define CPINF(...) GET_MACRO(__VA_ARGS__,CPINF_10,CPINF_9,CPINF_8,CPINF_7,CPINF_6,CPINF_5,CPINF_4,CPINF_3,CPINF_2,CPINF_1,CPINF_0)(__VA_ARGS__)

#define CPWRN_10(level,format,p0,p1,p2,p3,p4,p5,p6,p7) if ( (level) < this->debug_level() ) { PWRN(format,p0,p1,p2,p3,p4,p5,p6,p7); }
#define CPWRN_9(level,format,p0,p1,p2,p3,p4,p5,p6) if ( (level) < this->debug_level() ) { PWRN(format,p0,p1,p2,p3,p4,p5,p6); }
#define CPWRN_8(level,format,p0,p1,p2,p3,p4,p5) if ( (level) < this->debug_level() ) { PWRN(format,p0,p1,p2,p3,p4,p5); }
#define CPWRN_7(level,format,p0,p1,p2,p3,p4) if ( (level) < this->debug_level() ) { PWRN(format,p0,p1,p2,p3,p4); }
#define CPWRN_6(level,format,p0,p1,p2,p3) if ( (level) < this->debug_level() ) { PWRN(format,p0,p1,p2,p3); }
#define CPWRN_5(level,format,p0,p1,p2) if ( (level) < this->debug_level() ) { PWRN(format,p0,p1,p2); }
#define CPWRN_4(level,format,p0,p1) if ( (level) < this->debug_level() ) { PWRN(format,p0,p1); }
#define CPWRN_3(level,format,p0) if ( (level) < this->debug_level() ) { PWRN(format,p0); }
#define CPWRN_2(level,msg) if ( (level) < this->debug_level() ) { PWRN(msg); }
#define CPWRN_1(level)
#define CPWRN(...) GET_MACRO(__VA_ARGS__,CPWRN_10,CPWRN_9,CPWRN_8,CPWRN_7,CPWRN_6,CPWRN_5,CPWRN_4,CPWRN_3,CPWRN_2,CPWRN_1,CPWRN_0)(__VA_ARGS__)

// clang-format on


#endif  // __COMMON_LOGGING_H__
