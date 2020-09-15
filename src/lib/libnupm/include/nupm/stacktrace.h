/*
   Copyright [2017-2019] [IBM Corporation]
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



#ifndef _CORE_STACKTRACE_H_
#define _CORE_STACKTRACE_H_

#include <cxxabi.h>    // for __cxa_demangle
#include <dlfcn.h>     // for dladdr
#include <execinfo.h>  // for backtrace
#include <cxxabi.h>
#include <execinfo.h>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

namespace core
{
// This function produces a stack backtrace with demangled function & method
// names.
std::string stack_trace(int skip = 1) {
  void *callstack[128];
  const int nMaxFrames = sizeof(callstack) / sizeof(callstack[0]);
  char buf[1024];
  int nFrames = backtrace(callstack, nMaxFrames);
  char **symbols = backtrace_symbols(callstack, nFrames);

  std::ostringstream trace_buf;
  for (int i = skip; i < nFrames; i++) {
    Dl_info info;
    if (dladdr(callstack[i], &info) && info.dli_sname) {
      char *demangled = NULL;
      int status = -1;
      if (info.dli_sname[0] == '_') {
        demangled = abi::__cxa_demangle(info.dli_sname, NULL, 0, &status);
      }
      snprintf(buf, sizeof(buf), "%-3d %*p %s + %zd\n", i,
               int(2 + sizeof(void *) * 2), callstack[i],
               status == 0 ? demangled
                           : info.dli_sname == 0 ? symbols[i] : info.dli_sname,
               static_cast<char *>(callstack[i]) - static_cast<char *>(info.dli_saddr));
      free(demangled);

      trace_buf << buf;
    }
    else { /* only print resolved symbols */
      //          snprintf(buf, sizeof(buf), "%-3d %*p %s\n", i, int(2 +
      //          sizeof(void*) * 2), callstack[i], symbols[i]);
    }
  }
  free(symbols);
  if (nFrames == nMaxFrames) trace_buf << "[truncated]\n";
  return trace_buf.str();
}

// std::string stack_trace()
// {
//   std::stringstream trace_output;
//   void *array[10];
//   size_t size;
//   char **strings;
//   size_t i;

//   size = backtrace (array, 10);
//   strings = backtrace_symbols (array, size);

//   printf ("Obtained %zd stack frames.\n", size);

//   for (i = 0; i < size; i++)
//      printf ("%s\n", strings[i]);

//   ::free(strings);
//   return trace_output.str();
// }

// /** Print a demangled stack backtrace of the caller function to FILE* out. */
// static inline void print_stacktrace(FILE *out = stderr, unsigned int
// max_frames = 63)
// {
//     fprintf(out, "stack trace:\n");

//     // storage array for stack trace address data
//     void* addrlist[max_frames+1];

//     // retrieve current stack addresses
//     int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));

//     if (addrlen == 0) {
// 	fprintf(out, "  <empty, possibly corrupt>\n");
// 	return;
//     }

//     // resolve addresses into strings containing
//     "filename(function+address)",
//     // this array must be free()-ed
//     char** symbollist = backtrace_symbols(addrlist, addrlen);

//     // allocate string which will be filled with the demangled function name
//     size_t funcnamesize = 256;
//     char* funcname = (char*)malloc(funcnamesize);

//     // iterate over the returned symbol lines. skip the first, it is the
//     // address of this function.
//     for (int i = 1; i < addrlen; i++)
//     {
// 	char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

// 	// find parentheses and +address offset surrounding the mangled name:
// 	// ./module(function+0x15c) [0x8048a6d]
// 	for (char *p = symbollist[i]; *p; ++p)
// 	{
// 	    if (*p == '(')
// 		begin_name = p;
// 	    else if (*p == '+')
// 		begin_offset = p;
// 	    else if (*p == ')' && begin_offset) {
// 		end_offset = p;
// 		break;
// 	    }
// 	}

// 	if (begin_name && begin_offset && end_offset
// 	    && begin_name < begin_offset)
// 	{
// 	    *begin_name++ = '\0';
// 	    *begin_offset++ = '\0';
// 	    *end_offset = '\0';

// 	    // mangled name is now in [begin_name, begin_offset) and caller
// 	    // offset in [begin_offset, end_offset). now apply
// 	    // __cxa_demangle():

// 	    int status;
// 	    char* ret = abi::__cxa_demangle(begin_name,
// 					    funcname, &funcnamesize, &status);
// 	    if (status == 0) {
// 		funcname = ret; // use possibly realloc()-ed string
// 		fprintf(out, "  %s : %s+%s\n",
// 			symbollist[i], funcname, begin_offset);
// 	    }
// 	    else {
// 		// demangling failed. Output function name as a C function with
// 		// no arguments.
// 		fprintf(out, "  %s : %s()+%s\n",
// 			symbollist[i], begin_name, begin_offset);
// 	    }
// 	}
// 	else
// 	{
// 	    // couldn't parse the line? print the whole line.
// 	    fprintf(out, "??  %s\n", symbollist[i]);
// 	}
//     }

//     free(funcname);
//     free(symbollist);
// }

}  // namespace core
#endif  // _CORE_STACKTRACE_H_
