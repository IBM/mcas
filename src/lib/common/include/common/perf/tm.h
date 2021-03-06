/*
   Copyright [2017-2021] [IBM Corporation]
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

#ifndef MCAS_COMMON_PERF_TM_ACTUAL_H
#define MCAS_COMMON_PERF_TM_ACTUAL_H

#include "tm_fwd.h"

#include <common/perf/duration_stat.h>
#include <common/perf/timer_split.h>
#include <common/perf/timer_to_exit.h>
#include <common/perf/writer_at_exit.h>
#include <iostream>

/*
 * Definitions for opionally passing a timer_split argument.
 *
 * TM_ROOT define a timer, sink, and scope. and start the timer.
 * TM_SCOPE(tag) binds a timer to an output (tagged by the name "tag") until the end of scope.
 * TM_SPLIT(tag2) charges time accumulated since the start of scope (or the previous split) from
 *   the scope "tag" to the split "tag2". Note that the time tansferred to the split "tag2"
 *   *ends*, does not begin, at the split point. When possible, I suggest starting a new block
 *   and using TM_SCOPE rather than using TM_SPLIT.
 *
 */

#define QUOTE(X) #X

#if MCAS_TM_ENABLED
/* declaration of the root timer_split instancei: once per thread, and probably in a function call argument list */
#define TM_ROOT(tag) common::perf::timer_split tm; const common::perf::timer_to_exit *pa_ = nullptr; _TM_SCOPE_DEF(tag) common::perf::timer_to_exit tte_##tag{pa_, tm, w_##tag};
/* timer_split actual formal argument (without and with other args) */
#define TM_ACTUAL0 const common::perf::timer_to_exit *&pa_
#define TM_ACTUAL TM_ACTUAL0,
#define TM_REF0 pa_
#define TM_REF TM_REF0,
#define _TM_SCOPE_DEF(tag) \
	static common::perf::writer_at_exit<common::perf::duration_stat> w_##tag(std::cerr, __LINE__, __FILE__, __func__, QUOTE(tag));
/* Note we with to redefine the name tte_ or this scope. tter is a reference to refer to the old tte_
 * after its name has been shadowed.
 */
#define _TM_SCOPE_USE(tag) \
	_Pragma("GCC diagnostic push"); \
       	_Pragma("GCC diagnostic ignored \"-Wshadow\""); \
	common::perf::timer_to_exit tte_##tag{pa_, w_##tag}; \
       	_Pragma("GCC diagnostic pop");
#define TM_SCOPE(tag) _TM_SCOPE_DEF(tag) _TM_SCOPE_USE(tag);
#define TM_SPLIT(tag) _TM_SCOPE_DEF(tag) pa_->split(w_##tag)
#else
#define TM_ROOT(tag)
#define TM_ACTUAL0
#define TM_ACTUAL
#define TM_REF0
#define TM_REF
// #define TM_REF_VOID ((void) 0)
#define TM_SCOPE(tag)
#define TM_SPLIT(tag) do {} while (0)
#endif

#endif
