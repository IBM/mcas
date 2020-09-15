/*
   Copyright [2017-2020] [IBM Corporation]
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
#ifndef _mcas_COMMON_PROFILER_H_
#define _mcas_COMMON_PROFILER_H_

#if defined HAS_PROFILER
#include <gperftools/profiler.h> /* No __has_include until C++17 */
#else
namespace
{
	int ProfilerStart(const char *) { return -1; }
	void ProfilerStop() {}
	void ProfilerFlush() {}
}
#endif

#include <common/logging.h>
#include <common/moveable_value.h>
#include <cstdlib> /* getenv */
#include <string>

namespace common
{
	struct profiler
	{
	private:
		std::string _file;
		common::moveable_value<bool> _running;
	public:
		explicit profiler(const std::string &file_, bool start_ = true)
			: _file(file_)
			, _running(false)
		{
			if ( start_ || bool(std::getenv("PROFILE")) )
			{
				start();
			}
		}

		explicit profiler(const char *file_, bool start_ = true)
			: profiler(std::string(file_ ? file_ : ""), start_)
		{}

		~profiler()
		{
			if ( _running )
			{
				ProfilerStop();
				ProfilerFlush();
			}
		}

		void start()
		{
			if ( ! _running && _file.size() )
			{
				_running = bool(ProfilerStart(_file.c_str()));
				PLOG("Profile begins %s", _file.c_str());
			}
		}
	};
}

#endif
