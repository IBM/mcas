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


#ifndef COMANCHE_HSTORE_DEVDAX_MANAGER_H
#define COMANCHE_HSTORE_DEVDAX_MANAGER_H

#include <nupm/devdax_manager.h>

#include <string>

class Devdax_manager
	: public nupm::Devdax_manager
{
public:
	Devdax_manager(
		unsigned debug_level_
		, const std::string &dax_map
		, bool force_reset = false
	);
};

#endif
