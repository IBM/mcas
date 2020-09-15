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

#ifndef __MCAS_SECURITY_H__
#define __MCAS_SECURITY_H__

#include <memory>
#include <string>

namespace mcas
{
class Shard_security_state;

class Shard_security {
 private:
  unsigned debug_level() const { return 3; }

 public:
  Shard_security(const std::string& certs_path);

  inline bool auth_enabled() const { return _auth_enabled; }

 private:
  const std::string                     _certs_path;
  bool                                  _auth_enabled;
  std::shared_ptr<Shard_security_state> _state;
};
}  // namespace mcas

#endif  // __MCAS_SECURITY_H__
