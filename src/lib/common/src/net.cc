#include <sys/mman.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <common/exceptions.h>
#include <common/net.h>

namespace common
{

std::string get_eth_device_from_ip(const std::string& ipaddr)
{
  struct ifaddrs *ifaddr = nullptr, *ifa;
  int             s;
  char            host[NI_MAXHOST];
  std::string     result;

  if (getifaddrs(&ifaddr) == -1) throw General_exception("getifaddrs failed unexpectedly");

  for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;

    s = getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host, NI_MAXHOST, nullptr, 0, NI_NUMERICHOST);

    if (s == 0 && ipaddr == std::string(host)) {
      result = ifa->ifa_name;
      break;
    }
  }
  freeifaddrs(ifaddr);

  return result;
}

} // common
