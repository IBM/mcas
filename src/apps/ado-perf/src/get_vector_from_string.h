#ifndef _GET_VECTOR_FROM_STRING_H
#define _GET_VECTOR_FROM_STRING_H

#include <stdexcept> /* domain_error */
#include <string>
#include <sstream>
#include <utility> /* pair */
#include <vector>

template <typename T>
  struct half_open_interval
    : public std::pair<T, T>
  {};

template <typename T>
  std::istream &operator>>(std::istream &i_, half_open_interval<T> &r)
  {
    i_ >> r.first;
    if ( ! i_ ) { throw std::domain_error("ill-formed element"); }
    switch ( auto c = i_.peek() )
    {
    case '-': /* inclusive range */
      i_.get();
      i_ >> r.second;
      if ( ! i_ ) { throw std::domain_error("ill-formed range"); }
      ++r.second;
      break;
    case ':': /* length */
      i_.get();
      {
        unsigned length;
        i_ >> length;
        if ( ! i_ ) { throw std::domain_error("ill-formed length"); }
        r.second = r.first + length;
      }
      break;
    default:
      r.second = r.first + 1U;
      break;
    }
    return i_;
  }

/* was get_cpu_vector_from_string, but now also used for devices */
template <typename T>
  std::vector<T> get_vector_from_string(const std::string &core_string)
  {
    std::istringstream core_stream(core_string);
    std::vector<T> cores;

    do {
      half_open_interval<T> r;
      core_stream >> r;
      for ( ; r.first != r.second; ++r.first )
      {
        cores.push_back(r.first);
      }
    } while ( core_stream.get() == ',' );

    if ( core_stream )
    {
      std::string s;
      core_stream >> s; 
      throw std::domain_error("Unrecognized trailing characters '" + s + "' in list");
    }
    return cores;
  }

#endif
