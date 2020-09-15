#ifndef _DOTTED_PAIR_H_
#define _DOTTED_PAIR_H_

#include <istream> /* istream */
#include <ostream> /* ostream */
#include <sstream> /* ostrintstream */
#include <stdexcept> /* domain_error */

template <typename T>
  struct dotted_pair
  {
  public:
    T first;
    T second;
    explicit dotted_pair(T first_, T second_)
      : first(first_)
      , second(second_)
    {}
    dotted_pair() : dotted_pair(T(), T()) {}
    dotted_pair &operator++() { ++second; return *this; }
    dotted_pair &operator+=(T t_) { second += t_; return *this; }
  };

/* read a "dotted pair." If no dot, the first element is 0 and the second element is the value read */
template <typename T>
  std::istream &operator>>(std::istream &i_, dotted_pair<T> &p_)
  {
    i_ >> p_.first;
    if ( ! i_.eof() && i_.peek() == '.' )
    {
      i_.get();
      i_ >> p_.second;
    }
    else
    {
      p_.second = p_.first;
      p_.first = T();
    }
    return i_;
  }

/* write a "dotted pair." */
template <typename T>
  std::ostream &operator<<(std::ostream &o_, const dotted_pair<T> &p_)
  {
    return o_ << p_.first << "." << p_.second;
  }

/* compare "dotted pairs" */
template <typename T>
  void check_comparable(const dotted_pair<T> &a_, const dotted_pair<T> &b_)
  {
    if ( a_.first != b_.first )
    {
      std::ostringstream o;
      o << "dotted pair " << a_ << " is not comparable to dotted pair " << b_;
      throw std::domain_error(o.str());
    }
  }
template <typename T>
  bool operator==(const dotted_pair<T> &a_, const dotted_pair<T> &b_)
  {
    check_comparable(a_, b_);
    return a_.second == b_.second;
  }
template <typename T>
  bool operator!=(const dotted_pair<T> &a_, const dotted_pair<T> &b_)
  {
    return ! ( a_ == b_ );
  }
template <typename T>
  bool operator<(const dotted_pair<T> &a_, const dotted_pair<T> &b_)
  {
    check_comparable(a_, b_);
    return a_.second < b_.second;
  }

template <typename T>
  dotted_pair<T> operator+(dotted_pair<T> p_, T t_)
  {
    return p_ += t_;
  }

#endif
