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

#ifndef _MCAS_JSON_H_
#define _MCAS_JSON_H_

#include <common/common.h>
#include <cstdint>
#include <deque>
#include <string>
#include <iterator> // make_move_iterator
#include <memory> // unique_ptr
#include <utility> // forward

namespace common
{
  namespace json
  {
    namespace schema
    {
      /* JSON types */
      static constexpr const char *string = "string";
      static constexpr const char *integer = "integer";
      static constexpr const char *number = "number";
      static constexpr const char *object = "object";
      static constexpr const char *array = "array";
      static constexpr const char *boolean = "boolean";
      static constexpr const char *null = "null";
      /* common kyewords */
      static constexpr const char *type = "type";
      /* string keywords */
      static constexpr const char *minLength = "minLength";
      static constexpr const char *maxLength = "maxLength";
      static constexpr const char *pattern = "pattern";
      /* numeric keywords */
      static constexpr const char *minimum = "minimum";
      static constexpr const char *maximum = "maximum";
      static constexpr const char *exclusiveMinumum = "exclusiveMinumum";
      static constexpr const char *exclusiveMaximum = "exclusiveMaximum";
      static constexpr const char *multipleOf = "multipleOf";
      /* object keywords */
      static constexpr const char *properties = "properties";
      static constexpr const char *required = "required";
      static constexpr const char *dependencies = "dependencies";
      static constexpr const char *additionalProperties = "additionalProperties";
      /* array keywords */
      static constexpr const char *items = "items";
      static constexpr const char *contains = "contains";
      static constexpr const char *k_enum = "enum";
      static constexpr const char *minItems = "minItems";
      static constexpr const char *maxItems = "maxItems";
      static constexpr const char *uniqueItems = "uniqueItems";
      static constexpr const char *additionalItems = "additionalItems";
      /* generic keywords */
      static constexpr const char *title = "title";
      static constexpr const char *description = "description";
      static constexpr const char *k_default = "default";
      static constexpr const char *examples = "examples";
    }
    /* Names from ECMA-404 */

    struct dummy_writer
    {
      bool String(const char *, std::size_t) { return false; }
      bool String(const std::string &) { return false; }
      bool Bool(const bool) { return false; }
      bool StartObject() { return false; }
      bool EndObject() { return false; }
      bool StartArray() { return false; }
      bool EndArray() { return false; }
      bool RawNumber(const char *, std::size_t) { return false; }
    };

    /* This would have been a templated namespace, if C++ had such. See C++ Doc.No.: P1920R0 */
    template <typename Writer>
      struct serializer
      {
        struct value
        {
          value() noexcept {}
          value(value &&) noexcept = default;
          value &operator=(value &&) noexcept = default;
          virtual ~value() {};
          /*
           *  Quick and easy serializaton.
           * TODO: serialize through a rapidjson Writer
           */
          virtual std::string str() const = 0;
          virtual bool serialize(Writer &w) const = 0;
        };

        struct string
          : public value
        {
        private:
          std::string _s;
        public:
          string(std::string s_)
            : _s(s_)
          {}
          string(const char *s_)
            : _s(s_)
          {}
          string(string &&) noexcept = default;
          string &operator=(string &&o_) noexcept
          {
            static_cast<value &>(*this) = std::move(static_cast<value &&>(o_));
            _s = std::move(o_._s);
            return *this;
          } // = default;
          std::string str() const override { return "\"" + _s + "\""; }
          bool serialize(Writer &w_) const override
          {
            return w_.String(_s.c_str());
          }
        };

        struct number
          : public value
        {
        private:
          std::string _i; // sign and digits
          std::string _f; // digits
          std::string _e; // sign and digits
        public:
          number(std::string i_, std::string f_, std::string e_ = "")
            : _i(i_)
            , _f(f_)
            , _e(e_)
          {}

          number(std::uint64_t u_)
            : number(std::to_string(u_), "")
          {}

          number(unsigned u_)
            : number(uint64_t(u_))
          {}

          number(std::int64_t s_)
            : number(std::to_string(s_), "")
          {}

          number(int s_)
            : number(std::int64_t(s_))
          {}

          std::string str() const override //
          {
            return _i + (_f == "" ? "" : ("." + _f) ) + (_e == "" ? "" : ("e" + _e) );
          }

          bool serialize(Writer &w_) const override
          {
            auto ch = str();
            return w_.RawNumber(ch.c_str(), unsigned(ch.size()));
          }
        };

        struct boolean
          : public value
        {
        private:
           bool _b;
        public:
          explicit boolean(bool b_)
            : _b(b_)
          {}
          std::string str() const override { return _b ? "\"true\"" : "\"false\""; }
          bool serialize(Writer &w_) const override { return w_.Bool(_b); }
        };

        struct array;
        struct object;

      /* Note: We could hide member within the object struct, with
       * a bit of change to how objects are constructed.
       */
        struct member
        {
        private:
          string _s;
          std::unique_ptr<value> _v;
          struct incref {};
          template <typename T>
            member(const char *s_, T && v_, incref)
              : member(s_, std::make_unique<T>(std::move(v_)))
            {}
        public:
          member(string &&s_, std::unique_ptr<value> && v_) noexcept
            : _s(std::move(s_))
            , _v(std::move(v_))
          {}

          /* simplification : allow char * as the key */
          member(const char *s_, std::unique_ptr<value> && v_)
            : member(string(s_), std::move(v_))
          {}

          /* simplification : supply the conversion from rvalue to unique_ptr */
          member(const char *s_, array && v_)
            : member(s_, std::move(v_), incref{})
          {}
          member(const char *s_, boolean && v_)
            : member(s_, std::move(v_), incref{})
          {}
          member(const char *s_, number && v_)
            : member(s_, std::move(v_), incref{})
          {}
          member(const char *s_, object && v_)
            : member(s_, std::move(v_), incref{})
          {}

          member(const char *s_, string && v_)
          : member(s_, std::move(v_), incref{})
          {}

          member(member &&) noexcept = default;
          member &operator=(member &&o_) noexcept = default;

          std::string str() const
          {
            return _s.str() + ":" + _v->str();
          }

          bool serialize(Writer &w_) const
          {
            return _s.serialize(w_) && _v->serialize(w_);
          }
        };

        struct object
          : public value
        {
        private:
          std::deque<member> _d;
        public:
          object()
            : _d()
          {}
          template <typename ... Args>
            explicit object(member &&a_, Args&& ... args)
              : object(std::forward<Args>(args)...)
            {
              _d.push_front(std::move(a_));
            }

          object(const object &) = delete;
          object(object &&o_) noexcept
            : value(std::move(o_))
            , _d(std::move(o_._d))
          {} //  = default;
          object &operator=(const object &) = delete;
          object &operator=(object &&) noexcept = default;

          object &append(member &&m_)
          {
            _d.push_back(std::move(m_));
            return *this;
          }

          object &append(object &&b_) //
          {
            _d.insert( //
              _d.end() //
              , std::make_move_iterator(b_._d.begin())
              , std::make_move_iterator(b_._d.end())
            );
            return *this;
          }

          std::string str() const override
          {
            std::string s;
            for ( const auto &m : _d )
            {
              if ( s != "" )
              {
                s += ",";
              }
              s += m.str();
            }
            return "{" + s + "}";
          }

          bool serialize(Writer &w_) const override
          {
            bool ok = w_.StartObject();

            for ( const auto &m : _d )
            {
              ok &= m.serialize(w_);
              if ( ! ok ) break;
            }
            return ok && w_.EndObject();
          }
        };

        struct array
          : public value
        {
        private:
          std::deque<std::unique_ptr<value>> _d;

          struct incref {};
          template <typename T, typename ... Args>
            array(incref, T && v_, Args&& ... args)
              : array(std::make_unique<T>(std::move(v_)), std::forward<Args>(args)...)
            {}

        public:
          array()
            : _d()
          {}

          template <typename ... Args>
            explicit array(std::unique_ptr<value> &&a_, Args&& ... args)
              : array(std::forward<Args>(args)...)
            {
              _d.push_front(std::move(a_));
            }

          /* simplification : supply the conversion from rvalue to unique_ptr */
          template <typename ... Args>
            explicit array(array &&a_, Args&& ... args)
              : array(incref{}, std::move(a_), std::forward<Args>(args)...)
            {}
          template <typename ... Args>
            explicit array(boolean &&a_, Args&& ... args)
              : array(incref{}, std::move(a_), std::forward<Args>(args)...)
            {}
          template <typename ... Args>
            explicit array(number &&a_, Args&& ... args)
              : array(incref{}, std::move(a_), std::forward<Args>(args)...)
            {}
          template <typename ... Args>
            explicit array(object &&a_, Args&& ... args)
              : array(incref{}, std::move(a_), std::forward<Args>(args)...)
            {}
          template <typename ... Args>
            explicit array(string &&a_, Args&& ... args)
              : array(incref{}, std::move(a_), std::forward<Args>(args)...)
            {}

          array &append(array &&a_) //
          {
            _d.insert( //
              _d.end() //
              , std::make_move_iterator(a_._d.begin())
              , std::make_move_iterator(a_._d.end())
            );
            return *this;
          }

          std::string str() const override
          {
            std::string s;
            for ( const auto &e : _d )
            {
              if ( s != "" )
              {
                s += ",";
              }
              s += e->str();
            }
            return "[" + s + "]";
          }

          bool serialize(Writer &w_) const override
          {
            bool ok = w_.StartArray();
            for ( const auto &e : _d )
            {
              ok &= e->serialize(w_);
              if ( ! ok ) break;
            }
            return ok && w_.EndArray();
          }
        };
      };
  }
}

#endif
