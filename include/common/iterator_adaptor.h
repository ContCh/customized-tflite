#pragma once

#include <algorithm>
#include <type_traits>

template <typename T> class Range {
 public:
    typedef T iterator;

    Range(const iterator& begin, const iterator& end) : begin_(begin), end_(end) {}

    iterator begin() const { return begin_; }

    iterator end() const { return end_; }

    size_t size() const { return std::distance(begin_.base(), end_.base()); }

    bool empty() const { return begin_ == end_; }

 private:
    iterator begin_, end_;
};

template <typename Derived, typename Base, typename Value> class IteratorAdaptor {
 public:
    IteratorAdaptor(const Base& iter) : m_iterator_(iter) {}

    Derived& operator++() {
        ++m_iterator_;
        return AsDerived();
    }
    Derived operator++(int) {
        auto copy = AsDerived();
        ++m_iterator_;
        return copy;
    }

    bool operator==(const IteratorAdaptor& it) { return m_iterator_ == it.m_iterator_; }
    bool operator!=(const IteratorAdaptor& it) { return m_iterator_ != it.m_iterator_; }

    Value operator*() { return AsDerived().dereference(); }

    const Base& base() const { return m_iterator_; }

    virtual ~IteratorAdaptor() = default;

 private:
    Base m_iterator_;

    Derived& AsDerived() {
        static_assert(std::is_base_of_v<IteratorAdaptor, Derived>, "CRTP requires class derives from IteratorAdaptor.");
        return *static_cast<Derived*>(this);
    }
};
