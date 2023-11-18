#pragma once

#include <algorithm>

namespace common {

template <typename _Tp, typename _Func>
inline auto all_of(_Tp&& container, _Func&& unary_func) {
    std::all_of(std::begin(container), std::end(container), std::forward<_Func>(unary_func));
}

template <typename _Tp, typename _Func>
inline auto any_of(_Tp&& container, _Func&& unary_func) {
    std::any_of(std::begin(container), std::end(container), std::forward<_Func>(unary_func));
}

template <typename _Tp, typename _Func>
inline auto for_each(_Tp&& container, _Func&& unary_func) {
    std::for_each(std::begin(container), std::end(container), std::forward<_Func>(unary_func));
}

template <typename _Tp>
inline auto find(_Tp&& range, _Tp elem) {
    return std::find(std::begin(range), std::end(range), elem);
}

template <typename _Tp, typename _Func>
inline auto find_if(_Tp&& range, _Func unary_func) {
    return std::find_if(std::begin(range), std::end(range), std::forward<_Func>(unary_func));
}

template <typename _Tp>
inline auto get_first_index(const std::vector<_Tp>& range, _Tp elem) {
    return std::distance(std::begin(range), std::find(std::begin(range), std::end(range), elem));
}

template <typename... _Tp>
using Void_t = void;

template <typename _Tp, typename _Up, typename = void>
struct has_find_method : public std::false_type {};

template <typename _Tp, typename _Up>
struct has_find_method<_Tp, _Up, Void_t<decltype(std::declval<_Tp>().find(std::declval<_Up>()))>>
    : public std::true_type {};

template <typename _Tp, typename _Up>
bool contains(_Tp&& container, _Up&& elem) {
    if constexpr (has_find_method<_Tp, _Up>::value) {
        return container.find(elem) != container.end();
    } else {
        return common::find(std::forward<_Tp>(container), elem) != std::end(container);
    }
}

} // namespace common