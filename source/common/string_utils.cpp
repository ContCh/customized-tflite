#include "common/string_utils.h"

#include <algorithm>
#include <unordered_set>

namespace common {
std::string join(const std::vector<std::string>& strs, char connector) { return join(strs, std::string(1, connector)); }

std::string join(const std::vector<std::string>& strs, std::string_view connector) {
    std::string output;
    for (auto str_it = strs.begin(); str_it != strs.end(); str_it++) {
        if (str_it != strs.begin()) {
            output += connector;
        }
        output += *str_it;
    }
    return output;
}

std::vector<std::string> split(std::string str, char delimiter) { return split(str, std::string(1, delimiter)); }

std::vector<std::string> split(std::string str, std::string_view delimiter) {
    std::vector<std::string> output;
    size_t                   location = str.find_first_of(delimiter);
    while (location != std::string::npos) {
        output.push_back(str.substr(0, location));
        str      = str.substr(location + 1, str.size() - location - 1);
        location = str.find_first_of(delimiter);
    }
    output.push_back(str);
    return output;
}

std::string toupper(const std::string& str) {
    std::string upper_str;
    upper_str.resize(str.size());
    std::transform(str.begin(), str.end(), upper_str.begin(), ::toupper);
    return upper_str;
}

std::string tolower(const std::string& str) {
    std::string lower_str;
    lower_str.resize(str.size());
    std::transform(str.begin(), str.end(), lower_str.begin(), ::tolower);
    return lower_str;
}

std::string lstrip(const std::string& str) {
    static const std::unordered_set<char> blanks   = {' ', '\r', '\n', '\t', '\v', '\f'};
    int32_t                               index    = 0;
    int32_t                               str_size = str.size();
    while (index < str_size && blanks.find(str.at(index)) != blanks.end()) {
        ++index;
    }
    return index == str_size ? std::string {} : str.substr(index, str.size() - index);
}

std::string rstrip(const std::string& str) {
    static const std::unordered_set<char> blanks = {' ', '\r', '\n', '\t', '\v', '\f'};
    int32_t                               index  = str.size() - 1;
    while (index >= 0 && blanks.find(str.at(index)) != blanks.end()) {
        --index;
    }
    return index < 0 ? std::string {} : str.substr(0, index + 1);
}

std::string strip(const std::string& str) { return lstrip(rstrip(str)); }

}  // namespace common
