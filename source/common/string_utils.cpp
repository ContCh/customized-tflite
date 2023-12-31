#include "common/string_utils.h"

#include <algorithm>

namespace common {

std::string join(const std::vector <std::string> &strs, char connector) {
  std::string output;
  for (auto str_it = strs.begin(); str_it != strs.end(); str_it++) {
    if (str_it != strs.begin()) {
      output += connector;
    }
    output += *str_it;
  }
  return output;
}

std::vector <std::string> split(std::string str, char delimiter) {
  std::vector <std::string> output;
  size_t location = str.find_first_of(delimiter);
  while (location != std::string::npos) {
    output.push_back(str.substr(0, location));
    str = str.substr(location + 1, str.size() - location - 1);
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

std::string strip(const std::string& str) {
  // TODO: implement similar to python
  return std::string{};
}

} // namespace common
