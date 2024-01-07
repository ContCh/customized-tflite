#include <string.h>

#include <string>
#include <vector>

namespace common {

std::string join(const std::vector<std::string>& strs, char connector = ' ');

std::vector<std::string> split(std::string str, char delimiter = ' ');

std::string toupper(const std::string& str);

std::string tolower(const std::string& str);

std::string strip(const std::string& str);

}  // namespace common
