#pragma once

#include <functional>
#include <string>

#include "common/logging.h"
#include "common/stl_wrapper.h"

enum class REQUIRED { YES, NO };

template <typename T> constexpr const char* TypeName              = "UNKNOWN_TYPE";
template <> constexpr const char*           TypeName<std::string> = "STRING";
template <> constexpr const char*           TypeName<bool>        = "BOOL";
template <> constexpr const char*           TypeName<uint64_t>    = "UINT64";
template <> constexpr const char*           TypeName<int64_t>     = "INT64";
template <> constexpr const char*           TypeName<int32_t>     = "INT32";
template <> constexpr const char*           TypeName<uint32_t>    = "UINT32";
template <> constexpr const char*           TypeName<float>       = "FLOAT";
template <> constexpr const char*           TypeName<double>      = "DOUBLE";

template <typename OpT> class Option {
 public:
    Option() = default;

    Option(OpT value) {
        value_     = value;
        has_value_ = true;
    }

    Option(const char* value) { ParseValue(value); }

    bool HasValue() const { return has_value_; }
    OpT  GetValue() const {
        REPORT_ERROR_IF(!has_value_, "Option is not set.");
        return value_;
    }

    void ParseValue(std::string value) {
        has_value_ = true;
        ParseValueImpl(value);
    }

    Option(const Option&)            = delete;
    Option(Option&&)                 = delete;
    Option& operator=(const Option&) = delete;
    Option& operator=(Option&&)      = delete;

 private:
    bool has_value_ = false;
    OpT  value_;

    void ParseValueImpl(const std::string& value) { value_ = value; }
};

template <> void Option<float>::ParseValueImpl(const std::string& value) { value_ = std::stof(value); }
template <> void Option<double>::ParseValueImpl(const std::string& value) { value_ = std::stod(value); }

template <> void Option<int32_t>::ParseValueImpl(const std::string& value) {
    auto int_value = std::stoll(value);
    value_         = static_cast<int32_t>(int_value);
}

template <> void Option<int64_t>::ParseValueImpl(const std::string& value) {
    auto int_value = std::stoll(value);
    value_         = static_cast<int32_t>(int_value);
}

template <> void Option<uint32_t>::ParseValueImpl(const std::string& value) {
    auto int_value = std::stoll(value);
    value_         = static_cast<uint32_t>(int_value);
}

template <> void Option<uint64_t>::ParseValueImpl(const std::string& value) {
    auto int_value = std::stoull(value);
    value_         = static_cast<uint64_t>(int_value);
}

template <> void Option<bool>::ParseValueImpl(const std::string& value) {
    LOG(INFO) << "bool";
    static const std::vector<std::string> false_flags = {"false", "0", "off", "n", "no"};
    static const std::vector<std::string> true_flags  = {"true", "1", "on", "y", "yes"};

    std::string value_info = value;
    common::for_each(value_info, ::tolower);
    if (common::contains(false_flags, value_info)) {
        value_ = false;
    } else if (common::contains(true_flags, value_info)) {
        value_ = true;
    } else {
        // Set default value - true, if the given string is too fuzzy to deduce.
        value_ = true;
    }
}

class Flag {
 public:
    template <typename T>
    Flag(const char* name, Option<T>& opt, REQUIRED required, const std::string& usage_text)
        : is_required_(required), first_name_(name), usage_text_(usage_text) {
        parse_func_ = std::bind(&Option<T>::ParseValue, &opt, std::placeholders::_1);
        type_       = TypeName<T>;
    }

    template <typename T>
    Flag(const char* name, const char* alias, Option<T>& opt, REQUIRED required, const std::string& usage_text)
        : is_required_(required), first_name_(name), alias_(alias), usage_text_(usage_text) {
        parse_func_ = std::bind(&Option<T>::ParseValue, &opt, std::placeholders::_1);
        type_       = TypeName<T>;
    }

    friend class CommandLineParser;

 private:
    std::function<void(std::string)> parse_func_;
    REQUIRED                         is_required_;

    std::string first_name_;
    std::string alias_;
    std::string type_;
    std::string usage_text_;
};

class CommandLineParser {
 public:
    // Parse the command line, find target variables associated to Flag, and update the variables.
    // "-h" '--help" is help Flag to print usage, which is unique and added to flag_list automatically.
    // Don't bother to add help Flag.
    static void Parse(int argc, char** argv, const std::vector<Flag>& flag_list);

    // Print a usage message for given flag_list.
    static void PrintUsage(const std::string& binary_name, const std::vector<Flag>& flag_list);
};
