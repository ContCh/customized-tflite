#include "common/command_line_parser.h"

#include <ctype.h>

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>

void CommandLineParser::Parse(int argc, char** argv, const std::vector<Flag>& flag_list) {
    /// Prepare
    for (int32_t index = 1; index < argc; index++) {
        if (argv[index] == std::string {"--help"}) {
            PrintUsage(argv[0], flag_list);
        }
    }
    std::unordered_map<std::string, int32_t> flag_name_to_index;
    int32_t                                  flag_index = 0;
    for (const auto& flag : flag_list) {
        REPORT_ERROR_IF(common::contains(flag_name_to_index, flag.first_name_),
                        "Duplicate definition in option flags for ", flag.first_name_);
        flag_name_to_index[flag.first_name_] = flag_index;
        if (!flag.alias_.empty() && flag.first_name_ != flag.alias_) {
            REPORT_ERROR_IF(common::contains(flag_name_to_index, flag.alias_),
                            "Duplicate definition in option flags for ", flag.alias_);
            flag_name_to_index[flag.alias_] = flag_index;
        }
        ++flag_index;
    }

    /// Parse command line (main)
    std::unordered_map<std::string, std::string> option_to_value;
    int32_t                                      arg_index = 1;
    std::vector<uint32_t>                        parsed_flag_index;

    while (arg_index < argc) {
        try {
            REPORT_ERROR_IF(!common::contains(flag_name_to_index, argv[arg_index]),
                            "Unknown option flag : ", argv[arg_index]);
            parsed_flag_index.push_back(flag_name_to_index.at(argv[arg_index]));
            const auto& flag = flag_list.at(flag_name_to_index.at(argv[arg_index++]));
            std::string arg_value;
            if (arg_index < argc && !common::contains(flag_name_to_index, argv[arg_index])) {
                arg_value = argv[arg_index++];
            } else {
                REPORT_ERROR_IF(flag.type_ != TypeName<bool>, "Invalid input for flag : ", flag.first_name_);
                arg_value = "true";
            }
            flag.parse_func_(arg_value);
        } catch (...) {
            // TODO: LOG
            PrintUsage(argv[0], flag_list);
        }
    }
    /// Post process and necessary check
    // CHECK if all required option has been assigned by command line.
    flag_index = 0;
    for (const auto& flag : flag_list) {
        if (flag.is_required_ == REQUIRED::YES && !common::contains(parsed_flag_index, flag_index)) {
            std::cerr << "ERROR: " << flag.first_name_ << " is required while it is not set.\n";
            PrintUsage(argv[0], flag_list);
        }
        ++flag_index;
    }
}

void CommandLineParser::PrintUsage(const std::string& binary_name, const std::vector<Flag>& flag_list) {
    std::filesystem::path binary_path(binary_name);
    std::cerr << "Usage [" << binary_path.filename().string() << "]:\n\n";
    const size_t      usage_max_cols       = 60;
    const size_t      left_trace_limit     = 8;
    const std::string item_segment         = "  ";
    size_t            flag_name_max_length = 0;
    size_t            type_name_max_length = 0;
    common::for_each(flag_list, [&](const Flag& flag) {
        auto print_name      = flag.first_name_ + (flag.alias_.empty() ? item_segment : (", " + flag.alias_));
        flag_name_max_length = std::max(flag_name_max_length, print_name.size());
        type_name_max_length = std::max(type_name_max_length, flag.type_.size());
    });

    auto usage_align_loc = (flag_name_max_length + type_name_max_length + 2 * item_segment.size());
    for (const auto& flag : flag_list) {
        auto print_name = flag.first_name_ + (flag.alias_.empty() ? item_segment : (", " + flag.alias_));
        std::cerr << std::left << std::setw(flag_name_max_length) << print_name << item_segment;
        std::cerr << std::left << std::setw(type_name_max_length) << flag.type_ << item_segment;

        std::string_view usage_str(flag.usage_text_);
        bool             first_usage_line = true;
        int32_t          index            = 0;
        while (index < static_cast<int32_t>(usage_str.size())) {
            int32_t end_index     = std::min(usage_str.size() - 1, index + usage_max_cols - 1);
            int32_t newline_index = end_index;
            while (newline_index >= index && usage_str[newline_index] != '\n') {
                newline_index--;
            }
            if (newline_index >= index) {
                end_index = newline_index;
            }

            // 1. \n means mandatory newline
            // 2. If the signature in next location is space, then print util next location
            // 3. If not the cases above, trace back util meet a space, it is helpful for printing a complete word
            if (usage_str[end_index] != '\n' && end_index + 1 < static_cast<int32_t>(usage_str.size()) &&
                isspace(usage_str[end_index + 1])) {
                end_index = end_index + 1;
            } else if (end_index + 1 != static_cast<int32_t>(usage_str.size())) {
                size_t traceback = 0;
                while (traceback < left_trace_limit && !isspace(usage_str[end_index - traceback])) {
                    traceback++;
                }
                end_index = traceback >= left_trace_limit ? end_index : end_index - traceback;
            }
            auto substr_to_print = usage_str.substr(index, end_index - index + 1);
            if (first_usage_line) {
                std::cerr << substr_to_print << (usage_str[end_index] != '\n' ? '\n' : '\0');
                first_usage_line = false;
            } else {
                std::cerr << std::setw(usage_align_loc) << ' ' << substr_to_print
                          << (usage_str[end_index] != '\n' ? '\n' : '\0');
            }
            index = end_index + 1;
        }
    }
    std::cerr << '\n' << std::setw(usage_align_loc) << "--help " << "Print help message for each option.\n";
    exit(0);
}
