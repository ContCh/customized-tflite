#include "common/command_line_parser.h"

#include <ctype.h>

#include <algorithm>
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
    std::cerr << "Usage [" << binary_name << "]\n\n";
    const size_t      USAGE_MAX_COLS       = 59;
    const std::string SEGMENT              = "  ";
    size_t            flag_name_max_length = 0;
    size_t            type_name_max_length = 0;
    common::for_each(flag_list, [&](const Flag& flag) {
        auto print_name      = flag.first_name_ + ", " + flag.alias_;
        flag_name_max_length = std::max(flag_name_max_length, print_name.size());
        type_name_max_length = std::max(type_name_max_length, flag.type_.size());
    });

    auto usage_align_loc = (flag_name_max_length + type_name_max_length + 2 * SEGMENT.size());
    for (const auto& flag : flag_list) {
        auto print_name = flag.first_name_ + ", " + flag.alias_;
        std::cerr << std::left << std::setw(flag_name_max_length) << print_name << SEGMENT;
        std::cerr << std::left << std::setw(type_name_max_length) << flag.type_ << SEGMENT;

        auto   usage_str       = flag.usage_text_;
        size_t usage_print_len = std::min(usage_str.size(), USAGE_MAX_COLS);
        std::cerr << usage_str.substr(0, usage_print_len) << '\n';
        usage_str = usage_str.substr(usage_print_len, usage_str.size() - usage_print_len);
        while (!usage_str.empty()) {
          usage_print_len = std::min(usage_str.size(), USAGE_MAX_COLS);
          std::cerr << std::setw(usage_align_loc) << ' ' << usage_str.substr(0, usage_print_len) << '\n';
          usage_str = usage_str.substr(usage_print_len, usage_str.size() - usage_print_len);
        }
    }
    std::cerr << '\n' << std::setw(usage_align_loc) << "--help " << "Print help message for each option.\n";
    exit(0);
}
