#include "common/command_line_parser.h"

#include "common/logging.h"

struct ArgCs {
    Option<std::string> text = "val";
    Option<int32_t>     val;
    Option<bool>        trigger;
};

int main(int argc, char** argv) {
    ArgCs             command_args;
    std::vector<Flag> flags = {
        Flag("--text", command_args.text, REQUIRED::NO, "It is a test text, you can type what you want [Required] "
                                                         "it is a loooooooooooooooooooooong str, which tests xxx"),
        Flag("--input", "-i", command_args.val, REQUIRED::YES, "It is an input."),
        Flag("--trigger", "-t", command_args.trigger, REQUIRED::NO, "It is Nothing"),
    };
    CommandLineParser::Parse(argc, argv, flags);
    LOG(INFO) << command_args.text.GetValue() << "  (" << command_args.val.GetValue() << ").\n";

    return 0;
}
