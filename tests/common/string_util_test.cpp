#include "common/string_utils.h"
#include "googletest/include/gtest/gtest.h"

TEST(STRING_UTIL_TEST, JoinTestTemporaryString) {
    std::vector<std::string> strs            = {"Please", "don't", "make", "joke"};
    std::string              expected_result = "Please  don't  make  joke";
    EXPECT_EQ(common::join(strs, "  "), expected_result);
}

TEST(STRING_UTIL_TEST, JoinTestConstString) {
    char                     connect[]       = "//";
    std::vector<std::string> strs            = {"/home", "user", "workspace"};
    std::string              expected_result = "/home//user//workspace";
    EXPECT_EQ(common::join(strs, connect), expected_result);
}

TEST(STRING_UTIL_TEST, JoinTestChar) {
    std::vector<std::string> strs            = {"Please", "don't", "make", "joke"};
    std::string              expected_result = "Please don't make joke";
    EXPECT_EQ(common::join(strs, ' '), expected_result);
}
