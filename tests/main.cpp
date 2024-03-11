#include <iostream>

#include "common/logging.h"
#include "googletest/include/gtest/gtest.h"
#include "googlemock/include/gmock/gmock.h"

int main(int argc, char** argv) {
    LOG(INFO) << "Start Unit Test!";
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
