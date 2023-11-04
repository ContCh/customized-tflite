#include <iostream>

#include "common/logging.h"
#include "common/id_generator.h"

int main() {
    LOG(INFO) << GenUniqueID();
    SET_MINIMUM_LOG_LEVEL(1);
    LOG(INFO) << "This should not be printed";
    LOG(TEMP) << "This should be printed\n";
    LOG(FATAL) << GenUniqueID() << " break here.";

    return 0;
}
