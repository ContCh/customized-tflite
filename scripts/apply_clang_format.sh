#!/bin/bash

# Initialize the necessary parameters
CLANG_FORMAT_BIN=/usr/bin/clang-format
WORK_SCOPE="include source tests tools"
WORK_DIRECTORY=$( realpath "$( dirname ${BASH_SOURCE[0]} )"/.. )

# Check clang-format version
CLANG_FORMAT_VERSION=`${CLANG_FORMAT_BIN} --version | grep "clang-format version .\{2,\}"`
if [ ${#CLANG_FORMAT_VERSION} -eq 0 ]; then
    echo "ERROR: clang-format binary is invalid"
    exit 1
fi
CLANG_FORMAT_VERSION=${CLANG_FORMAT_VERSION#*clang-format version}
CLANG_FORMAT_VERSION=${CLANG_FORMAT_VERSION:1:2}
REQUIRED_VERSION=18
if [ "${CLANG_FORMAT_VERSION}" -lt "${REQUIRED_VERSION}" ]; then
    echo "Required clang-format minimum version ${REQUIRED_VERSION}, current version is ${CLANG_FORMAT_VERSION}"
    exit 1
fi

# Run clang-format and modify each file written in cpp file
cd ${WORK_DIRECTORY}
TARGET_FILES=$( find ${WORK_SCOPE} -iname *.h -o -iname *.hpp -o -iname *.cpp -o -iname *.cc -o -iname *.cu )

NUM_OF_PROCESSED_FILES=0
for TARGET_FILE in ${TARGET_FILES[@]};
do
    ${CLANG_FORMAT_BIN} -i ${TARGET_FILE}
    ((NUM_OF_PROCESSED_FILES++))
done
echo "Apply clang-format successfully to ${NUM_OF_PROCESSED_FILES} files."

