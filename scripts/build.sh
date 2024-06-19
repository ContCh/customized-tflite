#!/bin/bash

# >> Global Parameter controling the build phase <<
BUILD_DIRNAME=build
CLEAN_BUILD=false
WITH_TEST=false
COMPILE_MODE=Release

# >> Function definition in script <<
function print_help() {
    echo "(Usage) Build script for repo"
    echo ""
    echo "  --debug,-d    Debug mode, default is Release."
    echo "  --test,-t     Compile with gtest and test cases."
    echo "  --clean,-c    Clean cache and intermediate files will be deleted before build."
    echo "  --help,-h     Help information."
    echo ""
}

function build_main() {
    if [[ ${CLEAN_BUILD} = "true" && -e ${BUILD_DIRNAME} ]]; then
         rm -rf ${BUILD_DIRNAME}
    fi
    if [ ! -e ${BUILD_DIRNAME} ]; then
         mkdir ${BUILD_DIRNAME}
    fi
    CMAKE_COMMAND="cmake -DCMAKE_BUILD_TYPE=${COMPILE_MODE} "
    MAKE_COMMAND="make -j 32"
    if [ ${WITH_TEST} = "true" ]; then
        CMAKE_COMMAND="${CMAKE_COMMAND} -DENABLE_UNIT_TEST=ON "
    fi
    echo "${CMAKE_COMMAND}"
    cd ${BUILD_DIRNAME}
    ${CMAKE_COMMAND} ..
    ${MAKE_COMMAND}
}

# Main
Arg_number=$#
Args=($@)
for (( idx=0 ; idx<$# ; idx++ ))
do
    case ${Args[idx]} in
        "--debug" | "-d" )
            COMPILE_MODE=Debug
            ;;
        "--test" | "-t" )
            WITH_TEST=true
            ;;
        "--clean" | "-c" )
            CLEAN_BUILD=true
            ;;
        "--help" | "-h" )
            print_help
            exit 0
            ;;
        * )
            print_help
            exit 1
            ;;
    esac
done

build_main
