cmake_minimum_required(VERSION 3.15)
project(CUSTOM_TFLITE)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(ENABLE_UNIT_TEST "Enable unit test verifying base functions" OFF)

string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE)
if (BUILD_TYPE STREQUAL "DEBUG")
    add_compile_options(-O0 -g)
else()
    add_compile_options(-O3 -DNDEBUG)
endif()
add_compile_options(-Wall -Werror)

set(SCHEMA_DIRECTORY ${PROJECT_SOURCE_DIR}/schema)

set(THIRD_PARTY_INCLUDE)
# Third-party libraries compile
include(third_party/third_party.cmake)

include_directories(include)
include_directories(schema)

add_subdirectory(source)
add_subdirectory(tests)
