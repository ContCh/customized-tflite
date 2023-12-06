project(CUSTOM_TFLITE)

cmake_minimum_required(VERSION 3.15)
add_compile_options(-Wall -Werror -O0)
set(CMAKE_CXX_STANDARD 17)

set(SCHEMA_DIRECTORY ${PROJECT_SOURCE_DIR}/schema)

set(THIRD_PARTY_INCLUDE)
# Third-party libraries compile
include(third_party/third_party.cmake)

include_directories(include)
include_directories(schema)

add_subdirectory(source)
add_subdirectory(tests)
