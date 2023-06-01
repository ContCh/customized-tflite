project(CUSTOM_TFLITE)

cmake_minimum_required(VERSION 3.15)
add_compile_options(-Wall -Werror -O0)
set(CMAKE_CXX_STANDARD 17)

set(SCHEMA_DIRECTORY ${PROJECT_SOURCE_DIR}/schema)

# Third-party libraries compile
add_subdirectory(third_party/flatbuffers)
