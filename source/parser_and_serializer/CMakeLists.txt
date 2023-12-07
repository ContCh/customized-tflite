file(GLOB_RECURSE IR_PORT_SRC_FILES "./*.cpp")
add_library(parse_and_serialize STATIC ${IR_PORT_SRC_FILES})
target_link_libraries(parse_and_serialize PUBLIC common_library model_representation)
target_include_directories(parse_and_serialize PUBLIC
                           ${PROJECT_SOURCE_DIR}/schema
                           ${PROJECT_BINARY_DIR}/third_party/flatbuffers/source/include)
add_dependencies(parse_and_serialize generated_schema_ir)