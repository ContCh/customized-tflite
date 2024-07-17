# common library
add_subdirectory(common)
# model/graph intermediate presentation
add_subdirectory(model)
# parse and serialize
add_subdirectory(parser_and_serializer)

add_executable(main main.cpp)
target_link_libraries(main common_library model_representation parse_and_serialize)
