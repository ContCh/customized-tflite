add_executable(main_test ./test.cpp)
target_link_libraries(main_test common_library)

add_executable(clp_test ./command_line_parser_test.cpp)
target_link_libraries(clp_test common_library)
