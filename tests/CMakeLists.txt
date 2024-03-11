# unit test based on googletest
if (ENABLE_UNIT_TEST)
    file(GLOB_RECURSE ALL_TESTS_TARGET common/*cpp)
    add_executable(test_suite_entry main.cpp ${ALL_TESTS_TARGET})
    set(TEST_LIBS common_library)
    target_link_libraries(test_suite_entry PUBLIC gtest gmock ${TEST_LIBS})
    target_include_directories(test_suite_entry PRIVATE ${googletest_INCLUDE_DIR})
endif()
