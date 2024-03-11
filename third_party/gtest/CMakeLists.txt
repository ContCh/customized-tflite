# version 1.14


if(TARGET googletest OR googletest_POPULATED)
    return()
endif()

include(FetchContent)

FetchContent_Declare(googletest
    GIT_REPOSITORY  https://github.com/google/googletest
    GIT_TAG         f8d7d77c06936315286eb55f8de22cd23c188571
    SOURCE_DIR      ${CMAKE_CURRENT_BINARY_DIR}/source
    BINARY_DIR      ${CMAKE_CURRENT_BINARY_DIR}/binary
)

FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory("${googletest_SOURCE_DIR}" "${googletest_BINARY_DIR}")
endif()

set(googletest_INCLUDE_DIR ${googletest_SOURCE_DIR} PARENT_SCOPE)
