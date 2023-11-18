# version 23.1.21

include(ExternalProject)

ExternalProject_Add(flatbuffers
    GIT_REPOSITORY  https://github.com/google/flatbuffers
    GIT_TAG         ee848a02e17a94edaacd1dd95a1664b59c6f06b2
    SOURCE_DIR      ${CMAKE_CURRENT_BINARY_DIR}/source
    BINARY_DIR      ${CMAKE_CURRENT_BINARY_DIR}/binary
    CMAKE_ARGS      -G "Unix Makefiles"
    INSTALL_COMMAND ""
)

add_custom_target(generated_schema_ir ALL
    COMMAND           ${CMAKE_CURRENT_BINARY_DIR}/binary/flatc --cpp schema.fbs
    WORKING_DIRECTORY ${SCHEMA_DIRECTORY}
    DEPENDS           flatbuffers
)

set(FLATBUFFERS_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/source/include PARENT_SCOPE)

