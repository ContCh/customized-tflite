# common library
add_subdirectory(common)
# model/graph intermediate presentation
add_subdirectory(model)
# parse and serialize
add_subdirectory(parser_and_serializer)
add_subdirectory(tools)

add_executable(main main.cpp)
target_link_libraries(main common_library model_representation parse_and_serialize)


include(GNUInstallDirs)
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)
set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib)
set(CMAKE_INSTALL_BINDIR ${CMAKE_INSTALL_PREFIX}/bin)
message(STATUS "LIBDIR: ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "BINDIR: ${CMAKE_INSTALL_BINDIR}")
install(TARGETS common_library model_representation parse_and_serialize
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS graph_cutter
        LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR})
