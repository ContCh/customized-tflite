# Graph Cutter Tool
file(GLOB_RECURSE GRAPH_CUTTER_SRC_FILES "graph_cutter/*cpp")
add_executable(graph_cutter ${GRAPH_CUTTER_SRC_FILES})
target_link_libraries(graph_cutter common_library parse_and_serialize model_representation)
