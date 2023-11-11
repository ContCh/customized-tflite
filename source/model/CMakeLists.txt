file(GLOB_RECURSE MODEL_SRC_FILES "./*cpp")
add_library(model_representation STATIC ${MODEL_SRC_FILES})
target_link_libraries(model_representation PUBLIC common_library)
