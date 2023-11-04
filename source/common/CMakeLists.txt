file(GLOB_RECURSE COMMON_SRC_FILES "./*cpp")
add_library(common_library STATIC ${COMMON_SRC_FILES})