file(GLOB_RECURSE COMMON_SRC_FILES "./*cpp")
add_library(common_library SHARED ${COMMON_SRC_FILES})