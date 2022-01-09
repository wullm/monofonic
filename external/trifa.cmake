cmake_minimum_required(VERSION 3.11)
include(FetchContent)
FetchContent_Declare(
    3fa
    GIT_REPOSITORY https://github.com/wullm/trifa.git
    GIT_TAG main
    GIT_SHALLOW YES
    GIT_PROGRESS TRUE
)

FetchContent_GetProperties(3fa)
if(NOT 3fa_POPULATED)
    set(FETCHCONTENT_QUIET OFF)
    FetchContent_Populate(3fa)
    add_subdirectory(${3fa_SOURCE_DIR} ${3fa_BINARY_DIR})
endif()
