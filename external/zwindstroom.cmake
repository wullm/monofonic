cmake_minimum_required(VERSION 3.11)
include(FetchContent)
FetchContent_Declare(
    zwindstroom
    GIT_REPOSITORY https://github.com/wullm/zwindstroom.git
    GIT_TAG MG
    GIT_SHALLOW YES
    GIT_PROGRESS TRUE
)

FetchContent_GetProperties(zwindstroom)
if(NOT zwindstroom_POPULATED)
    set(FETCHCONTENT_QUIET OFF)
    FetchContent_Populate(zwindstroom)
    add_subdirectory(${zwindstroom_SOURCE_DIR} ${zwindstroom_BINARY_DIR})
endif()
