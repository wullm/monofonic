cmake_minimum_required(VERSION 3.11)
include(FetchContent)
FetchContent_Declare(
    fastdf
    GIT_REPOSITORY https://github.com/wullm/fastdf.git
    GIT_TAG master
    GIT_SHALLOW YES
    GIT_PROGRESS TRUE
)

FetchContent_GetProperties(fastdf)
if(NOT fastdf_POPULATED)
    set(FETCHCONTENT_QUIET OFF)
    FetchContent_Populate(fastdf)
    add_subdirectory(${fastdf_SOURCE_DIR} ${fastdf_BINARY_DIR})
    set(WITH_CLASS 0)
endif()
