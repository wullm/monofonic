cmake_minimum_required(VERSION 3.11)
include(FetchContent)
FetchContent_Declare(
    genericio
    GIT_REPOSITORY https://xgitlab.cels.anl.gov/mbuehlmann/genericio.git
    GIT_TAG master
)

FetchContent_GetProperties(genericio)
if(NOT genericio_POPULATED)
    FetchContent_Populate(genericio)
    add_subdirectory(${genericio_SOURCE_DIR} ${genericio_BINARY_DIR})
endif()
