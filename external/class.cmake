cmake_minimum_required(VERSION 3.11)
include(FetchContent)
FetchContent_Declare(
    class
    GIT_REPOSITORY https://github.com/michaelbuehlmann/class_public.git
    GIT_TAG master
)

FetchContent_GetProperties(class)
if(NOT class_POPULATED)
    FetchContent_Populate(class)
    add_subdirectory(${class_SOURCE_DIR} ${class_BINARY_DIR})
endif()