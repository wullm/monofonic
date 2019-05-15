cmake_minimum_required(VERSION 3.9)
set(PRGNAME fastLPT)
project(fastLPT)

# include class submodule
include(${CMAKE_CURRENT_SOURCE_DIR}/external/class.cmake)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -march=native -Wall -fno-omit-frame-pointer -g  -fsanitize=undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -pedantic")
find_package(PkgConfig REQUIRED)

set(CMAKE_MODULE_PATH
        "${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}")


########################################################################################################################
# OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

########################################################################################################################
# mpi support
option(ENABLE_MPI "Enable MPI support" ON)
if(ENABLE_MPI)
  find_package(MPI)
  if(MPI_CXX_FOUND)
    message(STATUS "MPI_CXX_COMPILER = ${MPI_CXX_COMPILER}")
    message(STATUS "MPI_CXX_INCLUDE_PATH = ${MPI_CXX_INCLUDE_PATH}")
    message(STATUS "MPI_CXX_LIBRARIES = ${MPI_CXX_LIBRARIES}")
  endif(MPI_CXX_FOUND)
endif(ENABLE_MPI)


# FFTW
find_package(FFTW3 REQUIRED)

# GSL
find_package(GSL REQUIRED)

# HDF5
find_package(HDF5 REQUIRED)

########################################################################################################################
# INCLUDES
include_directories(${PROJECT_SOURCE_DIR}/include)

# SOURCES
# get all the *.cc files in the subfolders
file( GLOB SOURCES
  ${PROJECT_SOURCE_DIR}/src/*.cc
)

# PLUGINS
# get all the *.cc files in the plugin subfolder
file( GLOB PLUGINS
  ${PROJECT_SOURCE_DIR}/src/plugins/*.cc
)

add_executable(${PRGNAME} ${SOURCES} ${PLUGINS})
target_setup_class(${PRGNAME})

set_target_properties(${PRGNAME} PROPERTIES CXX_STANDARD 14)

# mpi flags
if(MPI_CXX_FOUND)
  if(FFTW_MPI_FOUND)
    target_link_libraries(${PRGNAME} ${FFTW_MPI_LIBRARIES})
    target_compile_options(${PRGNAME} PRIVATE "-DUSE_FFTW_MPI")
  endif(FFTW_MPI_FOUND)

  target_include_directories(${PRGNAME} PRIVATE ${MPI_CXX_INCLUDE_PATH})
  target_compile_options(${PRGNAME} PRIVATE "-DUSE_MPI")
  target_link_libraries(${PRGNAME} ${MPI_LIBRARIES})
endif(MPI_CXX_FOUND)

if(FFTW_THREADS_FOUND) 
  target_link_libraries(${PRGNAME} ${FFTW_THREADS_LIBRARIES})
  target_compile_options(${PRGNAME} PRIVATE "-DUSE_FFTW_THREADS")
endif(FFTW_THREADS_FOUND)

target_link_libraries(${PRGNAME} ${FFTW_LIBRARIES})
target_include_directories(${PRGNAME} PRIVATE ${FFTW_INCLUDE_DIR})

target_link_libraries(${PRGNAME} ${GSL_LIBRARIES})
target_include_directories(${PRGNAME} PRIVATE ${GSL_INCLUDE_DIR})

target_link_libraries(${PRGNAME} ${HDF5_LIBRARIES})
target_include_directories(${PRGNAME} PRIVATE ${HDF5_INCLUDE_DIR})
