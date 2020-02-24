cmake_minimum_required(VERSION 3.9)
set(PRGNAME monofonIC)
project(monofonIC)

# include class submodule
include(${CMAKE_CURRENT_SOURCE_DIR}/external/class.cmake)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -fno-omit-frame-pointer -g  -fsanitize=address")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -pedantic")
find_package(PkgConfig REQUIRED)

set(CMAKE_MODULE_PATH
        "${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}")


########################################################################################################################
# Add a custom command that produces version.cc, plus
# a dummy output that's not actually produced, in order
# to force version.cmake to always be re-run before the build
ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/version.cc
           ${CMAKE_CURRENT_BINARY_DIR}/_version.cc
    COMMAND ${CMAKE_COMMAND} -P
            ${CMAKE_CURRENT_SOURCE_DIR}/version.cmake)

########################################################################################################################
# OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

########################################################################################################################
# Pthreads
find_package(Threads REQUIRED)

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


########################################################################################################################
# FFTW
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()
if(ENABLE_MPI)
  find_package(FFTW3 COMPONENTS SINGLE DOUBLE OPENMP THREADS MPI)
else()
  find_package(FFTW3 COMPONENTS SINGLE DOUBLE OPENMP THREADS)
endif(ENABLE_MPI)

########################################################################################################################
# GSL
find_package(GSL REQUIRED)

########################################################################################################################
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
# add the auto generated version file
list (APPEND SOURCES "${CMAKE_CURRENT_BINARY_DIR}/version.cc")

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
  if(FFTW3_DOUBLE_MPI_FOUND)
    target_link_libraries(${PRGNAME} ${FFTW3_DOUBLE_MPI_LIBRARY})
    target_include_directories(${PRGNAME} PRIVATE ${FFTW3_INCLUDE_DIR_PARALLEL})
    target_compile_options(${PRGNAME} PRIVATE "-DUSE_FFTW_MPI")
  endif(FFTW3_DOUBLE_MPI_FOUND)

  target_include_directories(${PRGNAME} PRIVATE ${MPI_CXX_INCLUDE_PATH})
  target_compile_options(${PRGNAME} PRIVATE "-DUSE_MPI")
  target_link_libraries(${PRGNAME} ${MPI_LIBRARIES})
endif(MPI_CXX_FOUND)

if(FFTW3_DOUBLE_THREADS_FOUND) 
  target_link_libraries(${PRGNAME} ${FFTW3_DOUBLE_THREADS_LIBRARY})
  target_compile_options(${PRGNAME} PRIVATE "-DUSE_FFTW_THREADS")
endif(FFTW3_DOUBLE_THREADS_FOUND)

if(HDF5_FOUND)
  # target_link_libraries(${PRGNAME} ${HDF5_C_LIBRARY_DIRS})
  target_link_libraries(${PRGNAME} ${HDF5_LIBRARIES})
  target_include_directories(${PRGNAME} PRIVATE ${HDF5_INCLUDE_DIRS})
  target_compile_options(${PRGNAME} PRIVATE "-DUSE_HDF5")
endif(HDF5_FOUND)

target_link_libraries(${PRGNAME} ${FFTW3_LIBRARIES})
target_include_directories(${PRGNAME} PRIVATE ${FFTW3_INCLUDE_DIRS})

target_link_libraries(${PRGNAME} ${GSL_LIBRARIES})
target_include_directories(${PRGNAME} PRIVATE ${GSL_INCLUDE_DIR})

target_link_libraries(${PRGNAME} ${HDF5_LIBRARIES})
target_include_directories(${PRGNAME} PRIVATE ${HDF5_INCLUDE_DIR})
