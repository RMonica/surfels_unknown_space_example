cmake_minimum_required(VERSION 2.8.3)
project(surfels_unknown_space_lib)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 11)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCL REQUIRED)

###########
## Build ##
###########

add_definitions(-DCL_HPP_MINIMUM_OPENCL_VERSION=110)
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=120)

add_definitions(-Wno-ignored-attributes)

include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${OpenCL_INCLUDE_DIRS}
)

add_custom_command(
  COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_LIST_DIR}/cmake/surfels_unknown_space_cl.cmake"
  MAIN_DEPENDENCY "${CMAKE_CURRENT_LIST_DIR}/src/surfels_unknown_space.cl"
  OUTPUT "${CMAKE_CURRENT_LIST_DIR}/src/surfels_unknown_space.cl.h"
  VERBATIM
  )

add_library(surfels_unknown_space_lib SHARED
  src/surfels_unknown_space.cpp

  src/surfels_unknown_space.cl.h
  src/surfels_unknown_space_opencl.cpp
  )

target_link_libraries(surfels_unknown_space_lib
  ${PCL_LIBRARIES}
  ${Eigen3_LIBRARIES}
  ${OpenCL_LIBRARIES}
)

add_executable(surfels_unknown_space_example
  src/surfels_unknown_space_example.cpp
)

target_link_libraries(surfels_unknown_space_example
  ${PCL_LIBRARIES}
  ${Eigen3_LIBRARIES}
  ${OpenCL_LIBRARIES}
  surfels_unknown_space_lib
)

add_executable(compare_surfel_clouds
  src/compare_surfel_clouds.cpp
)

target_link_libraries(compare_surfel_clouds
  ${PCL_LIBRARIES}
  ${Eigen3_LIBRARIES}
)


