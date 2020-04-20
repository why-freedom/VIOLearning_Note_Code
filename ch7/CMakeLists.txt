cmake_minimum_required(VERSION 2.8)
project(dogleg)

set(CMAKE_CXX_FLAGS "-std=c++11")
include(CTest)
# enable_testing()
# find_package(OpenCV REQUIRED)
# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")
# find_package(Eigen3)

ADD_LIBRARY(DogLeg_lib SHARED
    dogleg.cpp 
    trustRegion.cpp
    )

# add_executable(hello_world main.cpp)
add_executable(dogleg main.cpp)
target_link_libraries(dogleg  DogLeg_lib)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)