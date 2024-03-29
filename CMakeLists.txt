cmake_minimum_required(VERSION 3.10)

project(alexNet-cpp VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

include_directories(src)

add_subdirectory(src)
add_subdirectory(test)
# add_subdirectory(lib/gtest)
# add_subdirectory(lib/opencv)
