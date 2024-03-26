cmake_minimum_required(VERSION 3.10)

project(alexNet-cpp VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(alexNet-cpp ${SOURCES})

# 链接OpenCV库
target_link_libraries(alexNet-cpp ${OpenCV_LIBS})
