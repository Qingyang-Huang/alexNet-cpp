enable_testing()

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

set(BINARY ${CMAKE_PROJECT_NAME}_tst)
file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.h *.cpp)
set(SOURCES ${TEST_SOURCES})
add_executable(${BINARY} ${TEST_SOURCES})
add_test(NAME ${BINARY} COMMAND ${BINARY})
# 链接src生成的lib库和gtest库
target_link_libraries(${BINARY} PUBLIC ${CMAKE_PROJECT_NAME}_lib ${GTEST_LIBRARIES} ${OpenCV_LIBS} pthread)


