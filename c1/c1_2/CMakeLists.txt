cmake_minimum_required(VERSION 3.14)
project(c1_2)

set(CMAKE_CXX_STANDARD 14)
include_directories( "/usr/include/eigen3" )
# 为使用 sophus，您需要使用find_package命令找到它
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

add_executable(c1_2 main.cpp)
target_link_libraries(c1_2 ${Sophus_LIBRARIES})