cmake_minimum_required(VERSION 3.14)
project(curve_fitting)

set(CMAKE_CXX_STANDARD 14)
# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
# Eigen
include_directories("/usr/include/eigen3")
add_executable(curve_fitting main.cpp)
target_link_libraries(curve_fitting ${CERES_LIBRARIES})