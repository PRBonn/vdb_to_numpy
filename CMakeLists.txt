cmake_minimum_required(VERSION 3.10)
project(vdb_to_numpy)

set(CMAKE_MODULE_PATH "/usr/local/lib/cmake/OpenVDB")
find_package(OpenVDB COMPONENTS)
find_package(Boost COMPONENTS python REQUIRED)

# Set build type
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wextra -pedantic -Werror)

link_directories(${OpenVDB_LIBRARY_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/Eigen/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/pybind11)
add_subdirectory(src/vdb_to_numpy/pybind)
