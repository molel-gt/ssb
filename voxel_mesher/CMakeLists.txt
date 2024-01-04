cmake_minimum_required(VERSION 3.19)

set(PROJECT_NAME voxel_mesher)
project(${PROJECT_NAME} LANGUAGES C CXX)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# boost
set(BOOST_REQUIRED_COMPONENTS
    filesystem
    program_options
)
set (OPENCV_REQUIRED_COMPONENTS
     opencv_core
     opencv_imgcodecs
     )
find_package(Boost COMPONENTS ${BOOST_REQUIRED_COMPONENTS} REQUIRED)

set(HDF5_PREFER_PARALLEL TRUE)
set(HDF5_FIND_DEBUG TRUE)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(OpenCV COMPONENTS ${OPENCV_REQUIRED_COMPONENTS} REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(
  ${PROJECT_NAME} main.cpp
)
target_include_directories(${PROJECT_NAME} PRIVATE include)
target_link_libraries(${PROJECT_NAME} PUBLIC  
    Boost::filesystem
    Boost::program_options
    hdf5::hdf5
    ${OpenCV_LIBS}
    opencv_core
    opencv_imgcodecs
)

# Do not throw error for 'multi-line comments' (these are typical in rst which
# includes LaTeX)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Wno-comment" HAVE_NO_MULTLINE)
set_source_files_properties(
  main.cpp
  PROPERTIES
    COMPILE_FLAGS
    "$<$<BOOL:${HAVE_NO_MULTLINE}>:-Wno-comment -Wall -Wextra -pedantic -Werror>"
)