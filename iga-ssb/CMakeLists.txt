cmake_minimum_required(VERSION 3.19)

set(PROJECT_NAME iga_ssb)
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
find_package(Boost COMPONENTS ${BOOST_REQUIRED_COMPONENTS} REQUIRED)
find_package(gismo REQUIRED)
include_directories(${GISMO_INCLUDE_DIRS})
link_directories   (${GISMO_LIBRARY_DIR} )
include_directories(${Boost_INCLUDE_DIRS})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(${PROJECT_NAME} poisson.cpp )

target_link_libraries(${PROJECT_NAME} PUBLIC
                      Boost::filesystem
                      Boost::program_options
                      gismo
                      )

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Wno-comment" HAVE_NO_MULTLINE)
set_source_files_properties(
  poisson.cpp
  PROPERTIES
    # COMPILE_FLAGS
    # "$<$<BOOL:${HAVE_NO_MULTLINE}>:-Wno-comment -Wall -Wextra -pedantic -Werror>"
)
