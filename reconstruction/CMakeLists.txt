cmake_minimum_required(VERSION 3.12...3.29)
set(PROJECT_NAME merge_tets)
set(CMAKE_BUILD_TYPE Release)
project(${PROJECT_NAME} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# find_package(CGAL REQUIRED)


find_package(Boost
  REQUIRED
  COMPONENTS program_options
  )


add_executable(${PROJECT_NAME} merge_tetgen_outputs.cpp)
target_link_libraries(${PROJECT_NAME} Boost::program_options)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wno-comment" HAVE_NO_MULTLINE)
set_source_files_properties(tomo_to_tets.cpp PROPERTIES COMPILE_FLAGS "$<$<BOOL:${HAVE_NO_MULTLINE}>:-Wno-comment -Wall -Wextra -pedantic -Werror>")
