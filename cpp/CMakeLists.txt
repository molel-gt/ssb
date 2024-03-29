cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME conduction)
project(${PROJECT_NAME} LANGUAGES C CXX)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT TARGET dolfinx)
  find_package(DOLFINX REQUIRED)
endif()

find_package(PkgConfig)

# PETSc
if (PKG_CONFIG_FOUND)
    pkg_check_modules(PETSC PETSc)
endif()

# Add target to compile UFL files
if (PETSC_SCALAR_COMPLEX EQUAL 1)
  set(SCALAR_TYPE "--scalar_type=double _Complex")
endif()
add_custom_command(
  OUTPUT conduction.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/conduction.py ${SCALAR_TYPE}
  VERBATIM DEPENDS conduction.py COMMENT "Compile conduction.py using FFCx")

add_executable(${PROJECT_NAME} conduction.cpp ${CMAKE_CURRENT_BINARY_DIR}/conduction.c)
target_link_libraries(${PROJECT_NAME} dolfinx)

# Do not throw error for 'multi-line comments' (these are typical in
# rst which includes LaTeX)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wno-comment" HAVE_NO_MULTLINE)
set_source_files_properties(conduction.cpp PROPERTIES COMPILE_FLAGS "$<$<BOOL:${HAVE_NO_MULTLINE}>:-Wno-comment -Wall -Wextra -pedantic -Werror>")

# Test targets (used by DOLFINx testing system)
set(TEST_PARAMETERS2 -np 2 ${MPIEXEC_PARAMS} "./${PROJECT_NAME}")
set(TEST_PARAMETERS3 -np 3 ${MPIEXEC_PARAMS} "./${PROJECT_NAME}")
add_test(NAME ${PROJECT_NAME}_mpi_2 COMMAND "mpirun" ${TEST_PARAMETERS2})
add_test(NAME ${PROJECT_NAME}_mpi_3 COMMAND "mpirun" ${TEST_PARAMETERS3})
add_test(NAME ${PROJECT_NAME}_serial COMMAND ${PROJECT_NAME})