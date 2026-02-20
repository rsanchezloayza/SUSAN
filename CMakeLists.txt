# Define Project

cmake_minimum_required(VERSION 3.14)
project(SUSAN LANGUAGES CXX CUDA)
set(CMAKE_BUILD_TYPE Release)

########################################################################
########################################################################

# Configure EIGEN
include(FetchContent)
set(EIGEN_VERSION 3.4)
find_package(Eigen3 ${EIGEN_VERSION} QUIET)
if(NOT EIGEN3_FOUND)
  set(BUILD_TESTING OFF CACHE INTERNAL "")
  FetchContent_Declare(eigen
    GIT_REPOSITORY  https://gitlab.com/libeigen/eigen.git
    GIT_TAG         ${EIGEN_VERSION}
    GIT_SHALLOW     ON)
  FetchContent_MakeAvailable(eigen)
  unset(BUILD_TESTING CACHE)
endif()

########################################################################
########################################################################

# Configure LodePNG
include(FetchContent)
FetchContent_Declare(
  lodepng
  GIT_REPOSITORY https://github.com/lvandeve/lodepng.git
  GIT_TAG        master
  GIT_SHALLOW    ON
)
FetchContent_MakeAvailable(lodepng)
add_library(lodepng ${lodepng_SOURCE_DIR}/lodepng.cpp)
target_include_directories(lodepng PUBLIC ${lodepng_SOURCE_DIR})

########################################################################
########################################################################

# Configure CUDA:
# Below SM 5.2: DEPRECATED [for SUSAN]
# v10.2: up to SM 7.5
# v11.0: SM 8.0
# v11.1: SM 8.6
# v11.5: SM 8.7
# v11.8: SM 9.0
# v12.0: up to SM 9.0
find_package(CUDAToolkit)
list(APPEND CUDA_ARCHS 52 53 60 61 62 70 72 75)
if(CUDAToolkit_VERSION_MAJOR LESS 10)
	message(FATAL_ERROR "SUSAN requires CUDA toolkit at least version 10.")
elseif(CUDAToolkit_VERSION_MAJOR EQUAL 11)
	list(APPEND CUDA_ARCHS 80)
	if(CUDAToolkit_VERSION_MINOR GREATER_EQUAL 1)
		list(APPEND CUDA_ARCHS 86)
	elseif(CUDAToolkit_VERSION_MINOR GREATER_EQUAL 5)
		list(APPEND CUDA_ARCHS 87)
	elseif(CUDAToolkit_VERSION_MINOR GREATER_EQUAL 8)
		list(APPEND CUDA_ARCHS 90)
	endif()
elseif(CUDAToolkit_VERSION_MAJOR EQUAL 12)
	list(APPEND CUDA_ARCHS 80 86 87 89 90)
elseif(CUDAToolkit_VERSION_MAJOR EQUAL 13)
        list(REMOVE_ITEM CUDA_ARCHS 52 53 60 61 62 70 72)
        list(APPEND CUDA_ARCHS 80 86 87 89 90 120)
else()
	message(FATAL_ERROR "SUSAN does not support your CUDA toolkit version.")
endif()
string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr")
string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --display_error_number")

########################################################################
########################################################################

# Basic Executables
set(target_list estimate_ctf reconstruct rec_subtomos aligner refs_aligner ctf_refiner crop_projections)
foreach(target IN LISTS target_list)
	set(bin susan_${target})
	add_executable(${bin} src/${target}.cu)
	target_link_libraries(${bin} PRIVATE cufft Eigen3::Eigen)
	if(target STREQUAL "estimate_ctf")
        target_link_libraries(${bin} PRIVATE lodepng)
    endif()
	set_property(TARGET ${bin} PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHS})
	add_custom_command(TARGET ${bin} POST_BUILD 
		COMMAND mkdir -p ${CMAKE_SOURCE_DIR}/susan/bin
		COMMAND cp -v ${CMAKE_CURRENT_BINARY_DIR}/${bin} ${CMAKE_SOURCE_DIR}/susan/bin/
	)
endforeach()

# Conditionally build if MPI is found
find_package(MPI)
if( MPI_FOUND )
	set(target_list reconstruct_mpi aligner_mpi ctf_refiner_mpi)
	foreach(target IN LISTS target_list)
		set(bin susan_${target})
		add_executable(${bin} src/${target}.cu)
		target_compile_definitions(${bin} PUBLIC OPENMPI)
		target_link_libraries(${bin} cufft ${MPI_CXX_LIBRARIES} Eigen3::Eigen)
		target_include_directories(${bin} PUBLIC ${MPI_CXX_INCLUDE_DIRS})
		set_property(TARGET ${bin} PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHS})
	endforeach()
endif( MPI_FOUND )

# Conditionally build if MATLAB is found
find_package(Matlab)
if( Matlab_FOUND )
	include_directories(${CMAKE_SOURCE_DIR}/src)
	file(GLOB_RECURSE mex_cpp LIST_DIRECTORIES false "${CMAKE_SOURCE_DIR}/+SUSAN/*.cpp")
	foreach(src IN LISTS mex_cpp)
		get_filename_component(folder ${src} DIRECTORY)
		get_filename_component(target ${src} NAME_WE)
		set(binary "${target}.${Matlab_MEX_EXTENSION}")
		matlab_add_mex(NAME ${target} SRC ${src} LINK_TO Eigen3::Eigen R2018a)
		add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_BINARY_DIR}/${binary}" "${folder}/.")
	endforeach()
endif( Matlab_FOUND )

# Add target to install binaries into the Python package [optional]
add_custom_target(prepare_python
	COMMAND mkdir -p ${CMAKE_SOURCE_DIR}/susan/bin
	COMMAND cp -v ${CMAKE_CURRENT_BINARY_DIR}/susan_* ${CMAKE_SOURCE_DIR}/susan/bin/
)


