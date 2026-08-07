# Define Project

cmake_minimum_required(VERSION 3.14)
project(SUSAN LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The CUDA executables (susan_aligner, susan_reconstruct, ...) are the only
# part of SUSAN that needs a GPU and the CUDA toolkit. Turn this OFF to build
# just the Python package and its Cython cores, which is enough to read and
# analyse SUSAN files on a machine that will never run a reconstruction.
option(SUSAN_BUILD_CUDA "Build the CUDA executables" ON)

if(SUSAN_BUILD_CUDA)
	enable_language(CUDA)
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
	message(STATUS "CUDA: DISABLED (Python package and Cython cores only)")
endif()

########################################################################
########################################################################

if(SUSAN_BUILD_CUDA)

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

# EXPERIMENTAL. Treat the exposure filter (def.ExpFilt) as a weight instead of a
# filter: it is squared into the Wiener denominator, so the reconstruction stops
# re-applying an attenuation the dose already caused. Maps come out sharper and
# noisier, and the high-frequency roll-off must then be set explicitly through
# the ad-hoc SSNR (ssnr.F). On by default; set to OFF to restore the
# numerator-only (filtering) behaviour. Requires an SSNR-regularised policy:
# with plain 'wiener' the denominator can approach zero at high frequency.
option(SUSAN_EXPFILT_WEIGHT "Square the exposure filter into the Wiener denominator" ON)
if(SUSAN_EXPFILT_WEIGHT)
	string(APPEND CMAKE_CUDA_FLAGS " -DSUSAN_EXPFILT_WEIGHT")
	message(STATUS "ExpFilt: WEIGHT (squared into the Wiener denominator) [experimental]")
else()
	message(STATUS "ExpFilt: FILTER (numerator only) [default]")
endif()

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

endif(SUSAN_BUILD_CUDA)

########################################################################
########################################################################

# Cython extensions

# If running inside an activated virtualenv and the caller did not pass an
# explicit -DPython_EXECUTABLE, pin to the venv's interpreter so a higher
# system Python (e.g. /usr/bin/python3.14) does not override it.
# This is skipped during pip's PEP 517 isolated build (where VIRTUAL_ENV
# points to a stripped temp env without NumPy) because setup.py already
# passes -DPython_EXECUTABLE=${sys.executable} in that path.
if(DEFINED ENV{VIRTUAL_ENV} AND NOT DEFINED Python_EXECUTABLE)
    set(Python_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_program(CYTHON_EXECUTABLE NAMES cython cython3 REQUIRED)

# Resolve NumPy from the *active* interpreter on every configure, instead of
# find_package(... NumPy). This is what lets `pip install ..` and a manual
# `cmake ../` coexist in this shared bin/ directory. find_package keys its
# NumPy cache on the interpreter path, which is identical for both: pip builds
# PEP 517-isolated while a manual cmake uses the venv NumPy -- same python3
# either way -- so CMake would reuse whichever include dir was cached first.
#
# PYTHONPATH is unset for the query because that is the mechanism pip uses to
# inject its build isolation overlay. Without this, a `pip install .` resolves
# NumPy to a throwaway /tmp/pip-build-env-*/overlay path, bakes it into
# bin/CMakeFiles/*/flags.make, and then pip deletes it. A plain `make` in bin/
# afterwards never reconfigures, so it keeps using the dead path and fails with
# "numpy/arrayobject.h: No such file". Querying with the overlay stripped yields
# the interpreter's own persistent NumPy, which is valid for both build paths
# and, being the same NumPy the package imports at runtime, is also the correct
# one to compile the extensions against.
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env --unset=PYTHONPATH ${Python_EXECUTABLE} -c
        "import numpy,sys; sys.stdout.write(numpy.get_include()+';'+numpy.__version__)"
    OUTPUT_VARIABLE Python_NumPy_INFO
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE Python_NumPy_RC)
if(NOT Python_NumPy_RC EQUAL 0)
    # No persistent NumPy: fall back to whatever the current environment offers
    # (an isolated build in a venv that genuinely lacks NumPy). This can still
    # bake a temporary path into the build tree, so warn.
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c
            "import numpy,sys; sys.stdout.write(numpy.get_include()+';'+numpy.__version__)"
        OUTPUT_VARIABLE Python_NumPy_INFO
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE Python_NumPy_RC)
    if(NOT Python_NumPy_RC EQUAL 0)
        message(FATAL_ERROR "NumPy not importable with ${Python_EXECUTABLE}; "
            "install it in the active environment.")
    endif()
    message(WARNING "NumPy resolved only from a build-isolation overlay; "
        "install NumPy into ${Python_EXECUTABLE} to keep bin/ reusable.")
endif()
list(GET Python_NumPy_INFO 0 Python_NumPy_INCLUDE_DIRS)
list(GET Python_NumPy_INFO 1 Python_NumPy_VERSION)
if(NOT EXISTS "${Python_NumPy_INCLUDE_DIRS}/numpy/arrayobject.h")
    message(FATAL_ERROR "NumPy headers not found at ${Python_NumPy_INCLUDE_DIRS}. "
        "If this is a stale path from an earlier build, re-run cmake in this "
        "directory to refresh it.")
endif()
message(STATUS "NumPy: v${Python_NumPy_VERSION} (${Python_NumPy_INCLUDE_DIRS})")

macro(add_cython_module MODULE PYX_FILE INSTALL_DIR)
    set(_c "${CMAKE_CURRENT_BINARY_DIR}/${MODULE}.c")
    add_custom_command(
        OUTPUT  ${_c}
        COMMAND ${CYTHON_EXECUTABLE} --output-file ${_c} ${PYX_FILE}
        DEPENDS ${PYX_FILE}
        COMMENT "Cythonizing ${PYX_FILE}"
    )
    Python_add_library(${MODULE} MODULE WITH_SOABI ${_c})
    target_include_directories(${MODULE} PRIVATE ${Python_NumPy_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/susan)
    if(Python_NumPy_VERSION VERSION_LESS "2.0")
        target_compile_options(${MODULE} PRIVATE -include ${CMAKE_SOURCE_DIR}/susan/numpy_compat.h)
    endif()
    install(TARGETS ${MODULE} DESTINATION ${INSTALL_DIR})
    add_custom_command(TARGET ${MODULE} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:${MODULE}>
            ${CMAKE_SOURCE_DIR}/${INSTALL_DIR}/
    )
endmacro()

add_cython_module(_functions_core
    ${CMAKE_SOURCE_DIR}/susan/utils/_functions_core.pyx
    susan/utils)

add_cython_module(_particles_core
    ${CMAKE_SOURCE_DIR}/susan/data/_particles_core.pyx
    susan/data)

add_cython_module(_ptclsgeom_core
    ${CMAKE_SOURCE_DIR}/susan/data/_ptclsgeom_core.pyx
    susan/data)


