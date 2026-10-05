# sscbirrt: the native C++20 core of sscbirrt. See docs/native-design.md.
cmake_minimum_required(VERSION 3.16)
project(sscbirrt VERSION 3.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(SSCBIRRT_BUILD_TESTS "Build the ctest targets" ON)
option(SSCBIRRT_BUILD_PYTHON "Build the sscbirrt._native pybind11 module (the wheel build turns this on)" OFF)
option(SSCBIRRT_SANITIZE "Build core and tests with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

# The core depends on the C++20 standard library and nothing else (design: dependency boundary).
include(GNUInstallDirs)
add_library(sscbirrt_core STATIC
  src/space.cpp
  src/sets.cpp
  src/motion.cpp
  src/config.cpp
  src/planner.cpp
  src/transform.cpp)
add_library(sscbirrt::core ALIAS sscbirrt_core)
target_include_directories(sscbirrt_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(sscbirrt_core PUBLIC cxx_std_20)
target_compile_options(sscbirrt_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(sscbirrt_core PROPERTIES POSITION_INDEPENDENT_CODE ON OUTPUT_NAME sscbirrt EXPORT_NAME core)

# The pose-region runtime: TSRs and TSR chains are sstsr's own C++ (sstsr >= 3.3, #184), found through its CMake
# package with no version pinned: pass -DSSCBIRRT_SSTSR_CMAKE_DIR=$(python -c "import tsr; print(tsr.get_cmake_dir())"),
# put it on CMAKE_PREFIX_PATH, or let the build ask the interpreter. tsr.get_cmake_dir() needs a built sstsr wheel; an
# editable install has no CMake package. sscbirrt::tsr adapts it to the core's types (the core stays standard library
# only) and is the only target that links it: the package's INTERFACE_SOURCES compile into whatever links it.
set(SSCBIRRT_SSTSR_CMAKE_DIR "" CACHE PATH "Directory holding sstsr_cppConfig.cmake (tsr.get_cmake_dir())")
if(SSCBIRRT_SSTSR_CMAKE_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${SSCBIRRT_SSTSR_CMAKE_DIR}")
else()
  find_package(Python QUIET COMPONENTS Interpreter)
  if(Python_EXECUTABLE)
    execute_process(COMMAND "${Python_EXECUTABLE}" -c "import tsr; print(tsr.get_cmake_dir(), end='')"
      OUTPUT_VARIABLE _sstsr_dir RESULT_VARIABLE _sstsr_rc ERROR_VARIABLE _sstsr_err)
    if(_sstsr_rc EQUAL 0 AND _sstsr_dir)
      list(APPEND CMAKE_PREFIX_PATH "${_sstsr_dir}")
    endif()
  endif()
endif()
find_package(sstsr_cpp CONFIG QUIET)
if(NOT TARGET sstsr::sstsr_cpp)
  message(FATAL_ERROR "sscbirrt needs sstsr's C++ core (sstsr >= 3.3): pip install 'sstsr>=3.3,<4' and pass "
    "-DSSCBIRRT_SSTSR_CMAKE_DIR=$(python -c \"import tsr; print(tsr.get_cmake_dir())\"). ${_sstsr_err}")
endif()
message(STATUS "sscbirrt: sstsr C++ core from ${sstsr_cpp_DIR}")
add_library(sscbirrt_tsr STATIC src/tsr/tsr_set.cpp)
add_library(sscbirrt::tsr ALIAS sscbirrt_tsr)
target_link_libraries(sscbirrt_tsr PUBLIC sscbirrt_core PRIVATE sstsr::sstsr_cpp)
target_include_directories(sscbirrt_tsr PUBLIC
  $<BUILD_INTERFACE:$<TARGET_PROPERTY:sstsr::sstsr_cpp,INTERFACE_INCLUDE_DIRECTORIES>>)
target_compile_options(sscbirrt_tsr PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # sstsr's sources compile here under our -Werror; GCC 12/13 raise a known false -Warray-bounds on them.
  target_compile_options(sscbirrt_tsr PRIVATE -Wno-array-bounds)
endif()
set_target_properties(sscbirrt_tsr PROPERTIES POSITION_INDEPENDENT_CODE ON OUTPUT_NAME sscbirrt_tsr EXPORT_NAME tsr)

# Eigen for the SSIK adapter: the system's package when one is installed, else the pinned Eigen release,
# downloaded once with a checked hash and exposed through a generated Eigen3Config.cmake so that ssik_cpp's
# find_dependency(Eigen3 NO_MODULE) resolves. Header-only: nothing of Eigen is built. This is what lets the
# sdist compile on a bare machine and the wheels build inside the manylinux image (#153).
set(SSCBIRRT_FETCH_EIGEN AUTO CACHE STRING "Download the pinned Eigen when no Eigen3 package is found (ON, OFF, or AUTO)")
set(SSCBIRRT_EIGEN_VERSION 3.4.0)
set(SSCBIRRT_EIGEN_SHA256 8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72)
if(NOT SSCBIRRT_WITH_SSIK STREQUAL "OFF" AND NOT SSCBIRRT_FETCH_EIGEN STREQUAL "OFF")
  find_package(Eigen3 CONFIG QUIET)
  if(NOT TARGET Eigen3::Eigen)
    set(_eigen_deps "${CMAKE_BINARY_DIR}/_deps")
    set(_eigen_root "${_eigen_deps}/eigen-${SSCBIRRT_EIGEN_VERSION}")
    set(_eigen_tar "${_eigen_deps}/eigen-${SSCBIRRT_EIGEN_VERSION}.tar.gz")
    if(NOT EXISTS "${_eigen_root}/Eigen/Core")
      file(MAKE_DIRECTORY "${_eigen_deps}")
      message(STATUS "sscbirrt: no Eigen3 package found; downloading Eigen ${SSCBIRRT_EIGEN_VERSION}")
      file(DOWNLOAD "https://gitlab.com/libeigen/eigen/-/archive/${SSCBIRRT_EIGEN_VERSION}/eigen-${SSCBIRRT_EIGEN_VERSION}.tar.gz"
        "${_eigen_tar}" EXPECTED_HASH SHA256=${SSCBIRRT_EIGEN_SHA256} STATUS _eigen_status)
      list(GET _eigen_status 0 _eigen_rc)
      if(_eigen_rc EQUAL 0)
        execute_process(COMMAND "${CMAKE_COMMAND}" -E tar xzf "${_eigen_tar}" WORKING_DIRECTORY "${_eigen_deps}" RESULT_VARIABLE _eigen_rc)
      endif()
      if(NOT _eigen_rc EQUAL 0 OR NOT EXISTS "${_eigen_root}/Eigen/Core")
        list(GET _eigen_status 1 _eigen_why)
        if(SSCBIRRT_FETCH_EIGEN STREQUAL "ON")
          message(FATAL_ERROR "sscbirrt: could not fetch Eigen ${SSCBIRRT_EIGEN_VERSION}: ${_eigen_why}")
        endif()
        message(STATUS "sscbirrt: could not fetch Eigen ${SSCBIRRT_EIGEN_VERSION} (${_eigen_why}); the SSIK adapter needs an installed Eigen3")
      endif()
    endif()
    if(EXISTS "${_eigen_root}/Eigen/Core")
      set(_eigen_cfg "${_eigen_deps}/eigen-config")
      file(WRITE "${_eigen_cfg}/Eigen3Config.cmake" "# Generated by sscbirrt for the downloaded Eigen ${SSCBIRRT_EIGEN_VERSION}
set(Eigen3_VERSION ${SSCBIRRT_EIGEN_VERSION})
set(EIGEN3_INCLUDE_DIR \"${_eigen_root}\")
set(EIGEN3_INCLUDE_DIRS \"${_eigen_root}\")
if(NOT TARGET Eigen3::Eigen)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES \"${_eigen_root}\")
endif()
set(Eigen3_FOUND TRUE)
")
      include(CMakePackageConfigHelpers)
      write_basic_package_version_file("${_eigen_cfg}/Eigen3ConfigVersion.cmake" VERSION ${SSCBIRRT_EIGEN_VERSION} COMPATIBILITY AnyNewerVersion)
      set(Eigen3_DIR "${_eigen_cfg}" CACHE PATH "Eigen3 package directory (sscbirrt's downloaded copy)" FORCE)
      find_package(Eigen3 CONFIG REQUIRED)
      message(STATUS "sscbirrt: using downloaded Eigen ${SSCBIRRT_EIGEN_VERSION} at ${_eigen_root}")
    endif()
  endif()
endif()

# The SSIK adapter: the only target that sees Eigen and ssik's header-only solvers. AUTO: build it when
# ssik_cpp (ssik >= 7.0, which ships its CMake package; pass -DSSCBIRRT_SSIK_CMAKE_DIR=$(python -c
# "import ssik; print(ssik.get_cmake_dir())") or put it on CMAKE_PREFIX_PATH) and Eigen3 are found.
set(SSCBIRRT_WITH_SSIK AUTO CACHE STRING "Build sscbirrt::ssik (ON, OFF, or AUTO)")
set(SSCBIRRT_SSIK_CMAKE_DIR "" CACHE PATH "Directory holding ssik_cppConfig.cmake (ssik.get_cmake_dir())")
set(SSCBIRRT_HAS_SSIK OFF)
if(NOT SSCBIRRT_WITH_SSIK STREQUAL "OFF")
  if(SSCBIRRT_SSIK_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${SSCBIRRT_SSIK_CMAKE_DIR}")
  elseif(SSCBIRRT_BUILD_PYTHON)
    # The wheel build: ask the build interpreter, if ssik is importable there.
    find_package(Python QUIET COMPONENTS Interpreter)
    if(Python_EXECUTABLE)
      execute_process(COMMAND "${Python_EXECUTABLE}" -c "import ssik; print(ssik.get_cmake_dir(), end='')"
        OUTPUT_VARIABLE _ssik_dir RESULT_VARIABLE _ssik_rc ERROR_QUIET)
      if(_ssik_rc EQUAL 0 AND _ssik_dir)
        list(APPEND CMAKE_PREFIX_PATH "${_ssik_dir}")
      endif()
    endif()
  endif()
  if(SSCBIRRT_WITH_SSIK STREQUAL "ON")
    find_package(ssik_cpp CONFIG REQUIRED)
  else()
    find_package(ssik_cpp CONFIG QUIET)
  endif()
  if(TARGET ssik::ssik_cpp)
    add_library(sscbirrt_ssik STATIC src/ssik/ssik_arm.cpp)
    add_library(sscbirrt::ssik ALIAS sscbirrt_ssik)
    target_link_libraries(sscbirrt_ssik PUBLIC sscbirrt_tsr PRIVATE ssik::ssik_cpp)
    target_compile_options(sscbirrt_ssik PRIVATE -Wall -Wextra)
    set_target_properties(sscbirrt_ssik PROPERTIES POSITION_INDEPENDENT_CODE ON OUTPUT_NAME sscbirrt_ssik EXPORT_NAME ssik)
    set(SSCBIRRT_HAS_SSIK ON)
    message(STATUS "sscbirrt: SSIK adapter enabled (ssik_cpp ${ssik_cpp_VERSION}, Eigen ${Eigen3_VERSION})")
  else()
    message(STATUS "sscbirrt: SSIK adapter disabled (ssik_cpp not found); TSR problems with SSIK report unsupported")
  endif()
endif()

# The MuJoCo scene: a separate library and, for Python, a separate module, so sscbirrt._native and
# `import sscbirrt` never touch MuJoCo. AUTO: build when a mujoco package (headers + library) is found,
# through -DSSCBIRRT_MUJOCO_DIR=<site-packages/mujoco> or the build interpreter.
set(SSCBIRRT_WITH_MUJOCO AUTO CACHE STRING "Build sscbirrt::mujoco (ON, OFF, or AUTO)")
set(SSCBIRRT_MUJOCO_DIR "" CACHE PATH "The installed mujoco Python package directory (holds include/ and libmujoco)")
set(SSCBIRRT_HAS_MUJOCO OFF)
if(NOT SSCBIRRT_WITH_MUJOCO STREQUAL "OFF")
  set(_mj_dir "${SSCBIRRT_MUJOCO_DIR}")
  if(NOT _mj_dir AND SSCBIRRT_BUILD_PYTHON)
    find_package(Python QUIET COMPONENTS Interpreter)
    if(Python_EXECUTABLE)
      execute_process(COMMAND "${Python_EXECUTABLE}" -c "import mujoco, os; print(os.path.dirname(mujoco.__file__), end='')"
        OUTPUT_VARIABLE _mj_dir RESULT_VARIABLE _mj_rc ERROR_QUIET)
      if(NOT _mj_rc EQUAL 0)
        set(_mj_dir "")
      endif()
    endif()
  endif()
  set(_mj_lib "")
  if(_mj_dir AND EXISTS "${_mj_dir}/include/mujoco/mujoco.h")
    file(GLOB _mj_libs "${_mj_dir}/libmujoco.*.dylib" "${_mj_dir}/libmujoco.so.*" "${_mj_dir}/mujoco.dll")
    list(LENGTH _mj_libs _n)
    if(_n GREATER 0)
      list(GET _mj_libs 0 _mj_lib)
    endif()
  endif()
  if(_mj_lib)
    add_library(mujoco_wheel SHARED IMPORTED)
    set_target_properties(mujoco_wheel PROPERTIES IMPORTED_LOCATION "${_mj_lib}" INTERFACE_INCLUDE_DIRECTORIES "${_mj_dir}/include")
    add_library(sscbirrt_mujoco STATIC src/mujoco/scene.cpp src/mujoco/snapshot.cpp src/mujoco/validator.cpp src/mujoco/sha256.cpp)
    add_library(sscbirrt::mujoco ALIAS sscbirrt_mujoco)
    target_link_libraries(sscbirrt_mujoco PUBLIC sscbirrt_core mujoco_wheel)
    target_compile_options(sscbirrt_mujoco PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(sscbirrt_mujoco PROPERTIES POSITION_INDEPENDENT_CODE ON OUTPUT_NAME sscbirrt_mujoco EXPORT_NAME mujoco)
    get_filename_component(_mj_libname "${_mj_lib}" NAME)
    set(SSCBIRRT_MUJOCO_LIBNAME "${_mj_libname}")
    if(APPLE)
      # The dylib's install name is a framework path inside the wheel; anything linking it must reference
      # @rpath/<libname> instead, as the mujoco wheel's own modules do.
      execute_process(COMMAND otool -D "${_mj_lib}" OUTPUT_VARIABLE _mj_otool OUTPUT_STRIP_TRAILING_WHITESPACE)
      string(REGEX REPLACE ".*\n" "" SSCBIRRT_MUJOCO_INSTALL_NAME "${_mj_otool}")
    endif()
    function(sscbirrt_link_mujoco_wheel target)
      if(APPLE)
        add_custom_command(TARGET ${target} POST_BUILD
          COMMAND install_name_tool -change "${SSCBIRRT_MUJOCO_INSTALL_NAME}" "@rpath/${SSCBIRRT_MUJOCO_LIBNAME}" "$<TARGET_FILE:${target}>"
          COMMENT "Point ${target} at @rpath/${SSCBIRRT_MUJOCO_LIBNAME}")
      endif()
    endfunction()
    set(SSCBIRRT_HAS_MUJOCO ON)
    message(STATUS "sscbirrt: MuJoCo scene enabled (${_mj_lib})")
  elseif(SSCBIRRT_WITH_MUJOCO STREQUAL "ON")
    message(FATAL_ERROR "SSCBIRRT_WITH_MUJOCO=ON but no mujoco package with include/mujoco/mujoco.h and libmujoco was found (set SSCBIRRT_MUJOCO_DIR)")
  else()
    message(STATUS "sscbirrt: MuJoCo scene disabled (no mujoco package found)")
  endif()
endif()

if(SSCBIRRT_SANITIZE)
  set(SSCBIRRT_SANITIZE_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
  target_compile_options(sscbirrt_core PUBLIC ${SSCBIRRT_SANITIZE_FLAGS})
  target_link_options(sscbirrt_core PUBLIC ${SSCBIRRT_SANITIZE_FLAGS})
endif()

# Install and export the package so a downstream C++ project can
#   find_package(sscbirrt REQUIRED)
#   target_link_libraries(app PRIVATE sscbirrt::core)
# examples/consumer/ is that project, built against the installed package in CI. Toggle with -DSSCBIRRT_INSTALL=OFF.
option(SSCBIRRT_INSTALL "Install headers, the library, and the sscbirrt CMake package" ON)
if(SSCBIRRT_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  install(DIRECTORY include/sscbirrt DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(TARGETS sscbirrt_core sscbirrt_tsr EXPORT sscbirrtTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(EXPORT sscbirrtTargets NAMESPACE sscbirrt:: FILE sscbirrtTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sscbirrt)
  configure_package_config_file(cmake/sscbirrtConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/sscbirrtConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sscbirrt)
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/sscbirrtConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sscbirrtConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/sscbirrtConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sscbirrt)
endif()

if(SSCBIRRT_BUILD_TESTS)
  enable_testing()
  function(sscbirrt_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE sscbirrt::core)
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME ${name} COMMAND ${name})
  endfunction()
  sscbirrt_test(test_space)
  sscbirrt_test(test_sets)
  sscbirrt_test(test_motion)
  sscbirrt_test(test_planner)
  sscbirrt_test(test_tsr_set)
  target_link_libraries(test_tsr_set PRIVATE sscbirrt::tsr)
  if(SSCBIRRT_HAS_MUJOCO)
    sscbirrt_test(test_mujoco_scene)
    target_link_libraries(test_mujoco_scene PRIVATE sscbirrt::mujoco)
    set_target_properties(test_mujoco_scene PROPERTIES BUILD_RPATH "${_mj_dir}")
    sscbirrt_link_mujoco_wheel(test_mujoco_scene)
    sscbirrt_test(test_mujoco_validator)
    target_link_libraries(test_mujoco_validator PRIVATE sscbirrt::mujoco)
    set_target_properties(test_mujoco_validator PROPERTIES BUILD_RPATH "${_mj_dir}")
    sscbirrt_link_mujoco_wheel(test_mujoco_validator)
  endif()
endif()

# The Python binding is the only target that sees Python. scikit-build-core drives this file with
# SSCBIRRT_BUILD_PYTHON=ON and puts pybind11's CMake package on the prefix path.
if(SSCBIRRT_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_native MODULE bindings/sscbirrt_native.cpp)
  target_link_libraries(_native PRIVATE sscbirrt::core sscbirrt::tsr)
  if(SSCBIRRT_HAS_SSIK)
    target_link_libraries(_native PRIVATE sscbirrt::ssik)
    target_compile_definitions(_native PRIVATE SSCBIRRT_HAS_SSIK=1)
  endif()
  target_compile_options(_native PRIVATE -Wall -Wextra)
  install(TARGETS _native DESTINATION sscbirrt)

  if(SSCBIRRT_HAS_MUJOCO)
    pybind11_add_module(_native_mujoco MODULE bindings/sscbirrt_native_mujoco.cpp)
    target_link_libraries(_native_mujoco PRIVATE sscbirrt::mujoco)
    target_compile_options(_native_mujoco PRIVATE -Wall -Wextra)
    target_compile_definitions(_native_mujoco PRIVATE SSCBIRRT_MUJOCO_LIBNAME="${SSCBIRRT_MUJOCO_LIBNAME}")
    # The module lives in site-packages/sscbirrt and the library in site-packages/mujoco.
    set_target_properties(_native_mujoco PROPERTIES BUILD_WITH_INSTALL_RPATH ON
      INSTALL_RPATH "$<IF:$<PLATFORM_ID:Darwin>,@loader_path/../mujoco,$ORIGIN/../mujoco>")
    sscbirrt_link_mujoco_wheel(_native_mujoco)
    install(TARGETS _native_mujoco DESTINATION sscbirrt)
  endif()
endif()
