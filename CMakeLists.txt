cmake_minimum_required(VERSION 3.20)

# Keep ggml's upstream Objective-C .m sources under its known-good C/ObjC
# compilation mode.  The MoGe sparse helper is the only Objective-C++ source;
# compile that one file through the C++ driver with -x objective-c++ rather
# than enabling OBJCXX globally (which can make CMake reinterpret ggml .m
# files as Objective-C++ on Apple Clang).
project(moge_ggml VERSION 0.4.2 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(MOGE_BUILD_CLI "Build the end-user moge-cli executable" ON)
option(MOGE_BUILD_CONVERT "Build the end-user moge-convert executable" ON)
option(MOGE_BUILD_DEV_TOOLS "Build benchmark and calibration developer tools" OFF)
option(MOGE_BUILD_TESTS "Build native validation utilities" OFF)
set(GGML_SOURCE_DIR "" CACHE PATH "Path to a standalone ggml source tree (e.g. llama.cpp/ggml)")
option(MOGE_HOMEBREW_OPENMP "Auto-detect Homebrew libomp for AppleClang" ON)

# AppleClang does not search Homebrew's keg-only libomp by default.  Apply the
# same hints used by tests/build.sh so direct `cmake -S . -B build` users do not
# lose CPU OpenMP merely because the package lives outside the SDK search path.
# Explicit OpenMP_* cache values always win.
if(APPLE AND MOGE_HOMEBREW_OPENMP)
    find_program(MOGE_BREW_EXECUTABLE brew)
    if(MOGE_BREW_EXECUTABLE)
        execute_process(
            COMMAND "${MOGE_BREW_EXECUTABLE}" --prefix libomp
            OUTPUT_VARIABLE MOGE_LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE MOGE_LIBOMP_RESULT
        )
        if(MOGE_LIBOMP_RESULT EQUAL 0
           AND EXISTS "${MOGE_LIBOMP_PREFIX}/include/omp.h"
           AND EXISTS "${MOGE_LIBOMP_PREFIX}/lib/libomp.dylib")
            if(NOT DEFINED OpenMP_C_FLAGS)
                set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -isystem${MOGE_LIBOMP_PREFIX}/include" CACHE STRING "OpenMP C flags")
            endif()
            if(NOT DEFINED OpenMP_CXX_FLAGS)
                set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -isystem${MOGE_LIBOMP_PREFIX}/include" CACHE STRING "OpenMP C++ flags")
            endif()
            if(NOT DEFINED OpenMP_C_LIB_NAMES)
                set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "OpenMP C libraries")
            endif()
            if(NOT DEFINED OpenMP_CXX_LIB_NAMES)
                set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "OpenMP C++ libraries")
            endif()
            if(NOT DEFINED OpenMP_omp_LIBRARY)
                set(OpenMP_omp_LIBRARY "${MOGE_LIBOMP_PREFIX}/lib/libomp.dylib" CACHE FILEPATH "Homebrew libomp")
            endif()
            message(STATUS "Homebrew libomp: ${MOGE_LIBOMP_PREFIX}")
        endif()
    endif()
endif()

if(GGML_SOURCE_DIR)
    # Honor the normal ggml feature switches: -DGGML_VULKAN=ON, -DGGML_METAL=ON,
    # -DGGML_NATIVE=ON, etc.  The caller controls those before add_subdirectory.
    add_subdirectory("${GGML_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)
else()
    find_package(ggml CONFIG REQUIRED)
endif()

if(TARGET ggml::ggml)
    set(MOGE_GGML_TARGET ggml::ggml)
elseif(TARGET ggml)
    set(MOGE_GGML_TARGET ggml)
else()
    message(FATAL_ERROR "ggml target not found (expected ggml::ggml or ggml)")
endif()

set(MOGE_LIBRARY_SOURCES
    src/mogg.cpp
    src/dense.cpp
    src/sparse.cpp
    src/postprocess.cpp
    src/moge.cpp
    src/moge_c.cpp
)
if(APPLE)
    list(APPEND MOGE_LIBRARY_SOURCES src/sparse_metal.mm src/calibration_metal.mm src/mpsgraph_sdpa.mm src/mpsgraph_conv.mm)
    set_source_files_properties(src/sparse_metal.mm PROPERTIES
        LANGUAGE CXX
        COMPILE_OPTIONS "-x;objective-c++"
    )
    set_source_files_properties(src/calibration_metal.mm PROPERTIES
        LANGUAGE CXX
        COMPILE_OPTIONS "-x;objective-c++"
    )
    set_source_files_properties(src/mpsgraph_sdpa.mm PROPERTIES
        LANGUAGE CXX
        COMPILE_OPTIONS "-x;objective-c++"
    )
    set_source_files_properties(src/mpsgraph_conv.mm PROPERTIES
        LANGUAGE CXX
        COMPILE_OPTIONS "-x;objective-c++"
    )
else()
    list(APPEND MOGE_LIBRARY_SOURCES src/sparse_metal_stub.cpp src/calibration_metal_stub.cpp src/mpsgraph_sdpa_stub.cpp src/mpsgraph_conv_stub.cpp)
endif()
add_library(moge-ggml STATIC ${MOGE_LIBRARY_SOURCES})
add_library(moge::moge-ggml ALIAS moge-ggml)
target_compile_features(moge-ggml PUBLIC cxx_std_17)
target_include_directories(moge-ggml
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(moge-ggml PUBLIC ${MOGE_GGML_TARGET})
if(APPLE)
    target_link_libraries(moge-ggml PUBLIC "-framework Metal" "-framework MetalPerformanceShaders" "-framework MetalPerformanceShadersGraph" "-framework Foundation")
endif()

if(MOGE_BUILD_CONVERT)
    add_executable(moge-convert tools/moge-convert.cpp)
    target_compile_features(moge-convert PRIVATE cxx_std_17)
    target_compile_definitions(moge-convert PRIVATE MOGE_PROJECT_VERSION_STRING="${PROJECT_VERSION}")
    target_include_directories(moge-convert PRIVATE ${GGML_SOURCE_DIR}/include)
    target_link_libraries(moge-convert PRIVATE ${MOGE_GGML_TARGET})
endif()

if(MOGE_BUILD_CLI)
    add_executable(moge-cli examples/moge-cli.cpp)
    target_compile_features(moge-cli PRIVATE cxx_std_17)
    target_include_directories(moge-cli PRIVATE third_party)
    target_link_libraries(moge-cli PRIVATE moge-ggml)
endif()

if(MOGE_BUILD_DEV_TOOLS)
    add_executable(moge-calibrate tools/moge-calibrate.cpp)
    target_compile_features(moge-calibrate PRIVATE cxx_std_17)
    target_include_directories(moge-calibrate PRIVATE third_party)
    target_link_libraries(moge-calibrate PRIVATE moge-ggml)

    add_executable(moge-bench examples/moge-bench.cpp)
    target_compile_features(moge-bench PRIVATE cxx_std_17)
    target_link_libraries(moge-bench PRIVATE moge-ggml)
endif()

if(MOGE_BUILD_TESTS)
    add_executable(moge-dump tests/native/moge-dump.cpp)
    target_compile_features(moge-dump PRIVATE cxx_std_17)
    target_link_libraries(moge-dump PRIVATE moge-ggml)

    add_executable(moge-backend-smoke tests/native/moge-backend-smoke.cpp)
    target_compile_features(moge-backend-smoke PRIVATE cxx_std_17)
    target_link_libraries(moge-backend-smoke PRIVATE ${MOGE_GGML_TARGET})

    add_executable(mogg-memory-smoke tests/native/mogg-memory-smoke.cpp src/mogg.cpp)
    target_compile_features(mogg-memory-smoke PRIVATE cxx_std_17)
    target_include_directories(mogg-memory-smoke PRIVATE src)

    add_executable(moge-c-api-smoke tests/native/moge-c-api-smoke.c)
    target_link_libraries(moge-c-api-smoke PRIVATE moge-ggml)
endif()

# Stable install surface. When MoGe is built against an installed ggml package,
# also install an exported CMake package. A vendored GGML_SOURCE_DIR build still
# installs the library/headers/CLI, but cannot export ggml's local build target
# as part of MoGe's package without taking ownership of ggml's install rules.
if(GGML_SOURCE_DIR)
    set(MOGE_CAN_EXPORT_PACKAGE OFF)
else()
    set(MOGE_CAN_EXPORT_PACKAGE ON)
endif()

if(MOGE_CAN_EXPORT_PACKAGE)
    install(TARGETS moge-ggml
        EXPORT moge-ggml-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
else()
    install(TARGETS moge-ggml
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()
if(TARGET moge-cli)
    install(TARGETS moge-cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
if(TARGET moge-convert)
    install(TARGETS moge-convert RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
if(TARGET moge-calibrate)
    install(TARGETS moge-calibrate RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
if(TARGET moge-bench)
    install(TARGETS moge-bench RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/moge_ggml DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES
    LICENSE
    THIRD_PARTY_NOTICES.md
    CREDITS.md
    CHANGELOG.md
    SECURITY.md
    docs/RELEASE_0.4.2.md
    DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/moge-ggml
)

if(MOGE_CAN_EXPORT_PACKAGE)
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/moge-ggml-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/moge-ggml-config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/moge-ggml
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/moge-ggml-config-version.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(EXPORT moge-ggml-targets
        FILE moge-ggml-targets.cmake
        NAMESPACE moge::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/moge-ggml
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/moge-ggml-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/moge-ggml-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/moge-ggml
    )
endif()
