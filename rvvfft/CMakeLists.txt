cmake_minimum_required(VERSION 3.22)
project(rvvfft
    VERSION 0.1.0
    DESCRIPTION "High-performance FFT library for self-developed RISC-V RVV CPU chips"
    LANGUAGES C)

# ─── Options ──────────────────────────────────────────────────────────────────
option(RVVFFT_BUILD_TESTS    "Build correctness test driver"  ON)
option(RVVFFT_BUILD_BENCH    "Build performance benchmark"    ON)
option(RVVFFT_BUILD_EXAMPLES "Build usage examples"           ON)
option(RVVFFT_VLEN_256       "Optimise for VLEN=256"          ON)
option(RVVFFT_VLEN_1024      "Also support VLEN=1024"         ON)

# ─── Compiler settings ────────────────────────────────────────────────────────
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Flags shared by all targets
set(RVVFFT_COMPILE_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -O3
    -ffast-math
    -fno-omit-frame-pointer   # Needed for profiling with perf
)

# RVV-specific flags (only meaningful when cross-compiling to RISC-V)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    list(APPEND RVVFFT_COMPILE_FLAGS
        -march=rv64gcv          # Enable V extension
        -mabi=lp64d
    )
    if(RVVFFT_VLEN_256)
        list(APPEND RVVFFT_COMPILE_FLAGS -DRVVFFT_VLEN_DEFAULT=256)
    endif()
    if(RVVFFT_VLEN_1024)
        list(APPEND RVVFFT_COMPILE_FLAGS -DRVVFFT_VLEN_1024_SUPPORT=1)
    endif()
endif()

# ─── Library sources ──────────────────────────────────────────────────────────
set(RVVFFT_LIB_SOURCES
    src/rvvfft.c
    src/planner/planner.c
    src/executor/executor_recursive.c
    src/utils/twiddle.c
    src/utils/bitrev.c
    src/utils/alignment.c
    src/kernel/handwritten/radix2_f32.c
)

# Generated kernel sources (populated by running codegen/generator.py)
# Only included when cross-compiling to RISC-V — they require <riscv_vector.h>
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    file(GLOB RVVFFT_GENERATED_SOURCES
         "src/kernel/generated/*.c")
    list(APPEND RVVFFT_LIB_SOURCES ${RVVFFT_GENERATED_SOURCES})
endif()

# ─── Static library target ────────────────────────────────────────────────────
add_library(rvvfft_static STATIC ${RVVFFT_LIB_SOURCES})
target_include_directories(rvvfft_static PUBLIC include)
target_compile_options(rvvfft_static PRIVATE ${RVVFFT_COMPILE_FLAGS})
target_link_libraries(rvvfft_static PUBLIC m)

# Convenience alias
add_library(rvvfft::rvvfft ALIAS rvvfft_static)

# ─── Shared library target ────────────────────────────────────────────────────
add_library(rvvfft_shared SHARED ${RVVFFT_LIB_SOURCES})
target_include_directories(rvvfft_shared PUBLIC include)
target_compile_options(rvvfft_shared PRIVATE ${RVVFFT_COMPILE_FLAGS})
target_link_libraries(rvvfft_shared PUBLIC m)
set_target_properties(rvvfft_shared PROPERTIES
    OUTPUT_NAME rvvfft
    SOVERSION   ${PROJECT_VERSION_MAJOR}
    VERSION     ${PROJECT_VERSION})

# ─── Code generation custom target ───────────────────────────────────────────
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    add_custom_target(rvvfft_codegen
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/codegen/generator.py
                --outdir ${CMAKE_SOURCE_DIR}/src/kernel/generated
                --registry ${CMAKE_SOURCE_DIR}/src/kernel/generated/codelet_registry.h
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/codegen
        COMMENT "Generating RVV butterfly kernels …"
        VERBATIM)
    add_custom_target(rvvfft_codegen_verify
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/codegen/verify_math.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/codegen
        COMMENT "Verifying code generation math rules (SymPy) …"
        VERBATIM)
endif()

# ─── Tests ────────────────────────────────────────────────────────────────────
if(RVVFFT_BUILD_TESTS)
    enable_testing()

    # Generate golden data
    if(Python3_FOUND)
        add_custom_target(gen_golden
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_SOURCE_DIR}/tests/gen_golden.py
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Generating FFT golden reference data …"
            VERBATIM)
    endif()

    add_executable(test_correctness
        tests/test_correctness.c)
    target_include_directories(test_correctness PRIVATE include)
    target_link_libraries(test_correctness PRIVATE rvvfft_static)
    target_compile_options(test_correctness PRIVATE ${RVVFFT_COMPILE_FLAGS})

    add_test(NAME correctness
        COMMAND test_correctness ${CMAKE_SOURCE_DIR}/tests/golden)
endif()

# ─── Benchmarks ───────────────────────────────────────────────────────────────
if(RVVFFT_BUILD_BENCH)
    add_executable(bench
        bench/bench_main.c
        bench/bench_utils.c)
    target_include_directories(bench PRIVATE include)
    target_link_libraries(bench PRIVATE rvvfft_static)
    target_compile_options(bench PRIVATE ${RVVFFT_COMPILE_FLAGS})
endif()

# ─── Examples ─────────────────────────────────────────────────────────────────
if(RVVFFT_BUILD_EXAMPLES)
    foreach(ex c2c r2c split)
        if(EXISTS "${CMAKE_SOURCE_DIR}/examples/example_${ex}.c")
            add_executable(example_${ex}
                examples/example_${ex}.c)
            target_include_directories(example_${ex} PRIVATE include)
            target_link_libraries(example_${ex} PRIVATE rvvfft_static)
            target_compile_options(example_${ex} PRIVATE ${RVVFFT_COMPILE_FLAGS})
        endif()
    endforeach()
endif()

# ─── Installation ─────────────────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS rvvfft_static rvvfft_shared
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES
    include/rvvfft.h
    include/rvvfft_types.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rvvfft)

# ─── Summary ──────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "── rvvfft configuration ─────────────────────────")
message(STATUS "  Version       : ${PROJECT_VERSION}")
message(STATUS "  System        : ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Compiler      : ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  VLEN=256      : ${RVVFFT_VLEN_256}")
message(STATUS "  VLEN=1024     : ${RVVFFT_VLEN_1024}")
message(STATUS "  Tests         : ${RVVFFT_BUILD_TESTS}")
message(STATUS "  Bench         : ${RVVFFT_BUILD_BENCH}")
message(STATUS "  Python3       : ${Python3_FOUND}")
message(STATUS "─────────────────────────────────────────────────")
message(STATUS "")
