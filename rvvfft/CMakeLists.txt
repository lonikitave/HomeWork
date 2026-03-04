cmake_minimum_required(VERSION 3.16)
project(rvvfft VERSION 0.1.0 LANGUAGES C)

# ── 选项 ──────────────────────────────────────────────────────
option(RVVFFT_ENABLE_RVV    "启用RVV intrinsic向量化"      OFF)
set(RVVFFT_VLEN 256 CACHE STRING "向量寄存器宽度（位），支持256或1024")
option(RVVFFT_BUILD_TESTS   "构建正确性测试"                ON)
option(RVVFFT_BUILD_BENCH   "构建性能benchmark"             ON)
option(RVVFFT_BUILD_EXAMPLES "构建使用示例"                 ON)
option(RVVFFT_ENABLE_OPENMP "启用OpenMP多核并行（Phase 4）" OFF)

# ── 编译器配置 ────────────────────────────────────────────────
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 公共编译选项
add_compile_options(
    -O3
    -Wall
    -Wextra
    -Wno-unused-parameter
    -ffast-math              # 允许浮点重排（FFT性能关键）
)

# VLEN宏定义（编译时注入到所有源文件）
add_compile_definitions(RVVFFT_VLEN=${RVVFFT_VLEN})

# RVV支持
if(RVVFFT_ENABLE_RVV)
    add_compile_definitions(RVVFFT_USE_RVV=1)
    # RISC-V向量扩展编译标志
    # GCC: -march=rv64gcv（包含V扩展）
    # Clang: -march=rv64gc_zve32f_zve64f（或更具体的V版本）
    if(CMAKE_C_COMPILER_ID MATCHES "GNU")
        add_compile_options(-march=rv64gcv -mabi=lp64d)
    elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
        add_compile_options(-march=rv64gc_zve32f -mabi=lp64d)
    endif()
    message(STATUS "RVV向量化已启用，VLEN=${RVVFFT_VLEN}")
else()
    message(STATUS "RVV向量化已禁用（标量C模式，适用于x86开发验证）")
endif()

# OpenMP（Phase 4多核并行）
if(RVVFFT_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# ── 源文件列表 ────────────────────────────────────────────────
set(RVVFFT_UTIL_SRCS
    src/utils/twiddle.c
    src/utils/bitrev.c
    src/utils/alignment.c
)

set(RVVFFT_KERNEL_SRCS
    src/kernel/handwritten/radix2_f32_interleaved.c
    src/kernel/handwritten/radix2_f32_split.c
    src/kernel/handwritten/radix4_f32.c
    src/kernel/handwritten/radix8_f32.c
)

# 收集代码生成器产出的kernel（Phase 2生成后自动加入）
file(GLOB RVVFFT_GENERATED_SRCS "src/kernel/generated/*.c")

set(RVVFFT_LAYOUT_SRCS
    src/layout/interleaved.c
    src/layout/split.c
)

set(RVVFFT_PLANNER_SRCS
    src/planner/planner.c
    src/planner/wisdom.c
)

set(RVVFFT_EXECUTOR_SRCS
    src/executor/executor_recursive.c
    src/executor/executor_iterative.c
)

# ── 主库目标 ──────────────────────────────────────────────────
add_library(rvvfft STATIC
    ${RVVFFT_UTIL_SRCS}
    ${RVVFFT_KERNEL_SRCS}
    ${RVVFFT_GENERATED_SRCS}
    ${RVVFFT_LAYOUT_SRCS}
    ${RVVFFT_PLANNER_SRCS}
    ${RVVFFT_EXECUTOR_SRCS}
)

target_include_directories(rvvfft PUBLIC include)

if(RVVFFT_ENABLE_OPENMP AND OpenMP_C_FOUND)
    target_link_libraries(rvvfft PUBLIC OpenMP::OpenMP_C)
endif()

target_link_libraries(rvvfft PUBLIC m)  # 链接libm（数学库）

# ── 测试目标 ──────────────────────────────────────────────────
if(RVVFFT_BUILD_TESTS)
    enable_testing()

    add_executable(test_correctness
        tests/test_correctness.c
    )
    target_link_libraries(test_correctness PRIVATE rvvfft)

    add_test(NAME correctness_split_n1024
             COMMAND test_correctness --n 1024)
    add_test(NAME correctness_all
             COMMAND test_correctness --all)

    message(STATUS "正确性测试已启用")
endif()

# ── Benchmark目标 ─────────────────────────────────────────────
if(RVVFFT_BUILD_BENCH)
    add_executable(bench
        bench/bench_main.c
        bench/bench_utils.c
    )
    target_link_libraries(bench PRIVATE rvvfft)
    message(STATUS "性能Benchmark已启用")
endif()

# ── 示例程序 ──────────────────────────────────────────────────
if(RVVFFT_BUILD_EXAMPLES)
    foreach(example c2c split r2c)
        add_executable(example_${example} examples/example_${example}.c)
        target_link_libraries(example_${example} PRIVATE rvvfft)
    endforeach()
    message(STATUS "示例程序已启用")
endif()

# ── 安装规则 ──────────────────────────────────────────────────
install(TARGETS rvvfft ARCHIVE DESTINATION lib)
install(FILES include/rvvfft.h include/rvvfft_types.h DESTINATION include/rvvfft)

# ── 摘要信息 ──────────────────────────────────────────────────
message(STATUS "")
message(STATUS "rvvfft 构建配置摘要:")
message(STATUS "  版本:          ${PROJECT_VERSION}")
message(STATUS "  编译器:        ${CMAKE_C_COMPILER}")
message(STATUS "  构建类型:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  VLEN:          ${RVVFFT_VLEN}位")
message(STATUS "  RVV向量化:     ${RVVFFT_ENABLE_RVV}")
message(STATUS "  OpenMP:        ${RVVFFT_ENABLE_OPENMP}")
message(STATUS "  构建测试:      ${RVVFFT_BUILD_TESTS}")
message(STATUS "  构建Benchmark: ${RVVFFT_BUILD_BENCH}")
message(STATUS "")
