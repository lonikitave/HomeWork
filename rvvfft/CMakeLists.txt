cmake_minimum_required(VERSION 3.18)
project(rvvfft C)

set(CMAKE_C_STANDARD 11)

# ── VLEN configuration ─────────────────────────────────────────────────────
set(RVVFFT_VLEN "256" CACHE STRING "RVV vector length (256 or 1024)")
set_property(CACHE RVVFFT_VLEN PROPERTY STRINGS "256" "1024")

add_compile_definitions(RVVFFT_VLEN=${RVVFFT_VLEN})

# ── Source files ───────────────────────────────────────────────────────────
file(GLOB_RECURSE KERNEL_SRC
    src/kernel/generated/*.c
    src/kernel/handwritten/*.c
)

add_library(rvvfft STATIC
    src/planner/planner.c
    src/planner/wisdom.c
    src/executor/executor_recursive.c
    src/executor/executor_iterative.c
    src/layout/interleaved.c
    src/layout/split.c
    src/utils/twiddle.c
    src/utils/alignment.c
    src/utils/bitrev.c
    ${KERNEL_SRC}
)

target_include_directories(rvvfft PUBLIC include)

# ── Benchmark ──────────────────────────────────────────────────────────────
add_executable(bench bench/bench_main.c bench/bench_utils.c)
target_link_libraries(bench rvvfft m)

# ── Correctness tests ──────────────────────────────────────────────────────
add_executable(test_correctness tests/test_correctness.c)
target_link_libraries(test_correctness rvvfft m)

enable_testing()
add_test(NAME correctness COMMAND test_correctness ${CMAKE_SOURCE_DIR}/tests/golden)
