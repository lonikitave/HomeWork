# Cross-compilation toolchain for the target RISC-V chip with RVV support.
# Usage: cmake -B build -DCMAKE_TOOLCHAIN_FILE=toolchain-riscv.cmake

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Adjust these to match the installed cross-compiler on your host.
set(RISCV_TOOLCHAIN_PREFIX "riscv64-unknown-linux-gnu-" CACHE STRING
    "Cross-compiler prefix (e.g. riscv64-unknown-linux-gnu-)")

set(CMAKE_C_COMPILER   "${RISCV_TOOLCHAIN_PREFIX}gcc")
set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_PREFIX}g++")
set(CMAKE_AR           "${RISCV_TOOLCHAIN_PREFIX}ar")
set(CMAKE_RANLIB       "${RISCV_TOOLCHAIN_PREFIX}ranlib")
set(CMAKE_STRIP        "${RISCV_TOOLCHAIN_PREFIX}strip")

# rv64gcv: RVV 1.0 vector extension enabled.
# Adjust -march and -mabi to match your chip's exact supported extensions.
set(RISCV_MARCH "rv64gcv" CACHE STRING "RISC-V -march flag")
set(RISCV_MABI  "lp64d"   CACHE STRING "RISC-V -mabi flag")

set(CMAKE_C_FLAGS_INIT   "-march=${RISCV_MARCH} -mabi=${RISCV_MABI}")
set(CMAKE_CXX_FLAGS_INIT "-march=${RISCV_MARCH} -mabi=${RISCV_MABI}")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
