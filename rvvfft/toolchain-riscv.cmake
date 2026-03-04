# toolchain-riscv.cmake — CMake toolchain file for self-developed RISC-V RVV chips.
#
# Usage (from the rvvfft/ directory):
#   cmake -B build -DCMAKE_TOOLCHAIN_FILE=../toolchain-riscv.cmake
#
# Adjust the TOOLCHAIN_PREFIX and sysroot to match your SDK installation.
# Common locations for RISC-V cross toolchains:
#   - Vendor SDK:  /opt/vendor-riscv-sdk/
#   - riscv-gnu-toolchain (upstream): /opt/riscv/
#   - Bootlin toolchain: /opt/bootlin/riscv64-lp64d-*/

cmake_minimum_required(VERSION 3.22)

# ─── Toolchain prefix ─────────────────────────────────────────────────────────
# Change this to match your SDK.  Examples:
#   riscv64-unknown-linux-gnu-       (upstream riscv-gnu-toolchain)
#   riscv64-linux-gnu-               (Debian/Ubuntu cross toolchain)
#   riscv64-vendor-elf-              (bare-metal vendor toolchain)
set(TOOLCHAIN_PREFIX "riscv64-unknown-linux-gnu-"
    CACHE STRING "Cross-compiler prefix (e.g. riscv64-unknown-linux-gnu-)")

# ─── System description ───────────────────────────────────────────────────────
set(CMAKE_SYSTEM_NAME      Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# ─── Compilers ────────────────────────────────────────────────────────────────
find_program(CMAKE_C_COMPILER   NAMES ${TOOLCHAIN_PREFIX}gcc
             DOC "RISC-V C cross-compiler")
find_program(CMAKE_CXX_COMPILER NAMES ${TOOLCHAIN_PREFIX}g++
             DOC "RISC-V C++ cross-compiler")
find_program(CMAKE_ASM_COMPILER NAMES ${TOOLCHAIN_PREFIX}gcc
             DOC "RISC-V assembler")

if(NOT CMAKE_C_COMPILER)
    message(FATAL_ERROR
        "RISC-V cross compiler '${TOOLCHAIN_PREFIX}gcc' not found.\n"
        "Set TOOLCHAIN_PREFIX to the correct prefix, or put the toolchain "
        "binaries in your PATH.")
endif()

# ─── Sysroot ──────────────────────────────────────────────────────────────────
# Set CMAKE_SYSROOT to the vendor SDK sysroot if available.
# If left empty, the toolchain's built-in sysroot is used.
set(RISCV_SYSROOT "" CACHE PATH "Sysroot for the target RISC-V system")
if(RISCV_SYSROOT)
    set(CMAKE_SYSROOT ${RISCV_SYSROOT})
endif()

# ─── Binutils ─────────────────────────────────────────────────────────────────
find_program(CMAKE_AR       ${TOOLCHAIN_PREFIX}ar)
find_program(CMAKE_RANLIB   ${TOOLCHAIN_PREFIX}ranlib)
find_program(CMAKE_STRIP    ${TOOLCHAIN_PREFIX}strip)
find_program(CMAKE_OBJDUMP  ${TOOLCHAIN_PREFIX}objdump)

# ─── Target ISA flags ─────────────────────────────────────────────────────────
# -march=rv64gcv  : RISC-V 64-bit base + G + Compressed + Vector extension
# -mabi=lp64d     : LP64 ABI with double-precision floating point
# -mcpu=           : Optional vendor-specific CPU target (set if known)
set(RISCV_MARCH "rv64gcv" CACHE STRING "RISC-V march string")
set(RISCV_MABI  "lp64d"   CACHE STRING "RISC-V mabi string")

set(CMAKE_C_FLAGS_INIT   "-march=${RISCV_MARCH} -mabi=${RISCV_MABI}")
set(CMAKE_CXX_FLAGS_INIT "-march=${RISCV_MARCH} -mabi=${RISCV_MABI}")

# ─── Linker ───────────────────────────────────────────────────────────────────
# For bare-metal targets you may need:
#   set(CMAKE_EXE_LINKER_FLAGS_INIT "-nostdlib -T <linker-script>")
# For Linux targets this is usually not needed.

# ─── Finder hints ─────────────────────────────────────────────────────────────
# Prevent CMake from finding host libraries/includes in the cross-compiled tree.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# ─── QEMU runner (for CTest on a non-RISC-V host) ────────────────────────────
# If QEMU_RISCV64 is set, CTest uses it to run RISC-V binaries on the host.
find_program(QEMU_RISCV64 NAMES qemu-riscv64 qemu-riscv64-static)
if(QEMU_RISCV64)
    message(STATUS "Found QEMU: ${QEMU_RISCV64} — CTest will use it to run RVV binaries")
    set(CMAKE_CROSSCOMPILING_EMULATOR "${QEMU_RISCV64};-L;${RISCV_SYSROOT}")
endif()

message(STATUS "Cross-compiler : ${CMAKE_C_COMPILER}")
message(STATUS "march / mabi   : ${RISCV_MARCH} / ${RISCV_MABI}")
if(RISCV_SYSROOT)
    message(STATUS "Sysroot        : ${RISCV_SYSROOT}")
endif()
