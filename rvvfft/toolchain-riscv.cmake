# toolchain-riscv.cmake — RISC-V交叉编译Toolchain文件
#
# 使用方法：
#   cmake -DCMAKE_TOOLCHAIN_FILE=toolchain-riscv.cmake \
#         -DRVVFFT_ENABLE_RVV=ON \
#         -DRVVFFT_VLEN=256 \
#         -B build-riscv
#
# 前提：
#   - 已安装 riscv64-unknown-linux-gnu 或 riscv64-linux-gnu 工具链
#   - 工具链支持 RVV 1.0（GCC >= 12.1 或 Clang >= 14）
#   - 或使用贵司SDK提供的专属交叉编译器

# ── 目标系统 ──────────────────────────────────────────────────
set(CMAKE_SYSTEM_NAME  Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# ── 编译器路径 ────────────────────────────────────────────────
# 优先使用环境变量 RISCV_TOOLCHAIN_PREFIX 指定前缀
if(DEFINED ENV{RISCV_TOOLCHAIN_PREFIX})
    set(RISCV_TOOLCHAIN_PREFIX $ENV{RISCV_TOOLCHAIN_PREFIX})
else()
    # 默认：标准riscv64 GNU工具链前缀
    set(RISCV_TOOLCHAIN_PREFIX "riscv64-linux-gnu")
endif()

find_program(CMAKE_C_COMPILER   NAMES "${RISCV_TOOLCHAIN_PREFIX}-gcc"
             HINTS /opt/riscv/bin /usr/bin)
find_program(CMAKE_CXX_COMPILER NAMES "${RISCV_TOOLCHAIN_PREFIX}-g++"
             HINTS /opt/riscv/bin /usr/bin)
find_program(CMAKE_AR           NAMES "${RISCV_TOOLCHAIN_PREFIX}-ar"
             HINTS /opt/riscv/bin /usr/bin)
find_program(CMAKE_RANLIB       NAMES "${RISCV_TOOLCHAIN_PREFIX}-ranlib"
             HINTS /opt/riscv/bin /usr/bin)

if(NOT CMAKE_C_COMPILER)
    message(WARNING "未找到RISC-V C编译器 '${RISCV_TOOLCHAIN_PREFIX}-gcc'。\n"
                    "请设置环境变量 RISCV_TOOLCHAIN_PREFIX 指向贵司SDK工具链前缀，\n"
                    "或在PATH中添加工具链路径。")
endif()

# ── 目标架构和ABI ─────────────────────────────────────────────
# rv64gc_zve32f: RISC-V 64位, GC基础指令集 + 最小向量扩展（Zve32f）
# 根据贵司芯片实际情况调整 -march 参数：
#   完整RVV 1.0:     -march=rv64gcv
#   仅整数向量:      -march=rv64gc_zve32x
#   贵司自定义扩展:   -march=rv64gcv_xcompanyext（假设扩展名）
if(NOT DEFINED RISCV_MARCH)
    set(RISCV_MARCH "rv64gcv")  # 完整RVV 1.0，最通用选项
endif()
set(CMAKE_C_FLAGS_INIT   "-march=${RISCV_MARCH} -mabi=lp64d")
set(CMAKE_CXX_FLAGS_INIT "-march=${RISCV_MARCH} -mabi=lp64d")

# ── Sysroot配置 ───────────────────────────────────────────────
# 若使用贵司SDK提供的sysroot，取消注释并填写路径
# set(CMAKE_SYSROOT "/path/to/company-sdk/sysroot")
# set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")

# ── 搜索策略：仅在目标sysroot中搜索库和头文件 ────────────────
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# ── QEMU仿真配置（用于CI测试） ───────────────────────────────
# 在x86机器上运行RISC-V二进制的QEMU命令
# cmake -DCMAKE_CROSSCOMPILING_EMULATOR="qemu-riscv64 -cpu rv64,v=true,vlen=256"
find_program(QEMU_RISCV64 qemu-riscv64)
if(QEMU_RISCV64)
    message(STATUS "找到QEMU: ${QEMU_RISCV64}")
    message(STATUS "可使用 CMAKE_CROSSCOMPILING_EMULATOR 配置测试仿真")
else()
    message(STATUS "未找到 qemu-riscv64，测试需要RISC-V硬件或FPGA")
endif()

message(STATUS "RISC-V Toolchain: ${RISCV_TOOLCHAIN_PREFIX}")
message(STATUS "RISC-V -march:    ${RISCV_MARCH}")
