#!/usr/bin/env python3
"""
gen_golden.py — Golden data生成脚本

使用NumPy（f64计算后截断为f32）生成FFT的期望输出，
存储为二进制文件供C测试驱动加载。

文件命名格式：golden/fft_n{N}_sign{-1|+1}.bin
文件格式：[n: uint32][sign: int32][real[n]: float32][imag[n]: float32]

运行：
    python gen_golden.py --max_n 20   # 生成2^1~2^20的golden data
    python gen_golden.py --n 1024      # 仅生成n=1024
"""
import numpy as np
import struct
import os
import argparse


GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")


def gen_golden(n: int, sign: int) -> None:
    """为指定n和sign生成golden data并保存。"""
    rng = np.random.default_rng(seed=n ^ (sign + 2))  # 确定性随机数

    # 生成f32精度的输入（用f64计算FFT，再截断，近似NumPy默认行为）
    x_r = rng.standard_normal(n).astype(np.float32)
    x_i = rng.standard_normal(n).astype(np.float32)
    x = (x_r + 1j * x_i).astype(np.complex64)

    # NumPy FFT（内部用f64计算，结果截断为f32）
    if sign == -1:
        X = np.fft.fft(x.astype(np.complex128)).astype(np.complex64)
    else:
        X = np.fft.ifft(x.astype(np.complex128)).astype(np.complex64) * n

    # 保存：文件头 + 输入 + 输出
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    filename = os.path.join(GOLDEN_DIR, f"fft_n{n}_sign{sign:+d}.bin")
    with open(filename, "wb") as f:
        f.write(struct.pack("<Ii", n, sign))   # n: uint32, sign: int32
        f.write(x_r.tobytes())                  # 输入实部
        f.write(x_i.tobytes())                  # 输入虚部
        f.write(X.real.astype(np.float32).tobytes())  # 期望输出实部
        f.write(X.imag.astype(np.float32).tobytes())  # 期望输出虚部
    print(f"生成: {filename} ({n}点, sign={sign:+d})")


def main():
    parser = argparse.ArgumentParser(description="生成FFT golden data")
    parser.add_argument("--n", type=int, default=0, help="仅生成指定n（0=生成所有）")
    parser.add_argument("--max_n", type=int, default=20,
                        help="最大2^k（默认2^20=1048576）")
    args = parser.parse_args()

    if args.n > 0:
        for sign in [-1, 1]:
            gen_golden(args.n, sign)
    else:
        for k in range(1, args.max_n + 1):
            n = 1 << k
            for sign in [-1, 1]:
                gen_golden(n, sign)

    print(f"\nGolden data已保存到 {GOLDEN_DIR}/")


if __name__ == "__main__":
    main()
