import os

import numpy as np
import scipy.io as sio


def _find_time_key(d: dict, prefer: str = "DE") -> str | None:
    """
    在 .mat 字典中自动寻找时域信号对应的 key。

    - 优先查找包含 'DE_time' 的键（驱动端 Drive End）
    - 若未找到，再查找包含 'FE_time' 的键（风扇端 Fan End）
    - 再不行，则兜底查找以 '_time' 结尾的键
    """
    prefer = prefer.upper()
    keys = [k for k in d.keys() if not k.startswith("__")]

    if prefer == "DE":
        cand = [k for k in keys if "DE_time" in k]
        if cand:
            return cand[0]
        cand = [k for k in keys if "FE_time" in k]
        if cand:
            return cand[0]
    elif prefer == "FE":
        cand = [k for k in keys if "FE_time" in k]
        if cand:
            return cand[0]
        cand = [k for k in keys if "DE_time" in k]
        if cand:
            return cand[0]

    # 兜底：任何以 _time 结尾的键
    cand = [k for k in keys if k.endswith("_time")]
    return cand[0] if cand else None


def read_and_print_b007_0() -> None:
    """
    以 B007_0.mat 为例，演示如何用 Python 读取 .mat 文件，
    并打印其中的时域振动信号部分。
    """
    # 项目根目录 = 当前文件所在目录的上一级目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 构造 B007_0.mat 的相对路径
    mat_path = os.path.join(
        project_root,
        "data",
        "CRWU",
        "12k Drive End Bearing Fault Data",
        "Ball",
        "0007",
        "B007_0.mat",
    )

    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"未找到示例文件: {mat_path}")

    # 使用 scipy.io.loadmat 读取 .mat 文件
    data = sio.loadmat(mat_path)

    print(f"成功加载文件: {mat_path}")
    print("可用键（去除内部字段 __*）:")

    # 打印所有非内部字段的 key 及其数据形状
    for key, value in data.items():
        if key.startswith("__"):
            continue
        shape = getattr(value, "shape", "")
        print(f"  - {key}: type={type(value)}, shape={shape}")

    # ===== 打印时域信号的一部分 =====
    time_key = _find_time_key(data, prefer="DE")
    if time_key is None:
        print(
            "\n未在 .mat 文件中找到形如 *_DE_time / *_FE_time / *_time 的时域键，"
            "无法演示时域数据。"
        )
        return

    # 通常 data[time_key] 是形如 (N, 1) 或 (1, N) 的二维数组，这里转成一维
    signal = np.ravel(data[time_key])

    print("\n================ 时域信号示例 ================")
    print(f"时域键名: {time_key}")
    print(f"信号长度（采样点数）: {len(signal)}")
    print("前 20 个采样点数值（单位：振动传感器输出，未做任何处理）：")
    print(signal[:20])
    print("============================================\n")

    print("解释：")
    print("  - 这个一维数组就是典型的时域振动信号序列，")
    print("    每个元素对应在固定采样频率(例如 12kHz)下采集到的一个瞬时加速度/位移值；")
    print("  - 数组下标可以看作时间轴的离散点，间隔为 1/采样频率 秒；")
    print("  - 整个数组按时间顺序排列，")
    print(
        "    在后续特征提取（时域/频域/时频域）和故障诊断模型中，都是以这个序列为基础。"
    )


if __name__ == "__main__":
    # 直接运行该脚本时，执行示例读取与打印
    read_and_print_b007_0()
