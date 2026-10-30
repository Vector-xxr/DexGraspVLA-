#!/usr/bin/env python3
"""量顶部相机比腕部相机晚多少：开关灯（或手机闪光灯）十来次，比两路画面亮度跳变的时刻。只开相机，不碰臂和夹爪。

    cd ~/yuhang_workspace/算法/DexGraspVLA-/SO101
    PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python -m atom_robot.ar5.diag_camera_latency
    PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python -m atom_robot.ar5.diag_camera_latency --seconds 60

为什么要量：腕部 305 走 V4L2，有内核记的拍摄时刻；顶部 335 走 OBSENSOR，只有「程序收到这一帧」的时刻，
从拍摄到收到中间晚了多少看不见。OmniPicker 采集对齐时给每一行挑「最近的那张顶部图」，不扣这段延迟就会
挑偏（collect.py 的 SYNC_LATENCY）。拿臂的运动做互相关太钝（动作慢而平滑，几十毫秒的错位相关性几乎不变，
10-21 第一条真数据试过），亮度的突变才锋利。

怎么算：每次跳变，305 第一张变了的图的拍摄时刻 t1、335 第一张变了的图的收到时刻 t2，t2 − t1 = 延迟 +
两台拍摄相位造成的一项。两台都是 30 fps、相位差固定，那一项只会取两个值（相差正好一帧），落在哪个取决于
跳变发生在节拍里的位置 —— 平均为 0，**所以取平均，不能取中位数**（中位数会落到两个值之一上，离线造数据
验过：真值 25 ms，中位数 38、平均 31）。单次的散布约 ±16 ms，20 次平均约 ±4 ms。
得出的数可以直接填进 SYNC_LATENCY["cam_high"]（秒）。必须先退出 collect —— 同一台相机只能给一个程序。
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.config import AR5Config  # noqa: E402

MASTER, OTHER = "cam_front", "cam_high"
MIN_GAP_S = 0.3              # 两次跳变至少隔这么久（一次开灯可能连着两三帧都在变亮）
MATCH_S = 0.15               # 两路的跳变在这以内才算同一次


def brightness(frame):
    import cv2  # noqa: PLC0415

    small = cv2.resize(frame, (80, 60), interpolation=cv2.INTER_AREA)
    return float(small.mean())


def edges(stamps, values):
    """亮度跳变的帧：[(时刻, 方向 +1/-1)]，取每串连续变化的第一帧。"""
    t, v = np.asarray(stamps), np.asarray(values)
    d = np.diff(v)
    if len(d) == 0:
        return []
    thresh = max(3.0, 0.3 * float(np.max(np.abs(d))))
    out, last = [], -1e9
    for i in np.where(np.abs(d) > thresh)[0]:
        if t[i + 1] - last >= MIN_GAP_S:
            out.append((float(t[i + 1]), 1 if d[i] > 0 else -1))
            last = t[i + 1]
    return out


def match(master_edges, other_edges):
    """同方向、相隔 MATCH_S 以内的配成一对，返回 t_other − t_master（秒）。"""
    diffs = []
    for t1, sign in master_edges:
        cands = [t2 - t1 for t2, s in other_edges if s == sign and abs(t2 - t1) <= MATCH_S]
        if cands:
            diffs.append(min(cands, key=abs))
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60.0, help="录多久，默认 60 秒")
    args = ap.parse_args()

    from atom_robot.ar5.drivers.cameras import OpenCVCameras  # noqa: PLC0415

    cfg = AR5Config().cameras
    if not {MASTER, OTHER} <= set(cfg.sources):
        print(f"  config 里要有 {MASTER} 和 {OTHER} 两路，现在是 {list(cfg.sources)}")
        return 1
    cams = OpenCVCameras(cfg)
    cams.connect()
    try:
        kinds = cams.stamp_kinds()
        print(f"  时刻：{MASTER} 用{'内核拍摄时刻' if kinds[MASTER] == 'kernel' else '收到时刻'}、"
              f"{OTHER} 用{'内核拍摄时刻' if kinds[OTHER] == 'kernel' else '收到时刻'}")
        print(f"\n  开始 —— {args.seconds:.0f} 秒内**开关灯 20 次左右**（或拿手机闪光灯照桌面、开一下关一下），")
        print("  两台相机都要看得见变化；开和关之间隔一秒左右。\n")
        series = {MASTER: ([], []), OTHER: ([], [])}
        seen = {MASTER: None, OTHER: None}
        start = time.perf_counter()
        next_note = 5.0
        while (elapsed := time.perf_counter() - start) < args.seconds:
            for name in (MASTER, OTHER):
                new = cams.frames_since(name, seen[name])
                if new:
                    seen[name] = new[-1][0]
                    for _, stamp, frame in new:
                        series[name][0].append(stamp)
                        series[name][1].append(brightness(frame))
            if elapsed >= next_note:
                print(f"  …… 还剩 {args.seconds - elapsed:.0f} 秒", flush=True)
                next_note += 5.0
            time.sleep(0.005)
    finally:
        cams.disconnect()

    for name, (t, _) in series.items():
        if len(t) > 1:
            print(f"  {name}: {len(t)} 帧，{(len(t) - 1) / (t[-1] - t[0]):.1f} fps")
    e1, e2 = edges(*series[MASTER]), edges(*series[OTHER])
    print(f"  检测到亮度跳变：{MASTER} {len(e1)} 次、{OTHER} {len(e2)} 次")
    diffs = np.asarray(match(e1, e2)) * 1000
    if len(diffs) < 5:
        print(f"\n  ✗ 只配上 {len(diffs)} 次，太少了。灯开关得更明显一点、多来几次再跑。")
        return 1
    # 离群的（配错了的）去掉：离中位数超过一帧半
    keep = diffs[np.abs(diffs - np.median(diffs)) <= 50]
    mean = float(keep.mean())
    print(f"\n  配上 {len(diffs)} 次：{OTHER} 比 {MASTER} 晚 " + " ".join(f"{d:+.0f}" for d in diffs) + " ms")
    print(f"  平均 {mean:+.1f} ms（用了 {len(keep)} 次；单次散布 {keep.std():.0f} ms，平均下来约 "
          f"±{keep.std() / np.sqrt(len(keep)):.0f} ms）。单次只会落在相差一帧的两个值上，看平均，别看中位数")
    print(f"\n  → collect.py 里填 SYNC_LATENCY = {{\"{OTHER}\": {mean / 1000:.3f}}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
