#!/usr/bin/env python3
"""比较多轮遥操轨迹，出图和汇总，全部落到一个目录里。

    python -m atom_robot.ar5.analyse_runs ~/screw_run*.npz --out ~/桌面/珞石测试

纯离线：只读 .npz，不碰机器人、不连控制器。

轨迹列的定义见 `teleop.py` 的 docstring。认两种布局：

    53 列  旧格式：关节力矩 + 本地 pinv(J^T) 反推
    66 列  新格式：多了控制器自己算的**关节空间外力估计** external_torque
           和**末端六维力估计** cart_force / cart_torque

列数对不上会直接报错，而不是静默画出错的图。两种混着给也行，估计那几项在旧格式
上显示 "-"。

**所有力都扣接合前的基线再看。** `payload_mass` 是 0 而手加工具约 1.2~1.3 kg，
那份重力在原始力矩、`external_torque`、`cart_force` 里**都没被扣掉**（实测空载
末端 Z 向 -18 N 就是它），所以绝对值一律不可用，只有变化量有意义。

66 列的数据里，控制器的末端力估计和本地 `pinv(J^T)·tau` **来路不同**，所以第 6
张图那个对比是真的交叉验证。而 53 列时只有本地一路，它和关节力矩同源，"对得上"
是必然的、不构成验证。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom_robot.ar5.config import ArmConfig  # noqa: E402

N_JOINTS = 7
RSC_J7_LIMIT = 17.0          # 从控制器日志 id=30400 解出来的，只有 J7 触发过

# 两种布局都认：53 列是只有关节力矩 + 本地反推的旧格式，66 列多了控制器自己算的
# 外力估计和末端六维力。混着比的时候只有两边都有的列能比，`has_est` 标出来。
LAYOUT = {
    53: {"tau": slice(40, 47), "jac": slice(47, 53), "has_est": False},
    66: {"tau": slice(40, 47), "ext": slice(47, 54), "cforce": slice(54, 57),
         "ctorque": slice(57, 60), "jac": slice(60, 66), "has_est": True},
}
COL = {
    "t": 0, "engaged": 1, "leader": slice(2, 5), "action": slice(5, 12),
    "pose": slice(12, 18), "q_goal": slice(18, 25), "q_cmd": slice(25, 32),
    "q_actual": slice(32, 39), "sent": 39,
}


def load(path: Path) -> dict:
    a = np.load(path)["trace"]
    lay = LAYOUT.get(a.shape[1])
    if lay is None:
        raise SystemExit(
            f"{path.name} 有 {a.shape[1]} 列，只认 {sorted(LAYOUT)}。"
            "列的含义见 teleop.py 的 docstring。"
        )
    t = a[:, COL["t"]] - a[0, COL["t"]]
    eng = a[:, COL["engaged"]] > 0.5
    # 基线取接合之前那一段：手是空载的，重力和零漂都在里面。
    # 没有接合前的数据就退回用最前面 1 秒，并在汇总里标出来。
    if eng.any() and (~eng[: np.argmax(eng)]).sum() > 5:
        base = np.zeros(len(a), bool)
        base[: np.argmax(eng)] = True
    else:
        base = t < 1.0
    r = {
        "name": path.stem, "path": path, "t": t, "eng": eng, "base": base,
        "pose": a[:, COL["pose"]], "q_goal": a[:, COL["q_goal"]],
        "q_cmd": a[:, COL["q_cmd"]], "q_actual": a[:, COL["q_actual"]],
        "sent": a[:, COL["sent"]], "tau": a[:, lay["tau"]],
        # **取反**：统一到控制器的约定。
        # SDK 文档说 `cart_force` 是「末端**受到**的力」，而 `pinv(J^T)·tau` 算出来
        # 的是臂**施加**的力 —— 同一个物理量、反号。2026-09-17 实测：末端压到最低
        # 点时控制器报 +73.81 N、本地报 −65.18 N，六个分量的相关系数全是 −0.95~−0.995。
        # 取反之后 r 变成 +0.95~+0.995、RMS 差从 40~51 N 掉到 5.6 N。
        # 统一之后：**Fz 为正 = 在往下压**。
        "jac": -a[:, lay["jac"]], "has_est": lay["has_est"], "raw": a,
    }
    if lay["has_est"]:
        r["ext"] = a[:, lay["ext"]]
        # 控制器的末端估计拼成一个六维力旋量，和本地反推那一路同格式好对比
        r["est"] = np.hstack([a[:, lay["cforce"]], a[:, lay["ctorque"]]])
    return r


_KIN = [None]


def kinematics():
    """本地正解，用缓存的 DH 表。只算一次。"""
    if _KIN[0] is None:
        import json

        from atom_robot.ar5.kinematics import AR5Kinematics

        cfg = ArmConfig()
        dh = json.loads((Path(__file__).resolve().parent / "ar5_dh.json").read_text())["dh"]
        _KIN[0] = AR5Kinematics(np.array(dh).reshape(-1, 4), cfg.num_joints,
                                cfg.joint_min, cfg.joint_max)
    return _KIN[0]


def tracking(r: dict) -> dict:
    """接合期间的跟踪精度。

    关节误差用 `q_goal − q_actual`（上层要的 vs 实际到的），不是 `q_cmd − q_actual`
    —— 后者只反映伺服环，把跟踪器自己的平滑给漏掉了。

    位姿误差按需求表的定义算：**ξ = Log(T_d^-1 · T)^v**，其中 T_d 是 `q_goal` 的正解。
    拆成位置（mm）和姿态（deg）两路报，因为它们的量纲不同、合成一个数没有意义。
    """
    from atom_robot.ar5.transforms import matrix_to_axis_angle

    i = np.where(r["eng"])[0]
    err = r["q_goal"][i] - r["q_actual"][i]
    kin = kinematics()
    pos, rot = [], []
    for k in i:
        Td = kin.fk(r["q_goal"][k])
        T = kin.fk(r["q_actual"][k])
        # T_d⁻¹·T，取平移部分和旋转部分
        Rd, Ra = Td[:3, :3], T[:3, :3]
        pos.append(np.linalg.norm(Rd.T @ (T[:3, 3] - Td[:3, 3])))
        rot.append(np.linalg.norm(matrix_to_axis_angle(Rd.T @ Ra)))
    a = np.abs(err)
    # **把误差拆开：是跟踪器在限速，还是伺服跟不上？**
    #   q_goal − q_cmd  跟踪器自己的限速/平滑（max_joint_speed 那一道）
    #   q_cmd − q_actual 伺服环真正的跟随误差
    # 实测前者占总误差的 92~93%（p95），所以"跟踪精度差"基本都是限速的结果、
    # 不是臂跟不上。不拆开看会把设计行为误判成缺陷。
    lag = np.abs(r["q_goal"][i] - r["q_cmd"][i])
    servo = np.abs(r["q_cmd"][i] - r["q_actual"][i])
    # 慢速段 = 关节速度最低的 25%，接近静态精度。快速段的误差主要是限速滞后，
    # 谈"精度"通常指的是前者。
    t = r["t"][i]
    v = np.abs(np.gradient(r["q_actual"][i], t, axis=0)).max(axis=1)
    slow = v < np.percentile(v, 25)
    return {
        "i": i, "err": err,
        "p50": np.percentile(a, 50, axis=0),
        "p95": np.percentile(a, 95, axis=0),
        "pmax": a.max(axis=0),
        "lag": lag, "servo": servo, "slow": slow,
        "pos_err": np.array(pos), "rot_err": np.array(rot),
    }


def summarise(r: dict) -> dict:
    t, eng, base = r["t"], r["eng"], r["base"]
    i = np.where(eng)[0]
    dt = np.diff(t)
    # 四路估计全部扣接合前的基线：`payload_mass=0` 而手+工具约 1.2~1.3 kg，
    # 那份重力在原始力矩、external_torque、cart_force 里**都没被扣掉**，
    # 所以绝对值一律不可用，只有变化量有意义。
    tau0 = r["tau"][base].mean(0)
    dtau = r["tau"] - tau0
    dwr = r["jac"] - r["jac"][base].mean(0)          # 本地反推
    dext = (r["ext"] - r["ext"][base].mean(0)) if r["has_est"] else None
    dest = (r["est"] - r["est"][base].mean(0)) if r["has_est"] else None
    cfg = ArmConfig()
    margins = [
        float(np.degrees(min((r["q_actual"][i, j] - cfg.joint_min[j]).min(),
                             (cfg.joint_max[j] - r["q_actual"][i, j]).min())))
        for j in range(N_JOINTS)
    ] if len(i) else [float("nan")] * N_JOINTS
    j7 = np.abs(r["tau"][i, 6]) if len(i) else np.array([0.0])
    out = {
        "轨迹": r["name"],
        "列数": int(r["raw"].shape[1]),
        "步数": int(len(t)),
        "时长_s": round(float(t[-1]), 1),
        "接合步数": int(eng.sum()),
        "基线帧数": int(base.sum()),
        "步长中位_ms": round(float(np.median(dt) * 1000), 1),
        "步长最大_ms": round(float(dt.max() * 1000), 1),
        "RT发送率_每秒": round(float((r["sent"][-1] - r["sent"][0]) / t[-1])),
        "跟踪误差最大_deg": round(float(np.degrees(
            np.abs(r["q_cmd"][i] - r["q_actual"][i]).max())), 3) if len(i) else None,
        "末端行程_mm": round(float(np.linalg.norm(
            np.diff(r["pose"][i, :3], axis=0), axis=1).sum() * 1000), 1) if len(i) else 0.0,
        "末端z最低_mm": round(float(r["pose"][i, 2].min() * 1000), 1) if len(i) else None,
        "J7力矩峰值_Nm": round(float(j7.max()), 2),
        "J7占RSC限_pct": round(float(j7.max() / RSC_J7_LIMIT * 100), 1),
        "J7超17的帧数": int((j7 > RSC_J7_LIMIT).sum()),
        "J7行程余量最小_deg": round(margins[6], 2),
        "下压力峰值_N": round(float(dwr[i, 2].max()), 1) if len(i) else None,
        "倾覆力矩Mx峰值_Nm": round(float(np.abs(dwr[i, 3]).max()), 2) if len(i) else None,
        "_dtau": dtau, "_dwr": dwr, "_dext": dext, "_dest": dest,
        "_track": tracking(r) if len(i) else None,
    }
    tr = out["_track"]
    if tr is not None:
        out.update({
            "跟踪误差p95最差轴": f"J{int(np.argmax(tr['p95'])) + 1}",
            "跟踪误差p95_deg": round(float(np.degrees(tr["p95"].max())), 4),
            "跟踪误差中位_deg": round(float(np.degrees(np.median(tr["p50"]))), 4),
            "位置误差p95_mm": round(float(np.percentile(tr["pos_err"], 95) * 1000), 3),
            "位置误差最大_mm": round(float(tr["pos_err"].max() * 1000), 3),
            "姿态误差p95_deg": round(float(np.degrees(np.percentile(tr["rot_err"], 95))), 4),
            "跟踪器滞后p95_deg": round(float(np.degrees(np.percentile(tr["lag"], 95))), 4),
            "伺服误差p95_deg": round(float(np.degrees(np.percentile(tr["servo"], 95))), 4),
            "伺服误差最大_deg": round(float(np.degrees(tr["servo"].max())), 4),
            "慢速段总误差p95_deg": round(float(np.degrees(
                np.percentile(np.abs(tr["err"])[tr["slow"]], 95))), 5),
            "滞后占总误差_pct": round(float(
                np.percentile(tr["lag"], 95) / max(np.percentile(np.abs(tr["err"]), 95), 1e-12)
                * 100), 1),
        })
    if r["has_est"] and len(i):
        out.update({
            "外力估计峰值_Nm": round(float(np.abs(dext[i]).max()), 2),
            "外力峰值在哪轴": f"J{int(np.abs(dext[i]).max(0).argmax()) + 1}",
            "控制器末端力峰值_N": round(float(np.linalg.norm(dest[i, :3], axis=1).max()), 1),
            "控制器下压力峰值_N": round(float(dest[i, 2].max()), 1),
            "控制器末端力矩峰值_Nm": round(float(np.linalg.norm(dest[i, 3:], axis=1).max()), 2),
            # 控制器的估计 vs 本地 pinv(J^T) 反推：来路不同，这才是真的交叉验证
            # 两路已经统一到同一个符号约定（见 load），所以这里是真的一致性检验
            "两路Fz相关系数": round(float(np.corrcoef(dest[i, 2], dwr[i, 2])[0, 1]), 3),
            "两路Fz_RMS差_N": round(float(np.sqrt(((dest[i, 2] - dwr[i, 2]) ** 2).mean())), 2),
            "两路Fz幅度比_本地比控制器": round(float(
                np.std(dwr[i, 2]) / max(np.std(dest[i, 2]), 1e-9)), 3),
            "两路Mx相关系数": round(float(np.corrcoef(dest[i, 3], dwr[i, 3])[0, 1]), 3),
        })
    return out


def plot_all(runs, summaries, out: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    for cand in ("Noto Sans CJK JP", "WenQuanYi Micro Hei", "Droid Sans Fallback"):
        try:
            matplotlib.rcParams["font.sans-serif"] = [cand]
            break
        except Exception:  # noqa: BLE001
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False

    written = []
    colours = plt.cm.tab10.colors

    def engaged_spans(r):
        e = r["eng"].astype(int)
        edges = np.diff(np.concatenate([[0], e, [0]]))
        return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1))

    # 1) J7 力矩 vs RSC 限 —— 这一路决定臂停不停机，单独一张
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k, (r, s) in enumerate(zip(runs, summaries)):
        ax.plot(r["t"], np.abs(r["tau"][:, 6]), lw=1.1, color=colours[k % 10],
                label=f'{r["name"]}  峰值 {s["J7力矩峰值_Nm"]:.2f} Nm')
    ax.axhline(RSC_J7_LIMIT, color="crimson", ls="--", lw=1.6,
               label=f"RSC 限定 {RSC_J7_LIMIT} Nm（超过就安全停止、掉电抱闸）")
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("|J7 关节力矩| (Nm)")
    ax.set_title("J7 关节力矩 vs RSC 碰撞力限制　（一手数据，未扣基线）")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    p = out / "1_J7力矩_vs_RSC限制.png"; fig.tight_layout(); fig.savefig(p, dpi=150)
    plt.close(fig); written.append(p)

    # 2) 每轮一张：末端 z、下压力、倾覆力矩、J7 —— 打螺丝那一下四条曲线要对齐着看
    for k, (r, s) in enumerate(zip(runs, summaries)):
        fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
        dwr, dtau = s["_dwr"], s["_dtau"]
        for ax, y, lab, unit in (
            (axes[0], r["pose"][:, 2] * 1000, "末端高度 z", "mm"),
            (axes[1], dwr[:, 2], "下压力 ΔFz", "N"),
            (axes[2], dwr[:, 3], "倾覆力矩 ΔMx", "Nm"),
            (axes[3], dtau[:, 6], "J7 力矩 Δ", "Nm"),
        ):
            ax.plot(r["t"], y, lw=1.1, color=colours[k % 10])
            ax.set_ylabel(f"{lab}\n({unit})", fontsize=9)
            ax.grid(alpha=.3)
            for lo, hi in engaged_spans(r):
                ax.axvspan(r["t"][lo], r["t"][hi], color="tab:green", alpha=.08)
        axes[3].set_xlabel("时间 (s)　　绿底 = 离合接合")
        axes[0].set_title(f'{r["name"]}　打螺丝过程　（力已扣接合前基线；'
                          f'末端力是 pinv(J^T)·τ 反推，绝对值未标定）')
        p = out / f'2_{r["name"]}_打螺丝过程.png'
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); written.append(p)

    # 3) 七轴力矩对比，看载荷都压在谁身上
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4.2), squeeze=False)
    for k, (r, s) in enumerate(zip(runs, summaries)):
        ax = axes[0][k]
        i = np.where(r["eng"])[0]
        pk = [float(np.abs(s["_dtau"][i, j]).max()) for j in range(N_JOINTS)]
        ax.bar([f"J{j+1}" for j in range(N_JOINTS)], pk, color=colours[k % 10])
        ax.set_title(r["name"], fontsize=10); ax.grid(alpha=.3, axis="y")
        ax.set_ylabel("接合期间 |Δ力矩| 峰值 (Nm)" if k == 0 else "")
    fig.suptitle("载荷分布：打螺丝时哪几个关节在出力")
    p = out / "3_七轴力矩峰值.png"; fig.tight_layout(); fig.savefig(p, dpi=150)
    plt.close(fig); written.append(p)

    # 4) 跟踪质量 + 通信 —— 上次臂"不动"就是从这两条看出来的
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for k, r in enumerate(runs):
        err = np.degrees(np.abs(r["q_cmd"] - r["q_actual"]).max(axis=1))
        axes[0].plot(r["t"], err, lw=1.0, color=colours[k % 10], label=r["name"])
        dt = np.diff(r["t"])
        axes[1].plot(r["t"][1:], np.diff(r["sent"]) / (dt * 1000),
                     lw=1.0, color=colours[k % 10], label=r["name"])
    axes[0].set_ylabel("跟踪误差 max|q_cmd−q_actual| (°)")
    axes[0].set_title("跟踪质量与通信　（跟踪误差涨上去 = 臂没在执行）")
    axes[1].axhline(1.0, color="grey", ls=":", lw=1)
    axes[1].set_ylabel("每毫秒发出的目标数\n(理想 1.0)"); axes[1].set_xlabel("时间 (s)")
    for ax in axes: ax.grid(alpha=.3); ax.legend(fontsize=8)
    p = out / "4_跟踪误差与通信.png"; fig.tight_layout(); fig.savefig(p, dpi=150)
    plt.close(fig); written.append(p)

    # 7) 跟踪精度：需求表要的是「误差谱、p95」和位姿误差 ξ=Log(T_d⁻¹T)^∨
    for k, (r, s) in enumerate(zip(runs, summaries)):
        tr = s.get("_track")
        if tr is None:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        i = tr["i"]
        t = r["t"][i]

        ax = axes[0][0]
        for j in range(N_JOINTS):
            ax.plot(t, np.degrees(tr["err"][:, j]), lw=.8, label=f"J{j+1}")
        ax.set_ylabel("q_goal − q_actual (°)"); ax.set_xlabel("时间 (s)")
        ax.set_title("逐轴跟踪误差（接合期间）", fontsize=10)
        ax.legend(fontsize=7, ncol=7); ax.grid(alpha=.3)

        # 误差来源分解：跟踪器限速 vs 伺服环。不拆开会把设计行为当成缺陷。
        ax = axes[0][1]
        labels = ["跟踪器滞后\nq_goal−q_cmd", "伺服误差\nq_cmd−q_actual",
                  "总误差\nq_goal−q_actual", "慢速段\n总误差"]
        vals50, vals95, valsmax = [], [], []
        for arr in (tr["lag"], tr["servo"], np.abs(tr["err"]),
                    np.abs(tr["err"])[tr["slow"]]):
            vals50.append(np.degrees(np.percentile(arr, 50)))
            vals95.append(np.degrees(np.percentile(arr, 95)))
            valsmax.append(np.degrees(arr.max()))
        xs = np.arange(len(labels)); w = 0.27
        ax.bar(xs - w, vals50, w, label="中位")
        ax.bar(xs, vals95, w, label="p95")
        ax.bar(xs + w, valsmax, w, label="最大")
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("|误差| (°)"); ax.set_yscale("log")
        ax.set_title(f"误差来源分解　滞后占 p95 的 "
                     f"{vals95[0]/max(vals95[2],1e-12)*100:.0f}%", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")

        ax = axes[1][0]
        ax.plot(t, tr["pos_err"] * 1000, lw=.9, color="tab:blue", label="位置 (mm)")
        ax.set_ylabel("位置误差 (mm)"); ax.set_xlabel("时间 (s)")
        ax2 = ax.twinx()
        ax2.plot(t, np.degrees(tr["rot_err"]), lw=.9, color="tab:orange", label="姿态 (°)")
        ax2.set_ylabel("姿态误差 (°)")
        ax.set_title("末端位姿误差 ξ = Log(T_d^-1 · T)^v　（T_d 由 q_goal 正解）",
                     fontsize=10)
        ax.grid(alpha=.3)
        ln1, lb1 = ax.get_legend_handles_labels()
        ln2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(ln1 + ln2, lb1 + lb2, fontsize=8)

        # 误差谱：需求表点名要的。峰值落在哪个频率能看出是抖动还是系统性滞后
        ax = axes[1][1]
        dt = float(np.median(np.diff(t)))
        for j in range(N_JOINTS):
            e = np.degrees(tr["err"][:, j])
            e = e - e.mean()
            spec = np.abs(np.fft.rfft(e * np.hanning(len(e)))) / len(e)
            freq = np.fft.rfftfreq(len(e), dt)
            ax.semilogy(freq[1:], spec[1:], lw=.8, label=f"J{j+1}")
        ax.set_xlabel("频率 (Hz)"); ax.set_ylabel("幅度 (°)")
        ax.set_title(f"跟踪误差谱（采样 {1/dt:.0f} Hz）", fontsize=10)
        ax.legend(fontsize=7, ncol=7); ax.grid(alpha=.3)

        fig.suptitle(
            f'{r["name"]}　跟踪精度　'
            f'伺服误差 p95 {np.degrees(np.percentile(tr["servo"], 95)):.3f}°　'
            f'慢速段总误差 p95 {np.degrees(np.percentile(np.abs(tr["err"])[tr["slow"]], 95)):.4f}°　'
            f'位置误差 p95 {np.percentile(tr["pos_err"], 95)*1000:.2f} mm')
        p = out / f'7_{r["name"]}_跟踪精度.png'
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); written.append(p)

    est_runs = [(r, s) for r, s in zip(runs, summaries) if r["has_est"]]
    if not est_runs:
        return written

    # 5) 关节空间外力估计 —— 控制器扣掉动力学模型之后的各轴外部力矩
    fig, axes = plt.subplots(len(est_runs), 1, figsize=(11, 3.1 * len(est_runs)),
                             sharex=False, squeeze=False)
    for k, (r, s) in enumerate(est_runs):
        ax = axes[k][0]
        for j in range(N_JOINTS):
            ax.plot(r["t"], s["_dext"][:, j], lw=.9, label=f"J{j+1}")
        for lo, hi in engaged_spans(r):
            ax.axvspan(r["t"][lo], r["t"][hi], color="tab:green", alpha=.08)
        ax.set_ylabel("Δ外部力矩 (Nm)", fontsize=9); ax.grid(alpha=.3)
        ax.set_title(f'{r["name"]}　关节空间外力估计（external_torque，已扣基线）',
                     fontsize=10)
        if k == 0: ax.legend(fontsize=7, ncol=7, loc="upper right")
    axes[-1][0].set_xlabel("时间 (s)　　绿底 = 离合接合")
    p = out / "5_关节空间外力估计.png"; fig.tight_layout(); fig.savefig(p, dpi=150)
    plt.close(fig); written.append(p)

    # 6) 控制器的末端六维力估计 vs 本地 pinv(J^T) 反推。
    #    **两者来路不同**（控制器的动力学模型 vs 伪逆），所以这是真的交叉验证；
    #    同源的两列对得上是必然的，不算验证。
    fig, axes = plt.subplots(2, len(est_runs), figsize=(5.4 * len(est_runs), 7),
                             squeeze=False)
    for k, (r, s) in enumerate(est_runs):
        ax = axes[0][k]
        ax.plot(r["t"], s["_dest"][:, 2], lw=1.0, label="控制器 cart_force")
        ax.plot(r["t"], s["_dwr"][:, 2], lw=1.0, ls="--", label="本地 −pinv(J^T)·τ")
        ax.set_title(f'{r["name"]}　下压力：两路估计', fontsize=10)
        ax.set_ylabel("ΔFz (N)　正=下压" if k == 0 else ""); ax.grid(alpha=.3)
        ax.legend(fontsize=8); ax.set_xlabel("时间 (s)")
        ax2 = axes[1][k]
        i = np.where(r["eng"])[0]
        ax2.scatter(s["_dest"][i, 2], s["_dwr"][i, 2], s=3, alpha=.3)
        lim = np.abs(np.concatenate([s["_dest"][i, 2], s["_dwr"][i, 2]])).max() * 1.05
        ax2.plot([-lim, lim], [-lim, lim], color="crimson", lw=1, ls=":")
        ax2.set_xlabel("控制器 ΔFz (N)")
        ax2.set_ylabel("本地反推 ΔFz (N)" if k == 0 else "")
        ax2.set_title(f'r = {s.get("两路Fz相关系数", float("nan")):.3f}　'
                      f'RMS 差 {s.get("两路Fz_RMS差_N", float("nan")):.2f} N　'
                      f'幅度比 {s.get("两路Fz幅度比_本地比控制器", float("nan")):.2f}',
                      fontsize=9)
        ax2.grid(alpha=.3); ax2.set_aspect("equal", "box")
    fig.suptitle("末端六维力：控制器估计 vs 本地反推（来路不同，这才算交叉验证）")
    p = out / "6_末端力两路对比.png"; fig.tight_layout(); fig.savefig(p, dpi=150)
    plt.close(fig); written.append(p)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="teleop --log-actions 存下来的 .npz")
    ap.add_argument("--out", default=str(Path.home() / "桌面" / "珞石测试"))
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    runs = [load(Path(p).expanduser()) for p in sorted(args.traces)]
    summaries = [summarise(r) for r in runs]

    # 键取所有轨迹的并集：53 列的没有估计那几项，缺的补 "-"，别让它们整列消失
    keys = []
    for s in summaries:
        for k in s:
            if not k.startswith("_") and k not in keys:
                keys.append(k)
    width = max(len(k) for k in keys) + 2
    print(f"\n  {'':<{width}}" + "".join(f'{s["轨迹"]:>16}' for s in summaries))
    for k in keys[1:]:
        print(f"  {k:<{width}}" + "".join(f'{str(s.get(k, "-")):>16}' for s in summaries))

    # 用最长的那份的键做表头，缺的补 "-"（53 列的没有估计那几项）
    # 原始数据也拷进去，图和数据放一起才能复现
    for r in runs:
        dest = out / r["path"].name
        if dest.resolve() != r["path"].resolve():
            dest.write_bytes(r["path"].read_bytes())

    (out / "汇总.json").write_text(json.dumps(
        [{k: v for k, v in s.items() if not k.startswith("_")} for s in summaries],
        ensure_ascii=False, indent=2), encoding="utf-8")

    # 每轮一份 CSV，方便贴到别处画图
    for r, s in zip(runs, summaries):
        i = np.where(r["eng"])[0] if r["eng"].any() else np.arange(len(r["t"]))
        cols = np.column_stack([
            r["t"][i], r["pose"][i, 2] * 1000, r["tau"][i, 6],
            s["_dtau"][i, 6], -s["_dwr"][i, 2], s["_dwr"][i, 3],
        ])
        np.savetxt(out / f'{r["name"]}.csv', cols, delimiter=",", fmt="%.5f",
                   header="t_s,ee_z_mm,J7_tau_Nm,J7_dtau_Nm,press_N,tilt_Mx_Nm",
                   comments="")

    written = plot_all(runs, summaries, out)
    print(f"\n  写到 {out}")
    for p in sorted(out.iterdir()):
        print(f"    {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
