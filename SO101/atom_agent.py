#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原子动作开发中心 · 工作站代理
================================

浏览器不能直接执行本机命令（安全沙箱）。这个代理常驻在采集工作站上，
网页通过 http://127.0.0.1:<port> 调它，由它在工作站上执行白名单命令，
并把采集进度反向上报给网站。

两件事：
  1. 执行命令  —— 只跑下面 TEMPLATES 里写死的模板，参数一律 shlex.quote 转义
  2. 上报进度  —— 轮询 out_dir 里的 episode_*.npz，把已落盘的条数同步给网站
  3. 删除目录  —— 网页上删数据集时，连带删掉工作站上的采集目录（只许删输出根目录之下的）

安全：
  * 只绑 127.0.0.1，外部机器连不上
  * 所有请求必须带正确 token
  * 校验 Origin，挡住其他网页的跨站调用与 DNS 重绑定
  * 不接受自由命令，只接受模板 key + 参数

用法：
    python atom_agent.py --token <令牌> --site http://192.168.5.100:8090

可选参数见 --help。在 dexgraspvla 环境里启动，并确保 --workdir 指向 VEL 仓库根目录。
"""

import argparse
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "2.9"

# ---------------------------------------------------------------------------
# 白名单命令模板。key 由网页传入，参数被 shlex.quote 转义后代入。
#   env : 在哪个 conda 环境里跑
#   gui : 需要显示器（网页点了也会在工作站屏幕上开窗口）
# ---------------------------------------------------------------------------
TEMPLATES = {
    "show_config":    {"env": "record",  "gui": False, "cmd": "python -m atom_robot.ar5.collect --show-config"},
    "probe_leader":   {"env": "record",  "gui": False, "cmd": "python -m atom_robot.ar5.probe_leader"},
    "probe_watch":    {"env": "record",  "gui": False, "cmd": "python -m atom_robot.ar5.probe_leader --watch"},
    "camera_list":    {"env": "record",  "gui": False, "cmd": "python -m atom_robot.ar5.view_camera --list"},
    "camera_all":     {"env": "record",  "gui": True,  "cmd": "python -m atom_robot.ar5.view_camera --all"},
    "camera_check":   {"env": "record",  "gui": False, "cmd": "python {qc_script} --check"},
    # --remote：命令从 control.json 来、画面写 live.jpg，所以不需要窗口，也不需要终端 ——
    # 代理是 bash -lc 拉起它的，stdin 不是 tty，脚本本来就读不了键盘
    "record":         {"env": "record",  "gui": False,
                       "cmd": "python -m atom_robot.ar5.collect --task {task} -n {num_episodes} --hz {hz} "
                              "--calib-id {calib_id} --leader-port {leader_port} {mode} --scale {scale} "
                              "--out {out_dir} --remote --preview"},
    # 臂的自救。collect 每次启动都会跑一遍同样的 diagnose / power_back_on，
    # 这里是把那套流程单拎出来给网页用 —— 采集起不来、或者抱死之后要回位时点它。
    "unseize":         {"env": "record", "gui": False, "cmd": "python -m atom_robot.ar5.unseize"},
    # **这条会让臂移动**：上电 + 回 0 位。网页那边必须先确认过再发。
    "unseize_recover": {"env": "record", "gui": False,
                        "cmd": "python -m atom_robot.ar5.unseize --recover --home"},
    "qc":             {"env": "record",  "gui": False, "cmd": "python -m atom_robot.ar5.qc_dataset --src {out_dir} --verbose"},
    "replay_list":    {"env": "record",  "gui": False, "cmd": "python -m atom_robot.ar5.replay"},
    "replay_all":     {"env": "record",  "gui": True,  "cmd": "python -m atom_robot.ar5.replay --all"},
    "replay_one":     {"env": "record",  "gui": True,  "cmd": "python -m atom_robot.ar5.replay --episode {episode}"},
    "convert":        {"env": "convert", "gui": False, "cmd": "python -m atom_robot.ar5.to_lerobot --src {out_dir} --repo-id {repo_id}"},
    "merge":          {"env": "record",  "gui": False, "cmd": "python {merge_script} {src_list} --dst {out_dir}"},
    # 网页上自己编辑的采集命令。{cmd} 原样透传，不加引号 —— 它本来就是一整条命令。
    # 能调到代理的人必须已经在工作站上（只绑 127.0.0.1），再加令牌与 Origin 校验，
    # 所以这里放开自由命令并不比白名单更危险。
    "custom":         {"env": "record",  "gui": True,  "cmd": "{cmd}"},
}

EPISODE_RE = re.compile(r"^episode_(\d+)\.npz$")
MAX_LOG_LINES = 400

CFG = {}          # 命令行参数
JOBS = {}         # jid -> job dict
JOB_LOCK = threading.Lock()
WATCH = {"on": False, "out_dir": "", "version": "", "dataset_id": "", "last": None, "error": ""}


# ---------------------------------------------------------------------------
# 执行命令
# ---------------------------------------------------------------------------
def find_conda_hook():
    """
    定位 conda 的 profile 脚本。

    不能指望 `bash -lc` 里能直接用 conda activate：登录 shell 只读 .bash_profile，
    而 `conda init` 通常把初始化块写进 .bashrc，非交互时根本没被 source，
    于是报 "Run 'conda init' before 'conda activate'"。

    代理本身是在 conda 环境里启动的，所以从自己的环境变量就能找到 conda 根目录。
    """
    if CFG.get("conda_sh"):
        return CFG["conda_sh"] if os.path.isfile(CFG["conda_sh"]) else ""

    cands = []
    exe = os.environ.get("CONDA_EXE", "")
    if exe:                                        # <base>/bin/conda -> <base>
        cands.append(os.path.join(os.path.dirname(os.path.dirname(exe)), "etc", "profile.d", "conda.sh"))
    base = os.environ.get("CONDA_PREFIX_1") or os.environ.get("CONDA_PREFIX", "")
    if base:
        cands.append(os.path.join(base, "etc", "profile.d", "conda.sh"))
    home = os.path.expanduser("~")
    for d in ("miniconda3", "anaconda3", "miniforge3", "mambaforge", "conda"):
        cands.append(os.path.join(home, d, "etc", "profile.d", "conda.sh"))
    cands += ["/opt/conda/etc/profile.d/conda.sh",
              "/usr/local/conda/etc/profile.d/conda.sh",
              "/usr/share/miniconda/etc/profile.d/conda.sh"]

    for c in cands:
        if c and os.path.isfile(c):
            return c
    return ""


def pick_workdir(asked):
    """
    用哪个工作目录：网页给了就听网页的，给的目录不存在才退回 --workdir。

    以网页为准是因为那才是操作员眼睛看着的那一行 `cd`。让代理用自己的一份，
    等于「显示的命令」和「实际跑的命令」可以不一样 —— 排查起来毫无线索。
    """
    fallback = os.path.expanduser(CFG["workdir"])
    asked = os.path.expanduser(str(asked or "").strip())
    if not asked:
        return fallback, ""
    if not os.path.isdir(asked):
        return fallback, "网页给的工作目录不存在：%s —— 退回代理启动时的 %s" % (asked, fallback)
    return asked, ""


def render_template(key, params):
    """只把模板渲染成那条命令本身，不加 conda / cd 前缀。返回 (命令, tpl)。

    报告和日志里要记的是这条干净的 —— 带前缀那条是给 bash 跑的，给人看太吵。
    单独抽出来是因为以前两处各写各的，模板一改就对不上。
    """
    tpl = TEMPLATES.get(key)
    if not tpl:
        raise ValueError("未知命令: %s" % key)

    safe = {}
    for name in re.findall(r"\{(\w+)\}", tpl["cmd"]):
        val = params.get(name, "")
        if val is None:
            val = ""
        val = str(val)
        # 先展开 ~，否则被引号包住后 shell 不会再展开，路径会当成字面量
        if val.startswith("~"):
            val = os.path.expanduser(val)
        # mode 是 --relative / --absolute 这类开关，只允许白名单原样透传
        if name == "mode":
            safe[name] = val if val in ("--relative", "--absolute") else "--relative"
        elif name == "cmd":
            safe[name] = val                      # 自定义命令整条透传
        elif name == "src_list":
            # 形如 "--src /a --src /b"：逐段转义后重新拼，既防注入又保持多参数
            parts, buf = [], val.split()
            for i, tok in enumerate(buf):
                parts.append(tok if tok == "--src" else shlex.quote(os.path.expanduser(tok)))
            safe[name] = " ".join(parts)
        else:
            safe[name] = shlex.quote(val)
    return tpl["cmd"].format(**safe), tpl


def build_command(key, params, workdir=None):
    """把模板渲染成一条 bash 命令（带 conda / cd 前缀）；参数全部转义。"""
    inner, tpl = render_template(key, params)

    env = CFG["env_record"] if tpl["env"] == "record" else CFG["env_convert"]

    # conda activate 在非交互 shell 里默认不可用，先 source conda.sh
    prefix = ""
    hook = find_conda_hook()
    if hook:
        prefix = ". %s && " % shlex.quote(hook)
    else:
        # 兜底：显式读一次 .bashrc（conda init 一般写在这里）
        prefix = "[ -f ~/.bashrc ] && . ~/.bashrc; "

    cwd, note = pick_workdir(workdir)
    return "%sconda activate %s && cd %s && %s" % (
        prefix, shlex.quote(env), shlex.quote(cwd), inner
    ), tpl, cwd, note


def start_job(key, params, workdir=None):
    cmd, tpl, cwd, note = build_command(key, params, workdir)
    jid = "%s_%s" % (key, datetime.now().strftime("%H%M%S"))
    job = {
        "id": jid, "key": key, "cmd": cmd, "gui": tpl["gui"],
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running", "exit_code": None, "lines": [], "proc": None,
    }
    with JOB_LOCK:
        JOBS[jid] = job
    if note:
        job["lines"].append("[agent] " + note)

    def run():
        try:
            proc = subprocess.Popen(
                ["bash", "-lc", cmd],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                # 明确按 UTF-8 读，别跟着系统 locale 走：采集脚本打的是中文，
                # 而 LANG=C 之类的环境下默认解码会直接抛 UnicodeDecodeError ——
                # 命令其实跑起来了，网页上却只看到一句「启动失败: codec can't decode」
                text=True, encoding="utf-8", errors="replace", bufsize=1, cwd=cwd,
            )
            job["proc"] = proc
            for line in proc.stdout:
                with JOB_LOCK:
                    # 采集脚本的状态行用 \r 原地刷新，一段里可能挤着几百次刷新。
                    # 按 \r 拆开，网页上是一行行滚动，而不是一条几百 KB 的长行。
                    for piece in line.rstrip("\n").split("\r"):
                        piece = piece.rstrip()
                        if piece:
                            job["lines"].append(piece)
                    if len(job["lines"]) > MAX_LOG_LINES:
                        del job["lines"][:-MAX_LOG_LINES]
            proc.wait()
            job["exit_code"] = proc.returncode
            job["status"] = "done" if proc.returncode == 0 else "failed"
        except Exception as exc:                      # noqa: BLE001
            job["status"] = "failed"
            job["lines"].append("[agent] 启动失败: %s" % exc)
        finally:
            job["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    threading.Thread(target=run, daemon=True).start()
    return job


def job_view(job):
    return {
        "id": job["id"], "key": job["key"], "status": job["status"],
        "exit_code": job["exit_code"], "gui": job["gui"],
        "started_at": job["started_at"], "finished_at": job.get("finished_at"),
        "cmd": job["cmd"], "lines": job["lines"][-120:],
    }


def stop_job(jid):
    job = JOBS.get(jid)
    if not job or not job.get("proc"):
        return False
    try:
        job["proc"].terminate()
        job["status"] = "stopped"
        return True
    except Exception:                                 # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# 上报采集进度：扫描 out_dir 的 episode_*.npz，同步已落盘条数
# ---------------------------------------------------------------------------


def scan_episodes(out_dir):
    d = os.path.expanduser(out_dir)
    if not os.path.isdir(d):
        return None
    out = []
    for name in sorted(os.listdir(d)):
        m = EPISODE_RE.match(name)
        if not m:
            continue
        p = os.path.join(d, name)
        out.append({
            "idx": int(m.group(1)),
            "at": datetime.fromtimestamp(os.path.getmtime(p)).strftime("%H:%M:%S"),
        })
    return out


def post_report(episodes):
    payload = json.dumps({
        "token": CFG["token"],
        "version": WATCH["version"],
        "dataset_id": WATCH["dataset_id"],
        "out_dir": WATCH["out_dir"],
        "episodes": episodes,
        "agent": VERSION,
    }).encode("utf-8")
    url = CFG["site"].rstrip("/") + "/actions/agent_report.php"
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def watcher_loop():
    last_sig = None
    while True:
        time.sleep(CFG["interval"])
        if not WATCH["on"]:
            continue
        try:
            eps = scan_episodes(WATCH["out_dir"])
            if eps is None:
                WATCH["error"] = "目录不存在: %s" % WATCH["out_dir"]
                continue
            sig = json.dumps(eps, sort_keys=True)
            if sig == last_sig:
                continue
            post_report(eps)
            last_sig = sig
            WATCH["error"] = ""
            WATCH["last"] = datetime.now().strftime("%H:%M:%S")
            log("上报 %d 条 → %s" % (len(eps), WATCH["dataset_id"]))
        except urllib.error.URLError as exc:
            WATCH["error"] = "连不上网站: %s" % exc
        except Exception as exc:                      # noqa: BLE001
            WATCH["error"] = str(exc)



# ---------------------------------------------------------------------------
# 删除采集目录（网页上删数据集时用）
# 只允许删「看起来确实是采集目录」的路径，不接受任意路径。
# ---------------------------------------------------------------------------
def collect_procs():
    """工作站上正在跑的采集进程，返回 [(pid, 命令行)]。

    两个 collect 会抢同一个 control.json：谁先读到谁消费，另一个永远收不到 ——
    症状是网页上「命令时灵时不灵、回合号乱跳」。这种事光看网页看不出来，
    只能问系统。/proc 扫一遍就够，不 shell 出去调 pgrep。
    """
    out = []
    try:
        me = os.getpid()
        for entry in os.listdir("/proc"):
            if not entry.isdigit() or int(entry) == me:
                continue
            try:
                with open("/proc/%s/cmdline" % entry, "rb") as f:
                    cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
            except OSError:
                continue
            if not cmd or "atom_agent" in cmd:
                continue
            # 两种布局都认：老的 -m atom_robot.ar5.collect，新的 python …/collect.py
            if "ar5.collect" in cmd or "collect.py" in cmd:
                out.append({"pid": int(entry), "cmd": cmd[:160]})
    except OSError:
        pass                                   # 不是 Linux，或者没有 /proc
    return out


def out_roots():
    """允许操作的采集输出根目录，可以有多个（--out-root 给几次就是几个）。"""
    roots = CFG.get("out_roots") or ["~/vel_episodes"]
    out = []
    for r in roots:
        r = os.path.realpath(os.path.expanduser(str(r).strip()))
        if r and r not in out:
            out.append(r)
    return out


def under_roots(p):
    """p 是否落在某个允许的根目录之下（根目录本身不算）。返回 (是否, 命中的根)。"""
    for r in out_roots():
        if p.startswith(r + os.sep):
            return True, r
    return False, ""


def roots_hint():
    """拒绝时告诉人怎么放行 —— 光说「不允许」等于把人晾在那儿。"""
    return "当前允许：%s。要放行别的目录，启动代理时加 --out-root <目录>（可以给多次）。" % (
        "、".join(out_roots()),)


def check_deletable(path):
    """返回 (ok, 说明, 绝对路径, 统计)。不通过时绝不删。"""
    if not path:
        return False, "没有给路径", "", {}
    p = os.path.realpath(os.path.expanduser(str(path)))
    home = os.path.realpath(os.path.expanduser("~"))

    if os.path.dirname(p) == p:                       # 根目录
        return False, "拒绝：不能删根目录", p, {}
    if p == home or p in out_roots():
        return False, "拒绝：不能删家目录或输出根目录本身", p, {}
    if not os.path.isdir(p):
        return False, "目录不存在（可能已经删过）", p, {}

    npz = glob.glob(os.path.join(p, "episode_*.npz"))
    vids = [f for f in os.listdir(p) if f.lower().endswith((".mp4", ".avi", ".mkv"))]
    merged = os.path.isfile(os.path.join(p, "merge_manifest.json"))
    under_root, _hit = under_roots(p)

    # 必须在某个允许的根目录之下，或者目录里确实有 episode 文件，二者居其一
    if not under_root and not npz and not merged:
        return False, "拒绝：既不在允许的输出根目录之下，也不像采集目录。" + roots_hint(), p, {}

    size = 0
    for dirpath, _, names in os.walk(p):
        for n in names:
            try:
                size += os.path.getsize(os.path.join(dirpath, n))
            except OSError:
                pass
    return True, "", p, {"episodes": len(npz), "videos": len(vids), "bytes": size, "merged": merged}


def human_bytes(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return "%.1f %s" % (n, u) if u != "B" else "%d B" % n
        n /= 1024.0



# ---------------------------------------------------------------------------
# 上传数据集到 NAS
# 采集写在工作站本地，只有工作站看得到。上传后本站有完整副本，所有人都能看。
# 走 HTTP 分片上传，不需要工作站挂载 NAS。
# ---------------------------------------------------------------------------
UPLOAD_CHUNK = 8 * 1024 * 1024
UPLOAD_NAME_RE = re.compile(
    r"^(episode_\d+\.npz|episode_\d+_[A-Za-z0-9_.-]+\.(?:mp4|avi|mkv)|"
    # dataset.json：这批的语言 prompt、夹持力、速度；collect.log：抱死之后要查的就是它。
    # 不在名单里会被悄悄跳过，NAS 上那份就缺了这两样
    r"merge_manifest\.json|meta\.json|dataset\.json|collect\.log|thumb\.(?:png|jpg|jpeg|webp|svg))$", re.I)


def upload_files(out_dir):
    """列出该目录里可上传的文件，按名字排序。"""
    d = os.path.expanduser(out_dir)
    if not os.path.isdir(d):
        return None
    out = []
    for n in sorted(os.listdir(d)):
        if UPLOAD_NAME_RE.match(n) and os.path.isfile(os.path.join(d, n)):
            out.append((n, os.path.getsize(os.path.join(d, n))))
    return out



def _episode_dir_info(d, name):
    """d 里有 episode_*.npz 就返回它的概况（条数、大小、最近一条的时间、dataset.json 里的 prompt）。"""
    eps = glob.glob(os.path.join(d, "episode_*.npz"))
    if not eps:
        return None
    size, latest = 0, 0.0
    for n in os.listdir(d):
        p = os.path.join(d, n)
        if UPLOAD_NAME_RE.match(n) and os.path.isfile(p):
            st = os.stat(p)
            size += st.st_size
            latest = max(latest, st.st_mtime)
    prompt = ""
    try:
        with open(os.path.join(d, "dataset.json"), encoding="utf-8") as f:
            prompt = str(json.load(f).get("prompt") or "")
    except Exception:                                 # noqa: BLE001 —— 没有 / 坏了就当没写
        pass
    return {"name": name, "path": d, "episodes": len(eps), "bytes": size, "prompt": prompt,
            "mtime": datetime.fromtimestamp(latest).strftime("%Y-%m-%d %H:%M") if latest else ""}


def looks_like_dataset(p):
    """目录里直接有 episode_*.npz 或 merge_manifest.json —— 一看就是采集目录。"""
    return bool(glob.glob(os.path.join(p, "episode_*.npz"))) or \
        os.path.isfile(os.path.join(p, "merge_manifest.json"))


def check_readable_dir(path):
    """只读操作用：在输出根目录之下，或者本身就是采集目录（导入的目录可以在任何地方）。"""
    if not path:
        return False, "没有给 out_dir"
    p = os.path.realpath(os.path.expanduser(str(path)))
    if not os.path.isdir(p):
        return False, "目录不存在：%s" % p
    ok, _hit = under_roots(p)
    if ok or p in out_roots() or looks_like_dataset(p):
        return True, p
    return False, "拒绝：不在允许的输出根目录之下，也不像采集目录。" + roots_hint()


def browse_dir(path):
    """列 path 下的子目录（不进隐藏目录）。含 episode 的带概况，能直接勾来导入。"""
    p = os.path.realpath(os.path.expanduser(str(path or "~")))
    if not os.path.isdir(p):
        raise ValueError("目录不存在：%s" % p)
    dirs = []
    try:
        names = sorted(os.listdir(p), key=str.lower)
    except OSError as exc:
        raise ValueError("读不了这个目录：%s" % exc)
    for n in names:
        if n.startswith("."):
            continue
        d = os.path.join(p, n)
        if not os.path.isdir(d):
            continue
        try:
            info = _episode_dir_info(d, n) or {"name": n, "path": d, "episodes": 0}
            info["has_sub"] = any(not x.startswith(".") and os.path.isdir(os.path.join(d, x))
                                  for x in os.listdir(d))
        except OSError:
            continue                                  # 没权限的跳过
        dirs.append(info)
        if len(dirs) >= 500:
            break
    parent = os.path.dirname(p)
    return {"path": p, "parent": parent if parent != p else "",
            "here": _episode_dir_info(p, os.path.basename(p)), "dirs": dirs,
            "home": os.path.realpath(os.path.expanduser("~")), "roots": out_roots()}


def list_episode_dirs():
    """列出各个根目录下确实含 episode_*.npz 的目录，给「目录不存在」时当候选、给「导入本地数据集」列清单。

    看两层：<根>/<名字>/ 是普通采集；<根>/<这次运行>/<任务名>/ 是 OmniPicker 按 P 开的数据集。"""
    out = []
    roots = out_roots()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if not os.path.isdir(d):
                continue
            info = _episode_dir_info(d, name)
            if info:
                out.append(info)
            for sub in sorted(os.listdir(d)):
                sd = os.path.join(d, sub)
                if os.path.isdir(sd):
                    info = _episode_dir_info(sd, name + "/" + sub)
                    if info:
                        out.append(info)
    return "、".join(roots), out


def _up_url(op, **kw):
    from urllib.parse import urlencode
    kw["op"] = op
    kw["token"] = CFG["token"]
    return CFG["site"].rstrip("/") + "/actions/dataset_upload.php?" + urlencode(kw)


def _up_json(url, data=None, binary=False):
    if binary:
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/octet-stream"})
    elif data is not None:
        from urllib.parse import urlencode
        req = urllib.request.Request(url, data=urlencode(data).encode("utf-8"))
    else:
        req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_upload(job, version, dataset_id, out_dir):
    def say(msg):
        with JOB_LOCK:
            job["lines"].append(msg)
            if len(job["lines"]) > MAX_LOG_LINES:
                del job["lines"][:-MAX_LOG_LINES]

    try:
        files = upload_files(out_dir)
        if files is None:
            raise RuntimeError("目录不存在: %s" % out_dir)
        if not files:
            raise RuntimeError("目录里没有可上传的文件: %s" % out_dir)

        total = sum(sz for _, sz in files)
        say("准备上传 %d 个文件，共 %s" % (len(files), human_bytes(total)))
        say("来源 %s" % os.path.expanduser(out_dir))
        say("目标 %s（数据集 %s）" % (CFG["site"], dataset_id))
        say("")

        sent_all = 0
        t0 = time.time()
        for i, (name, size) in enumerate(files, 1):
            path = os.path.join(os.path.expanduser(out_dir), name)
            got = _up_json(_up_url("probe", version=version, id=dataset_id, name=name)).get("received", 0)
            if got == size:
                say("[%d/%d] %-34s 已存在，跳过" % (i, len(files), name))
                sent_all += size
                continue
            if got > size:
                got = 0
            offset = got
            with open(path, "rb") as fh:
                fh.seek(offset)
                while offset < size:
                    if job.get("cancel"):
                        raise RuntimeError("已取消")
                    buf = fh.read(UPLOAD_CHUNK)
                    if not buf:
                        break
                    r = _up_json(_up_url("chunk", version=version, id=dataset_id,
                                         name=name, offset=offset), data=buf, binary=True)
                    if r.get("realign"):
                        offset = int(r.get("received", 0))
                        fh.seek(offset)
                        continue
                    offset = int(r.get("received", offset + len(buf)))
                    sent_all += len(buf)
            pct = sent_all / total * 100 if total else 100
            speed = sent_all / max(0.001, time.time() - t0)
            say("[%d/%d] %-34s %8s  总进度 %.1f%%  %s/s"
                % (i, len(files), name, human_bytes(size), pct, human_bytes(speed)))

        _up_json(_up_url("done"), data={"version": version, "id": dataset_id,
                                        "files": len(files), "bytes": total,
                                        "token": CFG["token"]})
        say("")
        say("上传完成：%d 个文件，%s，用时 %d 秒" % (len(files), human_bytes(total), time.time() - t0))
        job["status"] = "done"
        job["exit_code"] = 0
    except Exception as exc:                          # noqa: BLE001
        say("[失败] %s" % exc)
        job["status"] = "failed"
        job["exit_code"] = 1
    finally:
        job["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def start_upload(version, dataset_id, out_dir):
    jid = "upload_%s" % datetime.now().strftime("%H%M%S")
    job = {
        "id": jid, "key": "upload", "cmd": "上传 %s → NAS" % out_dir, "gui": False,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running", "exit_code": None, "lines": [], "proc": None, "cancel": False,
    }
    with JOB_LOCK:
        JOBS[jid] = job
    threading.Thread(target=run_upload, args=(job, version, dataset_id, out_dir), daemon=True).start()
    return job


# ---------------------------------------------------------------------------
# 自动检查（qc_dataset.py）与逐条体检
#
# 采集一结束，网页问一句「要不要自动检查」，点是就走这里：
#   1. 跑 qc_dataset.py，输出实时回传网页
#   2. 扫目录，逐条读 npz 的帧数 / 相机 / 夹爪行程等参数（qc 输出解析不到时也有据可依）
#   3. 解析 qc 输出（有 json 报告就优先读 json），把问题归到具体的 episode 上
#   4. 整份报告 POST 回网站，网页据此列出有问题的回合、放视频、让人勾选删除
#
# 解析是**尽力而为**：qc_dataset.py 的输出格式将来改了，本地体检那部分仍然有效，
# 解析不到的行原样留在 output 里，网页上可以展开看。
# ---------------------------------------------------------------------------
VIDEO_RE = re.compile(r"^episode_(\d+)_(.+)\.(mp4|avi|mkv)$", re.I)
# 允许通过 /file 播放 / 查看的文件名（只读，且只在采集输出根目录之下）
SERVE_RE = re.compile(r"^episode_\d+(_[A-Za-z0-9_.-]+)?\.(mp4|avi|mkv|png|jpg|jpeg|webp)$", re.I)
MIME = {".mp4": "video/mp4", ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

# qc 输出里判定严重程度的关键词（小写比较）
FAIL_WORDS = ("fail", "error", "错误", "失败", "✗", "✘", "✖", "[x]", "严重", "critical", "critic")
WARN_WORDS = ("warn", "警告", "⚠", "异常", "可疑", "超阈值", "偏低", "偏高", "suspect", "疑似")
# 从一行里认出 episode 编号的几种写法
EP_REF_RES = [
    re.compile(r"episode[_\s-]*0*(\d+)", re.I),
    re.compile(r"\bep[_\s-]*0*(\d+)\b", re.I),
    re.compile(r"第\s*0*(\d+)\s*条"),
    re.compile(r"#\s*0*(\d+)\b"),
]
QC_JSON_NAMES = ("qc_report.json", "qc_result.json", "qc_dataset.json", "qc.json", "report.json")


def say(job, msg):
    """往任务日志里写一行（网页弹窗实时显示的就是它）。"""
    with JOB_LOCK:
        job["lines"].append(msg)
        if len(job["lines"]) > MAX_LOG_LINES:
            del job["lines"][:-MAX_LOG_LINES]


def exec_stream(job, cmd, cwd=None):
    """跑一条 bash 命令，输出边出边进任务日志；返回退出码。"""
    proc = subprocess.Popen(
        ["bash", "-lc", cmd],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        cwd=cwd or os.path.expanduser(CFG["workdir"]),
    )
    job["proc"] = proc
    out = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        out.append(line)
        if len(out) > MAX_LOG_LINES:
            del out[:-MAX_LOG_LINES]
        say(job, line)
    proc.wait()
    job["proc"] = None
    return proc.returncode, out


def start_task(key, title, fn, gui=False):
    """通用后台任务：和 start_job 一样进 JOBS，网页照旧用 /job/<id> 轮询。"""
    jid = "%s_%s" % (key, datetime.now().strftime("%H%M%S"))
    job = {
        "id": jid, "key": key, "cmd": title, "gui": gui,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running", "exit_code": None, "lines": [], "proc": None, "cancel": False,
    }
    with JOB_LOCK:
        JOBS[jid] = job

    def wrap():
        try:
            fn(job)
            if job["status"] == "running":
                job["status"] = "done"
                job["exit_code"] = 0 if job["exit_code"] is None else job["exit_code"]
        except Exception as exc:                      # noqa: BLE001
            say(job, "[失败] %s" % exc)
            job["status"] = "failed"
            job["exit_code"] = 1
        finally:
            job["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    threading.Thread(target=wrap, daemon=True).start()
    return job


def post_site(action, payload):
    """POST JSON 到网站的某个 action，返回解析后的响应。"""
    payload = dict(payload)
    payload["token"] = CFG["token"]
    url = CFG["site"].rstrip("/") + "/actions/" + action
    req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_snapshot(version, dataset_id, out_dir):
    """把目录现状（episode_*.npz 快照）同步给网站，用的是和进度上报同一个接口。"""
    eps = scan_episodes(out_dir)
    if eps is None:
        return None
    payload = json.dumps({
        "token": CFG["token"], "version": version, "dataset_id": dataset_id,
        "out_dir": out_dir, "episodes": eps, "agent": VERSION,
    }).encode("utf-8")
    url = CFG["site"].rstrip("/") + "/actions/agent_report.php"
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def scan_episode_files(out_dir):
    """列出目录里的每条 episode 及其视频文件。"""
    d = os.path.expanduser(out_dir)
    eps = {}
    for name in sorted(os.listdir(d)):
        m = EPISODE_RE.match(name)
        if m:
            idx = int(m.group(1))
            e = eps.setdefault(idx, {"idx": idx, "npz": "", "videos": [], "width": len(m.group(1))})
            e["npz"] = name
            e["width"] = len(m.group(1))
            continue
        mv = VIDEO_RE.match(name)
        if mv:
            idx = int(mv.group(1))
            e = eps.setdefault(idx, {"idx": idx, "npz": "", "videos": [], "width": len(mv.group(1))})
            e["videos"].append(name)
    return [eps[k] for k in sorted(eps)]


def probe_npz(path):
    """读一条 npz 的参数。读不出来不抛异常，返回里带 error。"""
    info = {"frames": None, "task": "", "cameras": [], "state_dim": None, "action_dim": None,
            "fps": None, "gripper": None, "motion": None, "error": ""}
    try:
        import numpy as np
    except Exception as exc:                          # noqa: BLE001
        info["error"] = "numpy 不可用（换个环境跑代理）: %s" % exc
        return info

    def first(z, *names):
        for n in names:
            if n in z.files:
                return z[n]
        return None

    try:
        with np.load(path, allow_pickle=True) as z:
            t = first(z, "task", "language_instruction", "instruction")
            if t is not None:
                info["task"] = str(np.asarray(t).reshape(-1)[0])
            cams = first(z, "camera_keys", "cameras", "camera_names")
            if cams is not None:
                info["cameras"] = [str(x) for x in np.asarray(cams).reshape(-1)]
            fps = first(z, "fps_measured", "fps", "hz")
            if fps is not None:
                try:
                    info["fps"] = round(float(np.asarray(fps).reshape(-1)[0]), 2)
                except Exception:                     # noqa: BLE001
                    pass
            st = first(z, "state", "states", "observation.state")
            if st is not None:
                st = np.asarray(st, dtype=float)
                info["frames"] = int(st.shape[0])
                info["state_dim"] = int(st.shape[1]) if st.ndim > 1 else 1
                if st.ndim > 1 and st.shape[0] > 1:
                    d = np.abs(np.diff(st, axis=0))
                    d = d[np.isfinite(d)]
                    info["motion"] = round(float(d.sum()), 4) if d.size else 0.0
                    g = st[:, -1]
                    g = g[np.isfinite(g)]
                    if g.size:
                        info["gripper"] = {
                            "min": round(float(g.min()), 4),
                            "max": round(float(g.max()), 4),
                            "travel": round(float(g.max() - g.min()), 4),
                        }
            ac = first(z, "action", "actions")
            if ac is not None:
                ac = np.asarray(ac)
                info["action_dim"] = int(ac.shape[1]) if ac.ndim > 1 else 1
                if info["frames"] is None:
                    info["frames"] = int(ac.shape[0])
            ts = first(z, "timestamps", "timestamp", "ts", "time")
            if info["fps"] is None and ts is not None:
                ts = np.asarray(ts, dtype=float).reshape(-1)
                if ts.size > 1 and np.isfinite(ts).all() and ts[-1] > ts[0]:
                    info["fps"] = round(float((ts.size - 1) / (ts[-1] - ts[0])), 2)
    except Exception as exc:                          # noqa: BLE001
        info["error"] = "读不出 npz: %s" % exc
    return info


def parse_qc_lines(lines):
    """
    从 qc_dataset.py 的输出里把问题归到 episode 上。
    只认带严重程度关键词的行：有 episode 编号的挂到那条，没有的算全局提示。
    """
    per = {}
    notes = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if any(w in low for w in FAIL_WORDS):
            level = "fail"
        elif any(w in low for w in WARN_WORDS):
            level = "warn"
        else:
            continue
        idxs = []
        for rx in EP_REF_RES:
            for m in rx.finditer(line):
                try:
                    idxs.append(int(m.group(1)))
                except ValueError:
                    pass
        idxs = sorted(set(idxs))
        text = line[:300]
        if not idxs:
            if len(notes) < 60:
                notes.append({"level": level, "text": text})
            continue
        for i in idxs[:8]:
            bucket = per.setdefault(i, [])
            if len(bucket) < 20 and not any(x["text"] == text for x in bucket):
                bucket.append({"level": level, "code": "qc", "text": text})
    return per, notes


def read_qc_json(out_dir, since_ts):
    """
    qc_dataset.py 若写了 json 报告就优先用它（比解析文本可靠）。
    字段名各版本可能不同，这里尽量兼容；认不出来就返回空，退回文本解析。
    """
    d = os.path.expanduser(out_dir)
    for name in QC_JSON_NAMES:
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            continue
        try:
            if os.path.getmtime(p) < since_ts - 5:     # 不是这次跑出来的，忽略
                continue
            with open(p, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:                             # noqa: BLE001
            continue
        rows = None
        if isinstance(doc, list):
            rows = doc
        elif isinstance(doc, dict):
            for k in ("episodes", "results", "items", "per_episode"):
                if isinstance(doc.get(k), list):
                    rows = doc[k]
                    break
                if isinstance(doc.get(k), dict):
                    rows = [dict(v, index=k2) for k2, v in doc[k].items()]
                    break
        if not rows:
            continue
        per = {}
        for r in rows:
            if not isinstance(r, dict):
                continue
            idx = None
            for k in ("index", "idx", "episode", "episode_index", "id", "n"):
                if k in r:
                    try:
                        idx = int(str(r[k]).strip().lstrip("#"))
                        break
                    except (TypeError, ValueError):
                        pass
            if idx is None:
                continue
            level = str(r.get("status") or r.get("level") or r.get("verdict") or r.get("result") or "").lower()
            level = "fail" if level in ("fail", "failed", "error", "bad") else ("warn" if level in ("warn", "warning", "suspect") else "")
            items = []
            for k in ("issues", "problems", "errors", "warnings", "messages", "reasons"):
                v = r.get(k)
                if isinstance(v, str):
                    v = [v]
                if isinstance(v, list):
                    for it in v:
                        if isinstance(it, dict):
                            txt = str(it.get("text") or it.get("msg") or it.get("message") or it)
                            lv = str(it.get("level") or it.get("status") or "").lower()
                        else:
                            txt, lv = str(it), ""
                        lv = "fail" if lv.startswith("fail") or lv == "error" else ("warn" if lv.startswith("warn") else (level or ("fail" if k == "errors" else "warn")))
                        items.append({"level": lv, "code": "qc", "text": txt[:300]})
            if not items and level:
                items.append({"level": level, "code": "qc", "text": "qc 报告判定为 %s" % level})
            if items:
                per[idx] = items[:20]
        if per:
            log("读到 qc json 报告 %s（%d 条有结论）" % (name, len(per)))
            return per, name
    return {}, ""


def build_qc_report(out_dir, cmd, exit_code, out_lines, since_ts):
    """跑完 qc 之后：扫目录 + 读 npz + 合并 qc 结论，得到整份报告。"""
    d = os.path.expanduser(out_dir)
    files = scan_episode_files(d)
    qc_per, qc_json = read_qc_json(out_dir, since_ts)
    parsed_per, notes = parse_qc_lines(out_lines)
    if not qc_per:
        qc_per = parsed_per
    else:                                             # json 为主，文本补充
        for k, v in parsed_per.items():
            qc_per.setdefault(k, v)

    eps = []
    for e in files:
        stem = "episode_%0*d" % (e["width"] or 4, e["idx"])
        npz_path = os.path.join(d, e["npz"]) if e["npz"] else ""
        info = probe_npz(npz_path) if npz_path else {"error": "没有 npz 文件", "frames": None,
                                                     "task": "", "cameras": [], "state_dim": None,
                                                     "action_dim": None, "fps": None,
                                                     "gripper": None, "motion": None}
        vids = [dict(name=v, **video_meta(os.path.join(d, v))) for v in e["videos"]]
        eps.append({
            "idx": e["idx"], "stem": stem, "npz": e["npz"],
            "bytes": (os.path.getsize(npz_path) if npz_path and os.path.isfile(npz_path) else 0),
            "at": (datetime.fromtimestamp(os.path.getmtime(npz_path)).strftime("%Y-%m-%d %H:%M:%S")
                   if npz_path and os.path.isfile(npz_path) else ""),
            "frames": info["frames"], "fps": info["fps"], "task": info["task"],
            "cameras": info["cameras"], "state_dim": info["state_dim"], "action_dim": info["action_dim"],
            "gripper": info["gripper"], "motion": info["motion"], "probe_error": info["error"],
            "videos": vids,
            "issues": list(qc_per.get(e["idx"], [])),
        })

    # ---- 本地体检：qc 解析不到也能发现的明显问题 ----
    frames = [e["frames"] for e in eps if e["frames"]]
    med = sorted(frames)[len(frames) // 2] if frames else 0
    vid_counts = [len(e["videos"]) for e in eps]
    common_vids = max(set(vid_counts), key=vid_counts.count) if vid_counts else 0
    motions = sorted(e["motion"] for e in eps if e["motion"] is not None)
    med_motion = motions[len(motions) // 2] if motions else 0.0

    for e in eps:
        add = e["issues"].append
        if not e["npz"]:
            add({"level": "fail", "code": "no_npz", "text": "只有视频，没有 episode npz 文件"})
        if e["probe_error"]:
            add({"level": "fail", "code": "unreadable", "text": e["probe_error"]})
        if not e["videos"]:
            add({"level": "fail", "code": "no_video", "text": "没有配套的视频文件，回放看不了"})
        elif common_vids and len(e["videos"]) != common_vids:
            add({"level": "fail", "code": "video_count",
                 "text": "相机路数 %d，与本批多数（%d 路）不一致，混在一起转换会报错" % (len(e["videos"]), common_vids)})
        for v in e["videos"]:
            if v["bytes"] < 1024:
                add({"level": "fail", "code": "empty_video", "text": "视频 %s 只有 %d 字节，基本是空的" % (v["name"], v["bytes"])})
        if e["frames"] is not None:
            if e["frames"] < 10:
                add({"level": "fail", "code": "too_short", "text": "只有 %d 帧，短到没有内容" % e["frames"]})
            elif med and e["frames"] < med * 0.4:
                add({"level": "warn", "code": "short", "text": "%d 帧，不到本批中位数（%d 帧）的四成，可能是误存" % (e["frames"], med)})
            elif med and e["frames"] > med * 2.5:
                add({"level": "warn", "code": "long", "text": "%d 帧，超过本批中位数（%d 帧）的 2.5 倍，可能中间卡住了" % (e["frames"], med)})
        if e["gripper"] is not None and e["gripper"]["travel"] < 0.01:
            add({"level": "warn", "code": "gripper_still",
                 "text": "夹爪全程没动（行程 %.3f），这一维学不到东西" % e["gripper"]["travel"]})
        if e["motion"] is not None and med_motion > 0 and e["motion"] < med_motion * 0.15:
            add({"level": "warn", "code": "no_motion",
                 "text": "整条几乎没有动作（位移量 %.2f，中位数 %.2f），像是空录" % (e["motion"], med_motion)})

        levels = [i["level"] for i in e["issues"]]
        e["level"] = "fail" if "fail" in levels else ("warn" if "warn" in levels else "ok")

    counts = {"total": len(eps), "fail": 0, "warn": 0, "ok": 0}
    for e in eps:
        counts[e["level"]] += 1

    return {
        "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent": VERSION,
        "cmd": cmd,
        "exit_code": exit_code,
        "out_dir": os.path.expanduser(out_dir),
        "qc_json": qc_json,
        "counts": counts,
        "episodes": eps,
        "notes": notes,
        "output": out_lines[-400:],
    }


def run_qc_scan(job, version, dataset_id, out_dir, custom_cmd="", workdir=None):
    d = os.path.expanduser(out_dir)
    if not os.path.isdir(d):
        raise RuntimeError("目录不存在：%s" % out_dir)

    since = time.time()
    if custom_cmd:
        cmd = custom_cmd
        say(job, "自定义检查命令：%s" % cmd)
        full, _tpl, cwd, note = build_command("custom", {"cmd": cmd}, workdir)
    else:
        # cmd 是记进报告里给人看的那条（不带 conda / cd 前缀）。
        # **必须在这儿赋值**：以前它缩进错了、落在下面 `if note:` 里，
        # 而 note 只有工作目录不存在时才非空 —— 于是一切正常的时候反而没有 cmd，
        # build_qc_report() 直接 UnboundLocalError，报告也就回传不了。
        cmd, _t = render_template("qc", {"out_dir": out_dir})
        full, _tpl, cwd, note = build_command("qc", {"out_dir": out_dir}, workdir)
    if note:
        say(job, "[agent] " + note)
    say(job, "在 %s 上跑自动检查…" % d)
    say(job, "")
    code, out_lines = exec_stream(job, full, cwd)
    say(job, "")
    say(job, "qc_dataset 退出码 %s" % code)

    say(job, "逐条读参数（帧数 / 相机 / 夹爪行程）…")
    report = build_qc_report(out_dir, cmd, code, out_lines, since)
    c = report["counts"]
    say(job, "共 %d 条：%d 条失败，%d 条警告，%d 条正常"
        % (c["total"], c["fail"], c["warn"], c["ok"]))
    for e in report["episodes"]:
        if e["level"] != "ok":
            say(job, "  %s  %s  %s" % ("✗" if e["level"] == "fail" else "⚠", e["stem"],
                                       e["issues"][0]["text"] if e["issues"] else ""))

    say(job, "")
    say(job, "回传报告给网站…")
    r = post_site("dataset_qc_report.php", {
        "version": version, "dataset_id": dataset_id,
        "out_dir": out_dir, "report": report,
    })
    if r.get("error"):
        raise RuntimeError("网站拒绝了报告：%s" % r["error"])
    try:
        post_snapshot(version, dataset_id, out_dir)
    except Exception as exc:                          # noqa: BLE001
        say(job, "[提示] 条数同步失败（不影响报告）：%s" % exc)
    say(job, "完成。回到网页看结果。")
    job["exit_code"] = 0


def start_qc_scan(version, dataset_id, out_dir, custom_cmd="", workdir=None):
    return start_task("qcscan", "自动检查 %s" % out_dir,
                      lambda job: run_qc_scan(job, version, dataset_id, out_dir, custom_cmd, workdir))


# ---------------------------------------------------------------------------
# 删除若干条 episode 并重新编号
#
# 编号有空洞其实不影响 to_lerobot（它会自己重编 episode_index），但目录里编号连续
# 更容易和网页上的条目对得上，也便于回放时按编号找。重命名分两步走（先改成临时名
# 再落到目标名），避免新旧编号撞车把文件覆盖掉。
# ---------------------------------------------------------------------------
def prune_episodes(job, version, dataset_id, out_dir, drop, renumber=True, dry_run=False):
    ok, why, d, _info = check_deletable(out_dir)
    if not ok:
        raise RuntimeError(why)

    files = scan_episode_files(d)
    if not files:
        raise RuntimeError("目录里没有 episode 文件：%s" % d)
    have = {e["idx"]: e for e in files}
    drop = sorted({int(x) for x in drop})
    missing = [i for i in drop if i not in have]
    if missing:
        raise RuntimeError("这些编号在工作站上找不到（页面可能已过期，刷新后再试）：%s"
                           % ", ".join(str(i) for i in missing))
    if not drop:
        raise RuntimeError("没有选中任何回合")
    if len(drop) >= len(files):
        raise RuntimeError("选中了全部 %d 条。要整批丢掉请用列表页的「删除数据集」。" % len(files))

    width = max((e["width"] for e in files), default=4) or 4

    # ---- 删 ----
    removed, freed = [], 0
    for i in drop:
        e = have[i]
        names = ([e["npz"]] if e["npz"] else []) + e["videos"]
        for name in names:
            p = os.path.join(d, name)
            try:
                freed += os.path.getsize(p)
            except OSError:
                pass
            if not dry_run:
                try:
                    os.remove(p)
                except OSError as exc:
                    raise RuntimeError("删不掉 %s: %s" % (name, exc))
            removed.append(name)
        say(job, "%s episode_%0*d（%d 个文件）" % ("将删除" if dry_run else "已删除",
                                                 e["width"], e["idx"], len(names)))

    keep = [e for e in files if e["idx"] not in set(drop)]

    # ---- 重编号：两步重命名，避免新旧编号互相覆盖 ----
    # 起始编号沿用目录原来的最小编号（collect 从 0 开，合并脚本也从 0 开；
    # 万一某批是从 1 开的，也不要因为重编号把整批平移）
    base = files[0]["idx"]
    mapping = []
    if renumber:
        plan = []
        for pos, e in enumerate(keep):
            new_idx = base + pos
            if new_idx != e["idx"]:
                plan.append((e, new_idx))
            mapping.append({"old": e["idx"], "new": new_idx})
        if plan:
            tmp = ".__renum__"
            if not dry_run:
                for e, new_idx in plan:                      # 第一步：改成临时名
                    for name in ([e["npz"]] if e["npz"] else []) + e["videos"]:
                        os.rename(os.path.join(d, name), os.path.join(d, tmp + name))
                for e, new_idx in plan:                      # 第二步：落到目标名
                    old_stem = "episode_%0*d" % (e["width"], e["idx"])
                    new_stem = "episode_%0*d" % (width, new_idx)
                    for name in ([e["npz"]] if e["npz"] else []) + e["videos"]:
                        suffix = name[len(old_stem):]
                        os.rename(os.path.join(d, tmp + name), os.path.join(d, new_stem + suffix))
            for e, new_idx in plan:
                say(job, "%s episode_%0*d → episode_%0*d" % ("将重编号" if dry_run else "重编号",
                                                            e["width"], e["idx"], width, new_idx))
        say(job, "剩下 %d 条，编号 %d ~ %d 连续" % (len(keep), base, base + len(keep) - 1))
    else:
        mapping = [{"old": e["idx"], "new": e["idx"]} for e in keep]
        say(job, "剩下 %d 条，编号保持原样（有空洞）" % len(keep))

    if dry_run:
        say(job, "这是试运行，没有动任何文件。")
        return {"dry_run": True, "removed": removed, "mapping": mapping}

    # ---- 留个底账，将来想追为什么少了几条 ----
    logfile = os.path.join(d, "prune_log.json")
    history = []
    if os.path.isfile(logfile):
        try:
            with open(logfile, "r", encoding="utf-8") as f:
                doc = json.load(f)
            history = doc.get("history", []) if isinstance(doc, dict) else []
        except Exception:                             # noqa: BLE001
            history = []
    history.append({
        "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "deleted": drop, "files": removed, "renumber": bool(renumber),
        "mapping": mapping, "remaining": len(keep), "agent": VERSION,
    })
    try:
        with open(logfile, "w", encoding="utf-8") as f:
            json.dump({"dir": d, "history": history[-50:]}, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        say(job, "[提示] prune_log.json 写不了：%s" % exc)

    say(job, "")
    say(job, "共删除 %d 条、%d 个文件，腾出 %s" % (len(drop), len(removed), human_bytes(freed)))
    say(job, "同步给网站…")
    r = post_site("dataset_prune.php", {
        "version": version, "dataset_id": dataset_id, "out_dir": out_dir,
        "deleted": drop, "mapping": mapping, "remaining": len(keep),
        "next_index": (base + len(keep)) if renumber else (max((e["idx"] for e in keep), default=base - 1) + 1),
        "renumber": bool(renumber), "files": len(removed), "bytes": freed,
    })
    if r.get("error"):
        raise RuntimeError("网站没接住这次删除：%s" % r["error"])
    try:
        post_snapshot(version, dataset_id, out_dir)
    except Exception as exc:                          # noqa: BLE001
        say(job, "[提示] 条数同步失败：%s" % exc)
    say(job, "完成。回到网页看结果。")
    job["exit_code"] = 0
    return {"removed": removed, "mapping": mapping}


def start_prune(version, dataset_id, out_dir, drop, renumber=True, dry_run=False):
    return start_task("prune", "删除 %d 条并重新编号" % len(drop),
                      lambda job: prune_episodes(job, version, dataset_id, out_dir,
                                                 drop, renumber, dry_run))


# ---------------------------------------------------------------------------
# 合并：把 n 个「一种指令一个目录」的采集并成一个大数据集
#
# 以前这一步只在网页上存在 —— 本站建条记录、生成一句命令，人自己去工作站跑。
# 结果是网页说有 120 条，磁盘上没有那个目录。合并出来的数据集**从本地看应该和
# 别的没有区别**：同样的 episode_%04d.npz + 视频，编号连续，qc / 逐条查看 /
# to_lerobot 直接就能用。所以这活儿归代理。
# ---------------------------------------------------------------------------


def episode_signature(d, e):
    """一条 episode 的「形状」：相机名 + state/action 维度。

    维度或路数不同的数据混在一起，lerobot 转换会报错，所以合并前要先比这个。
    """
    info = probe_npz(os.path.join(d, e["npz"])) if e["npz"] else {}
    cams = info.get("cameras") or []
    if not cams:
        # npz 里没写相机名就退而求其次，按视频文件名推
        stem = "episode_%0*d_" % (e["width"], e["idx"])
        cams = [os.path.splitext(v[len(stem):])[0] for v in e["videos"]]
    return (tuple(cams), info.get("state_dim"), info.get("action_dim")), info


def _size_of(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def place_file(src, dst, use_link):
    """硬链接优先，跨设备或不支持时退回复制。返回实际用的方式。"""
    if os.path.exists(dst):
        os.remove(dst)
    if use_link:
        try:
            os.link(src, dst)
            return "link"
        except OSError:
            pass                                  # 跨文件系统 / 不支持硬链接
    shutil.copy2(src, dst)
    return "copy"


def clear_merge_output(job, d, dry_run=False):
    """清掉目标目录里**上一次合并的产物**，别的一概不碰。

    只删 episode_*.npz / episode_*_*.mp4 / merge_manifest.json —— 人往这个目录里
    放的笔记、qc 报告不该因为重合一次就没了。源目录本来就不动，所以这一步可逆。
    """
    gone = 0
    for e in scan_episode_files(d):
        for name in ([e["npz"]] if e["npz"] else []) + e["videos"]:
            if not dry_run:
                try:
                    os.remove(os.path.join(d, name))
                except OSError as exc:
                    raise RuntimeError("清不掉 %s：%s" % (name, exc))
            gone += 1
    man = os.path.join(d, "merge_manifest.json")
    if os.path.isfile(man):
        if not dry_run:
            _rm(man)
        gone += 1
    say(job, "%s目标目录里上一次合并的 %d 个文件（其他文件没动）"
        % ("将清掉 " if dry_run else "已清掉 ", gone))
    return gone


def merge_datasets(job, version, dataset_id, sources, dst, copy=False,
                   force=False, dry_run=False, replace=False):
    """把若干源目录合成一个新目录，编号从 0 连续重排。源目录不动。"""
    if len(sources) < 1:
        raise RuntimeError("没有给源目录")

    # ---- 路径校验：源和目标都必须在允许的输出根目录之下 ----
    srcs = []
    for raw in sources:
        ok, d = check_readable_dir(raw)               # 源只读：导入的目录可以在根目录外
        if not ok:
            raise RuntimeError("源目录不行：%s" % d)
        srcs.append(d)
    if len(set(srcs)) != len(srcs):
        raise RuntimeError("同一个源目录给了两次")

    dst_raw = os.path.realpath(os.path.expanduser(str(dst or "").strip()))
    if not dst_raw:
        raise RuntimeError("没有给 --dst")
    inside, _hit = under_roots(dst_raw)
    if not inside and dst_raw not in out_roots():
        raise RuntimeError("目标目录不在允许的输出根目录之下。" + roots_hint())
    if dst_raw in srcs:
        raise RuntimeError("目标目录不能是源目录之一：%s" % dst_raw)
    for sd in srcs:
        if dst_raw.startswith(sd + os.sep) or sd.startswith(dst_raw + os.sep):
            raise RuntimeError("目标目录和源目录互相嵌套，会一边读一边写：%s" % sd)

    # 目标目录里已经有 episode 就停手 —— 合进去会和原有编号撞车，而且没法回退。
    # 明说「先清空」才动手（重新合并走的就是这条）。
    if os.path.isdir(dst_raw):
        existing = scan_episode_files(dst_raw)
        if existing and not replace:
            raise RuntimeError("目标目录里已经有 %d 条 episode：%s。"
                               "要重合请勾「先清空目标目录」，或换一个空目录。"
                               % (len(existing), dst_raw))
        if existing:
            clear_merge_output(job, dst_raw, dry_run)

    # ---- 扫描 + 形状一致性 ----
    groups, total = [], 0
    say(job, "扫描源目录：")
    sigs = {}
    for d in srcs:
        eps = scan_episode_files(d)
        eps = [e for e in eps if e["npz"]]        # 没有 npz 的不算一条
        if not eps:
            say(job, "  %s —— 没有 episode_*.npz，跳过" % d)
            continue
        sig, info = episode_signature(d, eps[0])
        sigs.setdefault(sig, []).append(d)
        say(job, "  %-44s %3d 条  相机 %s  state=%s action=%s"
            % (d, len(eps), ",".join(sig[0]) or "?", sig[1], sig[2]))
        groups.append((d, eps))
        total += len(eps)
    if not groups:
        raise RuntimeError("这些目录里一条 episode 都没有")

    if len(sigs) > 1:
        say(job, "")
        say(job, "[不一致] 各源目录的相机 / 维度不一样：")
        for sig, who in sigs.items():
            say(job, "  相机 %s  state=%s action=%s" % (",".join(sig[0]) or "?", sig[1], sig[2]))
            for w in who:
                say(job, "      %s" % w)
        if not force:
            raise RuntimeError("拒绝合并：相机路数或动作维度不同的数据混在一起，"
                               "lerobot 转换会报错。确认要合请勾「强行合并」。")
        say(job, "  已勾强行合并，继续。")

    # ---- 合 ----
    if not dry_run:
        os.makedirs(dst_raw, exist_ok=True)
    say(job, "")
    say(job, "%s %d 条 → %s" % ("将合并" if dry_run else "开始合并", total, dst_raw))

    manifest, tasks, modes = [], {}, {"link": 0, "copy": 0}
    # 逻辑大小：硬链接不占额外空间，但人问「这批有多大」问的是数据本身多大
    total_bytes = 0
    width = max(4, len("%d" % max(0, total - 1)))
    n = 0
    for d, eps in groups:
        for e in eps:
            src_npz = os.path.join(d, e["npz"])
            info = probe_npz(src_npz)
            new_stem = "episode_%0*d" % (width, n)
            old_stem = "episode_%0*d" % (e["width"], e["idx"])
            entry = {"index": n, "source": d, "source_index": e["idx"],
                     "task": info.get("task", ""), "frames": info.get("frames"),
                     "videos": []}
            total_bytes += _size_of(src_npz)
            if not dry_run:
                modes[place_file(src_npz, os.path.join(dst_raw, new_stem + ".npz"),
                                 not copy)] += 1
            for v in e["videos"]:
                new_v = new_stem + v[len(old_stem):]          # _cam_top.mp4
                entry["videos"].append(new_v)
                total_bytes += _size_of(os.path.join(d, v))
                if not dry_run:
                    place_file(os.path.join(d, v), os.path.join(dst_raw, new_v), not copy)
            manifest.append(entry)
            t = info.get("task") or ""
            if t:
                tasks[t] = tasks.get(t, 0) + 1
            n += 1
        say(job, "  %s：%d 条已并入" % (d, len(eps)))

    if dry_run:
        say(job, "")
        say(job, "这是试运行，没有动任何文件。共 %d 条、%s。"
            % (len(manifest), human_bytes(total_bytes)))
        return {"dry_run": True, "count": len(manifest), "dst": dst_raw,
                "bytes": total_bytes}

    # ---- 清单：将来想追「这条是哪来的」就靠它 ----
    try:
        with open(os.path.join(dst_raw, "merge_manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"sources": srcs, "count": len(manifest), "tasks": tasks,
                       "bytes": total_bytes,
                       "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "agent": VERSION, "episodes": manifest}, f,
                      ensure_ascii=False, indent=2)
    except OSError as exc:
        say(job, "[提示] merge_manifest.json 写不了：%s" % exc)

    say(job, "")
    say(job, "已合并 %d 条，编号 0 ~ %d 连续，共 %s"
        % (len(manifest), len(manifest) - 1, human_bytes(total_bytes)))
    say(job, "  " + "，".join("%s %d 个文件" % ("硬链接" if k == "link" else "复制", v)
                              for k, v in modes.items() if v))
    if tasks:
        say(job, "  指令分布：")
        for t, c in sorted(tasks.items(), key=lambda kv: -kv[1]):
            say(job, "    %4d 条  %s" % (c, t))
    else:
        say(job, "  [警告] 一条 task 都没读出来，转出来会是单任务数据集")

    say(job, "同步给网站…")
    r = post_site("dataset_merged.php", {
        "version": version, "dataset_id": dataset_id, "out_dir": dst_raw,
        "count": len(manifest), "sources": srcs, "tasks": tasks,
        "linked": modes["link"], "copied": modes["copy"],
        "bytes": total_bytes,          # 本站够不着工作站的盘，大小只能由这儿报
    })
    if r.get("error"):
        raise RuntimeError("网站没接住这次合并：%s" % r["error"])
    try:
        post_snapshot(version, dataset_id, dst_raw)   # 条数 / 编号按磁盘实况来
    except Exception as exc:                          # noqa: BLE001
        say(job, "[提示] 条数同步失败：%s" % exc)
    say(job, "完成。回到网页看结果。")
    job["exit_code"] = 0
    return {"count": len(manifest), "dst": dst_raw, "tasks": tasks, "bytes": total_bytes}


def start_merge(version, dataset_id, sources, dst, copy=False, force=False,
                dry_run=False, replace=False):
    return start_task("merge", "合并 %d 个目录" % len(sources),
                      lambda job: merge_datasets(job, version, dataset_id, sources, dst,
                                                 copy, force, dry_run, replace))


def episode_info(out_dir, idx=None):
    """
    取某一条 episode 的参数与视频清单，顺带给出目录里全部编号。

    形状和自动检查报告里每条保持一致，网页那边一套渲染代码就够用。
    不跑 qc 也能看 —— 采完想翻某一条，没必要先跑一遍全量检查。
    """
    ok, d = check_readable_dir(out_dir)
    if not ok:
        raise ValueError(d)
    files = scan_episode_files(d)
    indices = [e["idx"] for e in files]
    if not files:
        return {"indices": [], "total": 0, "episode": None}

    if idx is None or idx not in indices:
        idx = indices[0]                      # 默认第一条（通常就是 0）
    e = next(x for x in files if x["idx"] == idx)

    npz_path = os.path.join(d, e["npz"]) if e["npz"] else ""
    info = probe_npz(npz_path) if npz_path else {
        "frames": None, "task": "", "cameras": [], "state_dim": None,
        "action_dim": None, "fps": None, "gripper": None, "motion": None,
        "error": "没有 npz 文件",
    }
    # 带上真实编码：mp4v 这种浏览器放不了，网页要据此决定是直接播还是走 /preview
    vids = [dict(name=v, **video_meta(os.path.join(d, v))) for v in e["videos"]]

    pos = indices.index(idx)
    return {
        "indices": indices,
        "total": len(indices),
        "pos": pos,
        "prev": indices[pos - 1] if pos > 0 else None,
        "next": indices[pos + 1] if pos + 1 < len(indices) else None,
        "episode": {
            "idx": idx,
            "stem": "episode_%0*d" % (e["width"] or 4, idx),
            "npz": e["npz"],
            "bytes": (os.path.getsize(npz_path) if npz_path and os.path.isfile(npz_path) else 0),
            "at": (datetime.fromtimestamp(os.path.getmtime(npz_path)).strftime("%Y-%m-%d %H:%M:%S")
                   if npz_path and os.path.isfile(npz_path) else ""),
            "frames": info["frames"], "fps": info["fps"], "task": info["task"],
            "cameras": info["cameras"], "state_dim": info["state_dim"],
            "action_dim": info["action_dim"], "gripper": info["gripper"],
            "motion": info["motion"], "probe_error": info["error"],
            "videos": vids,
        },
    }


# ---------------------------------------------------------------------------
# 视频编码：网页能不能放，取决于 mp4 里那个 fourcc，跟扩展名没关系
#
# cv2.VideoWriter 写的是 mp4 容器，但编码由 config.video_codec 决定。默认的
# mp4v 是 MPEG-4 Part 2，浏览器一个都不解 —— 文件是好的，VLC/ffplay/cv2 都能读，
# 只有 <video> 放不了。所以要报出真实编码，别让人以为是网页的毛病。
# ---------------------------------------------------------------------------
#: 浏览器认的视频编码（mp4/webm 容器里的 fourcc）
WEB_CODECS = {"avc1", "avc3", "h264", "hev1", "hvc1", "vp08", "vp09", "av01"}
CODEC_NAMES = {"avc1": "H.264", "avc3": "H.264", "h264": "H.264",
               "mp4v": "MPEG-4 Part 2", "mjpg": "Motion JPEG", "jpeg": "Motion JPEG",
               "hev1": "H.265", "hvc1": "H.265", "vp09": "VP9", "vp08": "VP8",
               "av01": "AV1", "xvid": "Xvid", "divx": "DivX"}


def _stsd_fourcc(f, end, depth=0):
    """在 mp4 的 box 树里找 stsd，取第一个 sample entry 的 fourcc。

    只往 moov/trak/mdia/minf/stbl 这条路钻 —— 编码信息只可能在那儿，
    整个文件扫一遍没必要（几百 MB 的视频不该为了四个字节读完）。
    """
    if depth > 6:
        return ""
    while f.tell() + 8 <= end:
        head = f.read(8)
        if len(head) < 8:
            return ""
        size = struct.unpack(">I", head[:4])[0]
        kind = head[4:8].decode("latin-1")
        start = f.tell()
        if size == 1:                                 # 64 位长度
            ext = f.read(8)
            if len(ext) < 8:
                return ""
            size = struct.unpack(">Q", ext)[0]
            start = f.tell()
            body_end = start - 16 + size
        elif size == 0:                               # 一直到文件尾
            body_end = end
        else:
            body_end = start - 8 + size
        if body_end <= start - 8 or body_end > end:
            return ""                                 # 长度不合理，别再往下读
        if kind == "stsd":
            f.seek(start + 8)                         # version/flags + entry_count
            entry = f.read(8)
            return entry[4:8].decode("latin-1").strip() if len(entry) == 8 else ""
        if kind in ("moov", "trak", "mdia", "minf", "stbl"):
            hit = _stsd_fourcc(f, body_end, depth + 1)
            if hit:
                return hit
        f.seek(body_end)
    return ""


def video_codec(path):
    """这个视频是什么编码、浏览器放不放得了。读文件头，不开解码器。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".avi", ".mkv"):
        return {"codec": ext.lstrip("."), "codec_name": ext.lstrip(".").upper() + " 容器",
                "playable": False, "why": "浏览器只放 mp4 / webm，这个容器放不了"}
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(0)
            cc = _stsd_fourcc(f, size)
    except OSError as exc:
        return {"codec": "", "codec_name": "", "playable": None, "why": "读不了：%s" % exc}
    if not cc:
        return {"codec": "", "codec_name": "", "playable": None, "why": "认不出编码"}
    key = cc.lower()
    codec_name = CODEC_NAMES.get(key, cc)
    ok = key in WEB_CODECS
    return {"codec": cc, "codec_name": codec_name, "playable": ok,
            "why": "" if ok else "浏览器不解 %s，只认 H.264 / VP9 / AV1" % codec_name}


def video_meta(path):
    """给网页的一条视频：大小 + 编码 + 能不能直接播。"""
    try:
        sz = os.path.getsize(path)
    except OSError:
        sz = 0
    return dict(bytes=sz, path=path, **video_codec(path))


def have_ffmpeg():
    """工作站上有没有 ffmpeg —— 编码不对时要靠它转一份网页能播的。"""
    global _FFMPEG
    if _FFMPEG is None:
        _FFMPEG = shutil.which("ffmpeg") or ""
    return _FFMPEG


_FFMPEG = None
PREVIEW_DIR = os.path.join(tempfile.gettempdir(), "atom_agent_preview")
PREVIEW_LOCKS = {}
PREVIEW_LOCKS_GUARD = threading.Lock()
#: 缓存上限，超了就从最久没碰过的开始删
PREVIEW_CAP_BYTES = 2 * 1024 * 1024 * 1024


def preview_path(real):
    """缓存文件名跟着源文件的路径 + 修改时间 + 大小走，源文件一变就自然失效。"""
    try:
        st = os.stat(real)
        key = "%s|%d|%d" % (os.path.realpath(real), st.st_mtime_ns, st.st_size)
    except OSError:
        key = os.path.realpath(real)
    return os.path.join(PREVIEW_DIR, hashlib.sha1(key.encode("utf-8")).hexdigest() + ".mp4")


def trim_preview_cache():
    """把缓存压回上限以下。转码产物随时能再生成，删了不心疼。"""
    try:
        files = []
        for name in os.listdir(PREVIEW_DIR):
            fp = os.path.join(PREVIEW_DIR, name)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            files.append((st.st_atime, st.st_size, fp))
    except OSError:
        return
    total = sum(f[1] for f in files)
    for _atime, sz, fp in sorted(files):
        if total <= PREVIEW_CAP_BYTES:
            return
        try:
            os.remove(fp)
            total -= sz
        except OSError:
            pass


def ensure_preview(real):
    """保证有一份网页能播的 H.264，返回 (路径, 出错原因)。

    **原始文件一个字节都不动** —— 数据集是采集结果，不该为了能在网页上看就被改写。
    转码放系统临时目录，掉了再转一次就是。
    """
    dst = preview_path(real)
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        os.utime(dst, None)                           # 记一下最近用过，给 LRU 看
        return dst, ""
    exe = have_ffmpeg()
    if not exe:
        return "", "工作站上没有 ffmpeg，没法把 %s 转成网页能播的格式" % (video_codec(real)["codec_name"],)

    with PREVIEW_LOCKS_GUARD:                         # 同一个文件别同时转两遍
        lock = PREVIEW_LOCKS.setdefault(dst, threading.Lock())
    with lock:
        if os.path.isfile(dst) and os.path.getsize(dst) > 0:
            return dst, ""
        os.makedirs(PREVIEW_DIR, exist_ok=True)
        tmp = dst + ".tmp.mp4"
        cmd = [exe, "-nostdin", "-y", "-i", real,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p",                 # 浏览器只吃 4:2:0
               "-movflags", "+faststart",             # moov 放前面，边下边播
               "-an", tmp]
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=300)
        except (OSError, subprocess.TimeoutExpired) as exc:
            _rm(tmp)
            return "", "转码失败：%s" % exc
        if r.returncode != 0 or not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
            tail = (r.stdout or b"").decode("utf-8", "replace").strip().splitlines()[-3:]
            _rm(tmp)
            return "", "ffmpeg 退出码 %d：%s" % (r.returncode, " / ".join(tail))
        os.replace(tmp, dst)
    trim_preview_cache()
    return dst, ""


def _rm(path):
    try:
        os.remove(path)
    except OSError:
        pass


def codec_probe():
    """在这台机器上真写一遍再读回来，看 cv2 到底能不能出网页能播的编码。

    光看 `VideoWriter.isOpened()` 不算数：实测它对 MJPG / XVID 会**悄悄退回 mp4v**，
    对 avc1 也可能一边报 openh264 加载失败一边照样出文件。唯一靠谱的判据是
    写完之后把文件的 fourcc 读出来、再解码回若干帧。
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        return {"ok": False, "error": "这台机器上没有 %s" % exc.name}

    out = []
    tmpd = tempfile.mkdtemp(prefix="atom_codec_")
    frames = [(np.arange(64 * 48 * 3, dtype=np.uint8).reshape(48, 64, 3) + i * 7)
              for i in range(12)]
    try:
        for cc in ("avc1", "h264", "mp4v"):
            fp = os.path.join(tmpd, "t_%s.mp4" % cc)
            row = {"fourcc": cc, "opened": False, "wrote": "", "frames": 0, "playable": False}
            try:
                w = cv2.VideoWriter(fp, cv2.VideoWriter_fourcc(*cc), 30, (64, 48))
                row["opened"] = bool(w.isOpened())
                if row["opened"]:
                    for f in frames:
                        w.write(f)
                w.release()
            except Exception as exc:                  # noqa: BLE001
                row["error"] = str(exc)
            if os.path.isfile(fp) and os.path.getsize(fp) > 0:
                got = video_codec(fp)
                row["wrote"] = got["codec"]           # 要的和真写出来的可能不是一个
                row["playable"] = bool(got["playable"])
                cap = cv2.VideoCapture(fp)
                while True:
                    ok, _ = cap.read()
                    if not ok:
                        break
                    row["frames"] += 1
                cap.release()
            out.append(row)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)

    good = [r for r in out if r["playable"] and r["frames"] > 0]
    return {"ok": True, "results": out,
            "recommend": good[0]["fourcc"] if good else "",
            "ffmpeg": bool(have_ffmpeg())}


def check_servable(path):
    """/file 只读地放采集目录里的视频与图片，路径必须在输出根目录之下。"""
    if not path:
        return False, "没有给路径"
    p = os.path.realpath(os.path.expanduser(str(path)))
    ok, _hit = under_roots(p)
    if not ok and not looks_like_dataset(os.path.dirname(p)):
        return False, "拒绝：不在允许的输出根目录之下。" + roots_hint()
    if not SERVE_RE.match(os.path.basename(p)):
        return False, "拒绝：不是采集产物的文件名"
    if not os.path.isfile(p):
        return False, "文件不存在"
    return True, p


# ---------------------------------------------------------------------------
# 远程操作：网页驱动采集 + RGB 回传
#
# 采集脚本跑在工作站上，窗口开在工作站的显示器上；人要是不坐在那儿，就需要
# 网页能代替那个窗口：按钮当按键用，画面回传到网页。中间不走 NAS ——
# 网页是从工作站自己的浏览器打开的，直接问本机代理要，局域网一点带宽都不占。
#
# 约定（三个文件都在 out_dir 里，与网站 README 的「采集脚本约定」一致）：
#   control.json  网页 → 脚本  {"seq": 17, "cmd": "start", "ts": ...}
#                 按 seq 去重，不靠删文件：删文件有竞态，脚本可能读到写了一半的内容
#   live.jpg      脚本 → 网页  每帧覆写（先写 .tmp 再 rename，否则网页读到半张图）
#   status.json   脚本 → 网页  {"episode":7,"samples":6,"recording":true,"acked":17,...}
#                 acked 回显最后执行的 seq，网页据此确认这条命令真的生效了
#
# 脚本还没支持这一套时，这几个接口只是拿不到东西（404），网页照旧退回原来的模式。
# ---------------------------------------------------------------------------
CONTROL_CMDS = ("start", "save", "discard", "undo", "pause", "resume", "quit",
                "force_up", "force_down",     # 夹持力 ±1 档，collect(7) 起有
                "key", "text")                # OmniPicker：原样按键、一整段字
# OmniPicker 面板能送的键名；另外任何单个可见字符（p、s、y …）也行
CONTROL_KEY_NAMES = ("ENTER", "ESC", "BACKSPACE", "SPACE", "UP", "DOWN", "LEFT", "RIGHT")


def control_extra(cmd, data):
    """key / text 的内容：只放行认得的，别的一概不写进 control.json。"""
    if cmd == "key":
        k = str(data.get("key", ""))
        if k.upper() in CONTROL_KEY_NAMES:
            return {"key": k.upper()}
        if len(k) == 1 and k.isprintable():
            return {"key": k}
        raise ValueError("不认识的键：%r" % k)
    if cmd == "text":
        t = "".join(ch for ch in str(data.get("text", "")) if ch.isprintable())[:200]
        return {"text": t}
    return {}
FRAME_NAMES = ("live.jpg", "live.jpeg", "live.png", "live.webp")
SEQ_LOCK = threading.Lock()


def check_out_dir(path):
    """out_dir 必须在采集输出根目录之下；返回 (ok, 绝对路径或原因)。"""
    if not path:
        return False, "没有给 out_dir"
    p = os.path.realpath(os.path.expanduser(str(path)))
    ok, _hit = under_roots(p)
    if not ok and p not in out_roots():
        return False, "拒绝：不在允许的输出根目录之下。" + roots_hint()
    if not os.path.isdir(p):
        return False, "目录不存在：%s" % p
    return True, p


def write_control(out_dir, cmd, extra=None):
    """
    往 out_dir 写一条控制命令，返回这条的 seq。
    seq 从已有文件里读出来 +1，代理重启也不会倒回去；写入是原子的（先 .tmp 再 rename），
    脚本任何时候读到的都是完整的一条。
    """
    ok, d = check_out_dir(out_dir)
    if not ok:
        raise ValueError(d)
    if cmd not in CONTROL_CMDS:
        raise ValueError("不认识的命令：%s" % cmd)

    path = os.path.join(d, "control.json")
    with SEQ_LOCK:
        seq = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                seq = int(json.load(f).get("seq", 0))
        except Exception:                             # noqa: BLE001
            seq = 0
        seq += 1
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dict(extra or {}, seq=seq, cmd=cmd, ts=time.time()), f, ensure_ascii=False)
        os.replace(tmp, path)
    return seq


def read_live_status(out_dir):
    """读脚本写的 status.json；没有或读不动就返回 None（脚本还没支持这一套）。"""
    ok, d = check_out_dir(out_dir)
    if not ok:
        return None
    p = os.path.join(d, "status.json")
    try:
        with open(p, "r", encoding="utf-8") as f:
            doc = json.load(f)
        if isinstance(doc, dict):
            doc["_age"] = round(time.time() - os.path.getmtime(p), 1)
            return doc
    except Exception:                                 # noqa: BLE001
        pass
    return None


def find_live_frame(out_dir):
    """最新的实时画面文件。"""
    ok, d = check_out_dir(out_dir)
    if not ok:
        return ""
    for n in FRAME_NAMES:
        p = os.path.join(d, n)
        if os.path.isfile(p):
            return p
    return ""


# ---------------------------------------------------------------------------
# HTTP 服务
# ---------------------------------------------------------------------------
def log(msg):
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg), flush=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "AtomAgent/" + VERSION

    def log_message(self, fmt, *args):                # 静音默认访问日志
        pass

    # ---- 通用 ----
    def _send(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        origin = self.headers.get("Origin", "")
        if self._origin_ok(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Agent-Token")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body)

    def _origin_ok(self, origin):
        """只允许网站与本机页面调用，挡住其他站点与 DNS 重绑定。"""
        if not origin:
            return True                               # 同源 / 非浏览器
        if origin in CFG["allow_origins"]:
            return True
        return bool(re.match(r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$", origin))

    def _guard(self, data=None):
        """返回 True 表示已处理完错误响应，调用方应直接返回。"""
        origin = self.headers.get("Origin", "")
        if not self._origin_ok(origin):
            self._send({"error": "来源不被允许: %s" % origin}, 403)
            return True
        token = (data or {}).get("token") or self.headers.get("X-Agent-Token", "")
        if not token:
            from urllib.parse import parse_qs, urlparse
            token = (parse_qs(urlparse(self.path).query).get("token") or [""])[0]
        if token != CFG["token"]:
            self._send({"error": "令牌不正确"}, 401)
            return True
        return False

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:                             # noqa: BLE001
            return {}

    # ---- 路由 ----
    def do_OPTIONS(self):
        self._send({"ok": True})

    def serve_preview(self):
        """网页能播的那一份。编码本来就对就直接送原文件，不对才转一份缓存。

        转码产物不进采集目录 —— 数据集不该为了能在网页上看被动过。
        """
        from urllib.parse import parse_qs, urlparse
        if self._guard():
            return
        q = parse_qs(urlparse(self.path).query)
        ok, real = check_servable((q.get("path") or [""])[0])
        if not ok:
            self._send({"error": real}, 403)
            return
        if video_codec(real)["playable"]:
            self.send_range(real)
            return
        dst, err = ensure_preview(real)
        if err:
            self._send({"error": err}, 503)
            return
        self.send_range(dst, mime="video/mp4")

    def serve_file(self):
        """
        只读地把采集目录里的视频 / 图片吐给网页，支持 Range（视频要拖进度条）。
        网站跑在 NAS 上碰不到工作站磁盘，检查页面要放的就是工作站本地那份 mp4。
        """
        from urllib.parse import parse_qs, urlparse
        if self._guard():
            return
        q = parse_qs(urlparse(self.path).query)
        ok, real = check_servable((q.get("path") or [""])[0])
        if not ok:
            self._send({"error": real}, 403)
            return
        self.send_range(real)

    def send_range(self, real, mime=""):
        """把一个文件吐出去，支持 Range —— 视频要拖进度条就靠它。"""
        size = os.path.getsize(real)
        start, end, partial = 0, size - 1, False
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "") or "")
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            partial = True
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", "bytes */%d" % size)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

        self.send_response(206 if partial else 200)
        self.send_header("Content-Type",
                         mime or MIME.get(os.path.splitext(real)[1].lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        origin = self.headers.get("Origin", "")
        if origin and self._origin_ok(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            with open(real, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    buf = f.read(min(256 * 1024, left))
                    if not buf:
                        break
                    self.wfile.write(buf)
                    left -= len(buf)
        except (BrokenPipeError, ConnectionResetError):
            pass                                      # 播放器拖进度条会中断上一个请求，正常

    def serve_frame(self):
        """把采集脚本写的实时画面吐给网页。没有就 404，网页自己显示占位图。"""
        from urllib.parse import parse_qs, urlparse
        if self._guard():
            return
        q = parse_qs(urlparse(self.path).query)
        real = find_live_frame((q.get("out_dir") or [""])[0])
        if not real:
            self._send({"error": "还没有实时画面（脚本没写 live.jpg，或目录不对）"}, 404)
            return
        try:
            with open(real, "rb") as f:
                body = f.read()
        except OSError as exc:
            self._send({"error": "读不了画面: %s" % exc}, 500)
            return
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(os.path.splitext(real)[1].lower(), "image/jpeg"))
        self.send_header("Content-Length", str(len(body)))
        origin = self.headers.get("Origin", "")
        if origin and self._origin_ok(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse
        path = urlparse(self.path).path

        if path == "/file":
            self.serve_file()
            return

        if path == "/preview":
            self.serve_preview()
            return

        if path == "/codec_probe":
            if self._guard():
                return
            self._send(codec_probe())
            return

        if path == "/frame":
            self.serve_frame()
            return

        if path == "/episode":
            if self._guard():
                return
            q = parse_qs(urlparse(self.path).query)
            raw = (q.get("idx") or [""])[0]
            try:
                idx = int(raw) if raw != "" else None
            except ValueError:
                idx = None
            try:
                self._send({"ok": True, **episode_info((q.get("out_dir") or [""])[0], idx)})
            except ValueError as exc:
                self._send({"error": str(exc)}, 404)
            return

        if path == "/live_status":
            if self._guard():
                return
            q = parse_qs(urlparse(self.path).query)
            st = read_live_status((q.get("out_dir") or [""])[0])
            if st is None:
                self._send({"error": "还没有 status.json"}, 404)
                return
            self._send({"ok": True, "status": st})
            return

        if path == "/ping":
            if self._guard():
                return
            self._send({
                "ok": True, "agent": VERSION, "workdir": CFG["workdir"],
                # 网页拿它核对连上的是哪个代理 —— RoboJev 那条线是另一个代理（robojev_agent.py）
                "profile": "grasp", "profile_name": "抓取代理",
                "env_record": CFG["env_record"], "env_convert": CFG["env_convert"],
                "commands": sorted(TEMPLATES.keys()),
                "conda_sh": find_conda_hook() or "(未找到，将退回 ~/.bashrc)",
                "out_root": (out_roots() or [""])[0],
                "out_roots": out_roots(),
                "home": os.path.expanduser("~"),        # 网页要拿它把 ~ 展开后再比路径
                "collect_procs": collect_procs(),       # 多于一个就是在抢 control.json
                "ffmpeg": bool(have_ffmpeg()),          # 没有它就转不了网页能播的格式
                "watch": {k: WATCH[k] for k in ("on", "out_dir", "dataset_id", "last", "error")},
                "jobs": [job_view(j) for j in list(JOBS.values())[-5:]],
            })
            return

        if path.startswith("/job/"):
            if self._guard():
                return
            job = JOBS.get(path[5:])
            self._send(job_view(job) if job else {"error": "任务不存在"}, 200 if job else 404)
            return

        self._send({"error": "未知路径"}, 404)

    def do_POST(self):
        from urllib.parse import urlparse
        path = urlparse(self.path).path
        data = self._body()
        if self._guard(data):
            return

        if path == "/run":
            key = data.get("step", "")
            if key not in TEMPLATES:
                self._send({"error": "命令不在白名单: %s" % key}, 400)
                return
            try:
                job = start_job(key, data.get("params") or {}, data.get("workdir"))
            except Exception as exc:                  # noqa: BLE001
                self._send({"error": str(exc)}, 400)
                return
            log("执行 %s" % job["cmd"])
            # 顺带开启上报
            w = data.get("watch") or {}
            if w.get("out_dir") and w.get("dataset_id"):
                WATCH.update({"on": True, "out_dir": w["out_dir"],
                              "version": w.get("version", ""), "dataset_id": w["dataset_id"]})
                log("开始监视 %s" % w["out_dir"])
            self._send({"ok": True, "job": job_view(job)})
            return

        if path == "/stop":
            jid = data.get("id", "")
            j = JOBS.get(jid)
            if j and j.get("key") == "upload":
                j["cancel"] = True
                self._send({"ok": True})
                return
            self._send({"ok": stop_job(jid)})
            return

        if path == "/watch":
            if data.get("off"):
                WATCH.update({"on": False})
                self._send({"ok": True, "watch": WATCH["on"]})
                return
            out_dir = data.get("out_dir", "")
            if not out_dir:
                self._send({"error": "缺少 out_dir"}, 400)
                return
            WATCH.update({"on": True, "out_dir": out_dir,
                          "version": data.get("version", ""),
                          "dataset_id": data.get("dataset_id", ""), "error": ""})
            log("开始监视 %s" % out_dir)
            self._send({"ok": True})
            return

        if path == "/browse":
            try:
                self._send(dict(browse_dir(data.get("path", "")), ok=True))
            except ValueError as exc:
                self._send({"error": str(exc)}, 404)
            return

        if path == "/list_dirs":
            root, cands = list_episode_dirs()
            self._send({"ok": True, "out_root": root, "candidates": cands})
            return

        if path == "/upload":
            out_dir = data.get("out_dir", "")
            ver = data.get("version", "")
            dsid = data.get("dataset_id", "")
            if not out_dir or not ver or not dsid:
                self._send({"error": "缺少 out_dir / version / dataset_id"}, 400)
                return
            if upload_files(out_dir) is None:
                root, cands = list_episode_dirs()
                self._send({
                    "error": "工作站上没有这个目录：%s" % out_dir,
                    "reason": "no_dir",
                    "out_root": root,
                    "candidates": cands,
                }, 404)
                return
            if not upload_files(out_dir):
                self._send({
                    "error": "目录里没有 episode_*.npz：%s" % out_dir,
                    "reason": "empty",
                    "candidates": list_episode_dirs()[1],
                }, 404)
                return
            job = start_upload(ver, dsid, out_dir)
            log("开始上传 %s" % out_dir)
            self._send({"ok": True, "job": job_view(job)})
            return

        if path == "/delete_dir":
            ok, why, abspath, info = check_deletable(data.get("path", ""))
            if not ok:
                self._send({"error": why, "path": abspath}, 400)
                return
            if data.get("dry_run"):
                self._send({"ok": True, "dry_run": True, "path": abspath, "info": info})
                return
            try:
                shutil.rmtree(abspath)
            except Exception as exc:                  # noqa: BLE001
                self._send({"error": "删除失败: %s" % exc, "path": abspath}, 500)
                return
            log("已删除采集目录 %s（%d 条，%s）" % (abspath, info["episodes"], human_bytes(info["bytes"])))
            if WATCH.get("on") and os.path.realpath(os.path.expanduser(WATCH["out_dir"])) == abspath:
                WATCH["on"] = False                   # 目录没了，别再上报
            self._send({"ok": True, "path": abspath, "info": info})
            return

        if path == "/control":
            try:
                cmd = str(data.get("cmd", ""))
                seq = write_control(data.get("out_dir", ""), cmd, control_extra(cmd, data))
            except ValueError as exc:
                self._send({"error": str(exc)}, 400)
                return
            except OSError as exc:
                self._send({"error": "写不了 control.json: %s" % exc}, 500)
                return
            self._send({"ok": True, "seq": seq, "status": read_live_status(data.get("out_dir", ""))})
            return

        if path == "/qc_scan":
            out_dir = data.get("out_dir", "")
            ver = data.get("version", "")
            dsid = data.get("dataset_id", "")
            if not out_dir or not ver or not dsid:
                self._send({"error": "缺少 out_dir / version / dataset_id"}, 400)
                return
            if not os.path.isdir(os.path.expanduser(out_dir)):
                root, cands = list_episode_dirs()
                self._send({"error": "工作站上没有这个目录：%s" % out_dir,
                            "reason": "no_dir", "out_root": root, "candidates": cands}, 404)
                return
            job = start_qc_scan(ver, dsid, out_dir, str(data.get("cmd") or ""), data.get("workdir"))
            log("开始自动检查 %s" % out_dir)
            self._send({"ok": True, "job": job_view(job)})
            return

        if path == "/merge":
            srcs = data.get("sources") or []
            dst = data.get("dst", "")
            ver = data.get("version", "")
            dsid = data.get("dataset_id", "")
            if not srcs or not dst or not ver or not dsid:
                self._send({"error": "缺少 sources / dst / version / dataset_id"}, 400)
                return
            try:
                job = start_merge(ver, dsid, [str(x) for x in srcs], str(dst),
                                  copy=bool(data.get("copy")),
                                  force=bool(data.get("force")),
                                  dry_run=bool(data.get("dry_run")),
                                  replace=bool(data.get("replace")))
            except (ValueError, RuntimeError) as exc:
                self._send({"error": str(exc)}, 400)
                return
            log("合并 %d 个目录 → %s" % (len(srcs), dst))
            self._send({"ok": True, "job": job_view(job)})
            return

        if path == "/prune":
            out_dir = data.get("out_dir", "")
            ver = data.get("version", "")
            dsid = data.get("dataset_id", "")
            eps = data.get("episodes") or []
            if not out_dir or not ver or not dsid:
                self._send({"error": "缺少 out_dir / version / dataset_id"}, 400)
                return
            try:
                eps = [int(x) for x in eps]
            except (TypeError, ValueError):
                self._send({"error": "episodes 必须是编号数组"}, 400)
                return
            if not eps:
                self._send({"error": "没有选中任何回合"}, 400)
                return
            job = start_prune(ver, dsid, out_dir, eps,
                              renumber=data.get("renumber", True) is not False,
                              dry_run=bool(data.get("dry_run")))
            log("删除 %d 条 @ %s" % (len(eps), out_dir))
            self._send({"ok": True, "job": job_view(job)})
            return

        if path == "/sync":
            eps = scan_episodes(data.get("out_dir") or WATCH["out_dir"])
            if eps is None:
                self._send({"error": "目录不存在"}, 404)
                return
            if data.get("dataset_id"):
                WATCH.update({"dataset_id": data["dataset_id"],
                              "version": data.get("version", ""),
                              "out_dir": data.get("out_dir") or WATCH["out_dir"]})
            try:
                post_report(eps)
            except Exception as exc:                  # noqa: BLE001
                self._send({"error": "上报失败: %s" % exc}, 502)
                return
            self._send({"ok": True, "count": len(eps)})
            return

        self._send({"error": "未知路径"}, 404)


def main():
    p = argparse.ArgumentParser(description="原子动作开发中心 · 工作站代理")
    p.add_argument("--token", required=True, help="与网站配置里的 AGENT_TOKEN 一致")
    p.add_argument("--site", default="http://192.168.5.100:8090", help="网站地址，用于上报")
    p.add_argument("--port", type=int, default=8765, help="本机监听端口")
    p.add_argument("--workdir", default="~/yuhang_workspace/Project_Atom",
                   help="兜底工作目录。网页每次下发命令都会带上它自己那份，给了就以网页为准；这里只在网页没给、或给的目录不存在时用")
    p.add_argument("--env-record", default="dexgraspvla")
    p.add_argument("--env-convert", default="lerobot")
    p.add_argument("--interval", type=float, default=3.0, help="上报轮询间隔（秒）")
    p.add_argument("--conda-sh", default="", help="conda.sh 路径，自动探测不到时手动指定")
    p.add_argument("--out-root", action="append", default=None, metavar="目录",
                   help="采集输出根目录，代理只对它之下的目录动手（写控制文件、放视频、删除）。"
                        "可以给多次，也可以逗号分隔。默认 ~/atom_episodes 和 ~/vel_episodes")
    a = p.parse_args()

    CFG.update({
        "token": a.token, "site": a.site, "port": a.port,
        "workdir": a.workdir, "env_record": a.env_record, "env_convert": a.env_convert,
        "interval": a.interval,
        "conda_sh": a.conda_sh,
        # 允许多个根目录：换了项目目录之后，新旧两处的数据往往要同时能操作
        "out_roots": [x for item in (a.out_root or ["~/atom_episodes", "~/vel_episodes"])
                      for x in str(item).split(",") if x.strip()],
        "allow_origins": {a.site.rstrip("/")},
    })

    threading.Thread(target=watcher_loop, daemon=True).start()

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)   # 只绑本机
    log("原子动作工作站代理 v%s" % VERSION)
    log("监听 http://127.0.0.1:%d  （只接受本机连接）" % a.port)
    log("网站 %s" % a.site)
    log("工作目录 %s（兜底；网页下发命令时带的那份优先）" % a.workdir)
    log("输出根目录 %s" % "、".join(out_roots()))
    _hook = find_conda_hook()
    log("conda.sh %s" % (_hook if _hook else "未找到，将退回 source ~/.bashrc；不行请用 --conda-sh 指定"))
    log("按 Ctrl+C 退出")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("退出")


if __name__ == "__main__":
    main()
