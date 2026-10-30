#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
远程操作小模块（放在采集脚本旁边，import 一下就行）
=====================================================

让 `atom_robot.ar5.collect` 能被网页驱动：网页上的按钮当按键用，
画面回传到网页。工作站上可以完全不开窗口，人不用坐在工作站前。

链路（三个文件都在 --out 目录里，不经过 NAS）：

    网页按钮 → 工作站代理写 control.json → 采集脚本轮询消费
    采集脚本写 live.jpg / status.json → 代理读 → 网页每 250ms 拉一次

改采集脚本只有三处，都不碰原来的状态机：

    from remote_ctl import RemoteLink              # 1) 引入
    link = RemoteLink(args.out)                    #    开录前建一个

    while True:
        frame = ...                                 # 你原来合成好的那张图
        link.publish_frame(frame)                   # 2) 发画面（内部限流，默认 10 fps）

        key = cv2.waitKey(1) & 0xFF if args.window else 255
        cmd = link.poll_command()                   # 3) 收网页来的命令
        if cmd:                                     #    把它当成按键喂给原来的分支
            key = {"start": ord(' '), "save": ord('s'), "discard": ord('r'),
                   "undo": 81, "quit": ord('q')}.get(cmd, 255)

        ...原来的按键分支不动...

        link.publish_status(episode=ep, samples=n, recording=is_recording,
                            fps=fps_measured, frames=len(buf))

要点：

* **命令按 seq 去重，不靠删文件。** 删文件有竞态：脚本可能读到写了一半的内容，
  或者两条命令挤在一起丢掉一条。这里网页每发一条 seq +1，脚本只认比自己见过的更大的那个。
* **画面先写 .tmp 再 rename。** 直接覆写 live.jpg 的话，网页有相当概率读到半张图。
* **发帧有限流，默认 10 fps。** 采集回路里塞 IO 是有代价的：文档写明写盘 600 帧两路相机
  要 2.21 秒，而臂 0.25 秒收不到目标就 latch。编码一张 640 宽的 jpg 大约 1~3 ms，
  10 fps 就是每秒 10~30 ms，均摊到 30 Hz 的回路里可以接受；嫌多就把 fps 调小，
  或者 `RemoteLink(out, frame_fps=0)` 彻底关掉画面回传，只留按钮。
* **poll_command 只碰文件系统，不发网络请求**，不会因为网断了把采集卡住。
"""

import json
import os
import time

try:
    import cv2
except Exception:                                     # noqa: BLE001
    cv2 = None


class RemoteLink:
    """采集脚本与网页之间的那层文件接口。"""

    #: 网页能发的命令 → 建议映射到的原按键
    # force_up / force_down：夹持力 ±1 档（collect 里映射到 ↑ ↓，只有能调力的末端认）
    # key / text：OmniPicker 用，送原样按键和一整段字，内容在 self.last 里
    COMMANDS = ("start", "save", "discard", "undo", "pause", "resume", "quit",
                "force_up", "force_down", "key", "text")

    def __init__(self, out_dir, frame_fps=10.0, jpeg_quality=70, max_width=960):
        self.dir = os.path.expanduser(str(out_dir))
        os.makedirs(self.dir, exist_ok=True)
        self.control_path = os.path.join(self.dir, "control.json")
        self.frame_path = os.path.join(self.dir, "live.jpg")
        self.status_path = os.path.join(self.dir, "status.json")

        self.frame_interval = (1.0 / frame_fps) if frame_fps and frame_fps > 0 else 0
        self.jpeg_quality = int(jpeg_quality)
        self.max_width = int(max_width)

        self._last_frame_at = 0.0
        self._last_status_at = 0.0
        # 开工时把已有的 seq 记下来：代理重启过、或者上一轮采集留下的命令，不该被当成新命令执行
        self._seen_seq = self._read_seq()
        self._acked = self._seen_seq
        # 最近一条命令的整条内容（OmniPicker 的 key / text 要从这里取）。
        # poll_command() 还是只返回命令名，老的采集脚本一行不用改
        self.last = {}

    # ---------------- 网页 → 脚本 ----------------

    def _read_seq(self):
        try:
            with open(self.control_path, "r", encoding="utf-8") as f:
                return int(json.load(f).get("seq", 0))
        except Exception:                             # noqa: BLE001
            return 0

    def poll_command(self):
        """
        有新命令就返回它（'start' / 'save' / 'discard' / 'undo' / 'quit' …），没有返回 None。
        每轮调一次，开销是一次 stat + 一次小文件读。
        """
        try:
            if not os.path.isfile(self.control_path):
                return None
            with open(self.control_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:                             # noqa: BLE001
            return None                               # 正在被写，下一轮再读

        seq = int(doc.get("seq", 0))
        if seq <= self._seen_seq:
            return None
        cmd = str(doc.get("cmd", ""))
        self._seen_seq = seq
        if cmd not in self.COMMANDS:
            return None
        self._acked = seq                             # 回显给网页：这条已经收到并执行了
        self.last = doc
        return cmd

    # ---------------- 脚本 → 网页 ----------------

    def frame_due(self, interval=None):
        """现在发一帧的话会不会被限流挡掉。

        采集脚本待机时本来不取画面（省一次相机 + 几次 SDK 往返），所以得先问一句
        「这一帧值不值得取」，再决定要不要 observe —— 取完才发现被限流挡了就白取了。
        """
        want = self.frame_interval if interval is None else interval
        return bool(want and want > 0 and (time.time() - self._last_frame_at) >= want)

    def publish_frame(self, image, force=False, rgb=False):
        """
        把一帧画面写成 live.jpg 给网页看。内部限流，超过设定帧率的调用直接返回。
        image 是 ndarray；相机驱动给的是 RGB，传 rgb=True 让它转成 BGR 再编码，
        否则网页上人是蓝的、蓝色套筒是红的。
        """
        if cv2 is None or image is None or self.frame_interval <= 0:
            return False
        now = time.time()
        if not force and now - self._last_frame_at < self.frame_interval:
            return False
        self._last_frame_at = now

        try:
            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if rgb else image
            h, w = img.shape[:2]
            if self.max_width and w > self.max_width:   # 网页上就那么大，传原图没意义
                scale = self.max_width / float(w)
                img = cv2.resize(img, (self.max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                return False
            tmp = self.frame_path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(buf.tobytes())
            os.replace(tmp, self.frame_path)            # 原子替换，网页不会读到半张图
            return True
        except Exception:                               # noqa: BLE001
            return False                                # 画面是附加功能，绝不能因为它中断采集

    def publish_status(self, min_interval=0.3, **fields):
        """
        写 status.json 给网页轮询。常用字段：
            episode, samples, recording, fps, frames, finished, log
        acked 由本模块自己填（网页据此确认命令真的生效了）。
        """
        now = time.time()
        if min_interval and now - self._last_status_at < min_interval and not fields.get("finished"):
            return False
        self._last_status_at = now
        doc = dict(fields)
        doc["acked"] = self._acked
        doc["ts"] = now
        try:
            tmp = self.status_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False)
            os.replace(tmp, self.status_path)
            return True
        except Exception:                               # noqa: BLE001
            return False

    def close(self, **fields):
        """采集退出时叫一下，把 finished 标上，网页那边就知道该跳走了。"""
        fields.setdefault("finished", True)
        fields.setdefault("recording", False)
        self.publish_status(min_interval=0, **fields)
