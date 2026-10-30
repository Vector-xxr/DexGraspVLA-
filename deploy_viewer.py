"""deploy_ar5.py 的实时画面窗口：左边俯视画面叠掩码（蓝），右边腕部画面，底下一条状态。

**独立进程**，从共享内存读 deploy_ar5.py 写进来的最新画面，不加载模型、不碰硬件。
窗口卡了、被关了都影响不到部署程序 —— 10-22 的教训：部署进程里开窗口会卡住发关节目标的线程，
AR5 0.25 s 收不到目标就锁停。采集的预览（collect.py LivePreview）是同进程的线程，这里更保守一点，
因为部署进程里还有上游框选用的 Tk 窗口，不想 Tk 和 OpenCV 的 Qt 在同一个进程里。

由 deploy_ar5.py 自动启动，不用单独跑。关掉窗口只是不看了，部署照常。
"""
import os
import signal
import sys
import time
from multiprocessing import resource_tracker, shared_memory

import cv2
import numpy as np

H, W = 480, 640
# 共享内存布局，和 deploy_ar5.ViewerFeed 一致
N_HDR = 8   # seq, alive, executing, step, mask_area, grip_now, grip_cmd, steps_per_s
HDR_BYTES = N_HDR * 8
IMG = H * W * 3


def views(buf):
    hdr = np.ndarray((N_HDR,), dtype=np.float64, buffer=buf, offset=0)
    head = np.ndarray((H, W, 3), dtype=np.uint8, buffer=buf, offset=HDR_BYTES)
    wrist = np.ndarray((H, W, 3), dtype=np.uint8, buffer=buf, offset=HDR_BYTES + IMG)
    mask = np.ndarray((H, W), dtype=np.uint8, buffer=buf, offset=HDR_BYTES + 2 * IMG)
    return hdr, head, wrist, mask


def compose(head, wrist, mask, hdr):
    h = head.copy()
    m = mask > 0
    h[m] = (0.5 * h[m] + [127, 0, 0]).astype(np.uint8)          # 掩码叠蓝色
    canvas = np.hstack([h, wrist])
    for x, name in ((0, "head + mask (blue)"), (W, "wrist")):
        cv2.rectangle(canvas, (x, 0), (x + W, 22), (0, 0, 0), -1)
        cv2.putText(canvas, name, (x + 8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    executing, step, area, gnow, gcmd, rate = hdr[2], int(hdr[3]), int(hdr[4]), hdr[5], hdr[6], hdr[7]
    bar = np.zeros((34, canvas.shape[1], 3), np.uint8)
    colour = (60, 60, 255) if executing else (180, 180, 180)
    cv2.circle(bar, (18, 17), 8, colour, -1)
    # 全是 ASCII：cv2.putText 没有中文字形
    text = (f"{'RUNNING' if executing else 'IDLE'}  step {step}  {rate:.1f} step/s   "
            f"mask {area} px{'  (EMPTY)' if area == 0 else ''}   gripper {gnow:.2f} -> cmd {gcmd:.2f}")
    cv2.putText(bar, text, (36, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return np.vstack([canvas, bar])


def main(name, snapshot=None):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C 交给部署程序处理，它退出时会让这里收工
    shm = shared_memory.SharedMemory(name=name)
    # 只是借用：别让这个进程退出时把共享内存删掉（Python 3.11 的 resource_tracker 会这么干）
    resource_tracker.unregister(shm._name, "shared_memory")
    hdr, head, wrist, mask = views(shm.buf)
    win = "DexGraspVLA live (head + mask | wrist)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 514)
    last_seq, last_snap, n_snap = -1.0, 0.0, 0
    try:
        while hdr[1] > 0:                                     # alive
            seq = hdr[0]
            if seq != last_seq and int(seq) % 2 == 0:           # 偶数 = 写完了
                frame = compose(head, wrist, mask, hdr)
                if hdr[0] == seq:                               # 拷的过程中没被改写
                    last_seq = seq
                    cv2.imshow(win, frame)
                    if snapshot and time.monotonic() - last_snap > 1.0:
                        if os.path.isdir(snapshot):  # 给的是目录：每秒一张带编号，事后能拼时间线
                            cv2.imwrite(os.path.join(snapshot, f"view_{n_snap:04d}.jpg"), frame)
                            n_snap += 1
                        else:                       # 给的是文件：只留最新一张
                            cv2.imwrite(snapshot, frame)
                        last_snap = time.monotonic()
            cv2.waitKey(30)
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break                                           # 窗口被关了：不看了，部署照常
    finally:
        cv2.destroyAllWindows()
        shm.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
