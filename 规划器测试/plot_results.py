"""把规划器测试的结果画成图：每个方案一张，上面是实际发给模型的提示词，下面四栏是四个模型实际框出来的结果。

只读 结果/ 里已有的结果文件，不调接口；作废的（在子目录里）、探针的、qwen3.7-flash 的不画。
输出到 结果/可视化/（图里有实验室画面，不进 git）。有新结果时重跑一次就更新。

    PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python 规划器测试/plot_results.py

中文字体按名字用 fc-match 找（Noto Sans CJK SC），提示词原文用 DejaVu Sans Mono。
"""
import glob
import json
import os
import re
import subprocess
import textwrap

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "结果")
CASE = os.path.join(HERE, "测试集", "案例01")
OUT = os.path.join(RES, "可视化")

MODELS = ["qwen3-vl-plus", "qwen3.6-35b-a3b", "qwen3.5-122b-a10b", "qwen3.5-397b-a17b"]
MODEL_NOTE = {"qwen3-vl-plus": "闭源（采用的方案）", "qwen3.6-35b-a3b": "开源 · 总 35B / 激活 3B",
              "qwen3.5-122b-a10b": "开源 · 总 122B / 激活 10B", "qwen3.5-397b-a17b": "开源 · 总 397B / 激活 17B"}

# 框的颜色用参考调色板前三个色位（任意两两都过了色盲校验）；青绿对比度偏低，所以处处配文字说明
HIT, MISS, TRUTH = (0x2a, 0x78, 0xd6), (0xeb, 0x68, 0x34), (0x1b, 0xaf, 0x7a)
GRAY = (120, 120, 120)
INK, INK2, INK3 = (11, 11, 11), (82, 81, 78), (120, 119, 114)
BG, PANEL, BAND = (252, 252, 251), (243, 242, 239), (226, 236, 250)

CROP = (230, 150, 630, 440)          # 场景图里物体所在的区域，放大 1.25 倍
COL_W, GAP, MARGIN = 500, 24, 32
SCALE = COL_W / (CROP[2] - CROP[0])
CELL_H = round((CROP[3] - CROP[1]) * SCALE)
W = MARGIN * 2 + COL_W * 4 + GAP * 3


def _font(pattern, size):
    try:
        out = subprocess.run(["fc-match", "-f", "%{file}|%{index}", pattern],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        raise SystemExit(f"找不到字体 {pattern}：要装 fontconfig 和 Noto Sans CJK / DejaVu 字体")
    path, _, idx = out.partition("|")
    return ImageFont.truetype(path, size, index=int(idx or 0))


F_TITLE = _font("Noto Sans CJK SC:style=Bold", 34)
F_H = _font("Noto Sans CJK SC:style=Bold", 24)
F_B = _font("Noto Sans CJK SC:style=Regular", 20)
F_S = _font("Noto Sans CJK SC:style=Regular", 17)
F_MONO = _font("DejaVu Sans Mono", 15)


# ---------- 读数据 ----------

def load_runs():
    runs = []
    for f in sorted(glob.glob(os.path.join(RES, "*", "结果.json"))):  # 只读一层：作废的在子目录里
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        if not isinstance(r, dict) or r.get("模型") not in MODELS:
            continue                                                # 探针结果是列表；qwen3.7-flash 不画
        d = os.path.basename(os.path.dirname(f))
        r["_dir"] = os.path.dirname(f)
        r["_方案"] = "两步法" if d.startswith("两步_") else "一次请求"
        r["输入"] = r.get("输入") or "图片"                           # 最早几次的结果文件没记这两项
        r["_顺序"] = r.get("图片顺序") or "参考在前"
        r["_框"] = r.get("最终框（像素）") if r["_方案"] == "两步法" else (r.get("预测框（像素）") or r.get("预测框"))
        r["_耗时"] = r.get("总耗时秒") if r["_方案"] == "两步法" else r.get("耗时秒")
        runs.append(r)
    return runs


def load_case():
    with open(os.path.join(CASE, "case.json"), encoding="utf-8") as f:
        return json.load(f)


def load_request(run_dir, name):
    with open(os.path.join(run_dir, name), encoding="utf-8") as f:
        return json.load(f)


# ---------- 画基本块 ----------

def text_block(lines, width, font, fill=INK, bg=None, line_gap=6, pad=0):
    """lines: [(文字, 字体或 None, 颜色或 None)]；按宽度自动换行，返回一张图。"""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    limit = width - 2 * pad
    rows = []
    for item in lines:
        txt, f, c = (item if isinstance(item, tuple) else (item, None, None))
        f, c = f or font, c or fill
        for para in txt.split("\n"):
            cur = ""
            # 英文按单词断行，中文按字断行；单个词比一行还长时才按字母断
            for tok in re.findall(r"[A-Za-z0-9_\-.,:;'\"()/+%~=<>\[\]{}*]+|\s+|.", para):
                if probe.textlength(cur + tok, font=f) <= limit:
                    cur += tok
                    continue
                if cur.strip():
                    rows.append((cur.rstrip(), f, c))
                cur = "" if tok.isspace() else tok
                while probe.textlength(cur, font=f) > limit:
                    cut = len(cur)
                    while cut > 1 and probe.textlength(cur[:cut], font=f) > limit:
                        cut -= 1
                    rows.append((cur[:cut], f, c))
                    cur = cur[cut:]
            rows.append((cur.rstrip(), f, c))
    heights = [f.getbbox("国Ag")[3] + line_gap for _, f, _ in rows]
    im = Image.new("RGB", (width, sum(heights) + 2 * pad), bg or BG)
    d = ImageDraw.Draw(im)
    y = pad
    for (txt, f, c), h in zip(rows, heights):
        d.text((pad, y), txt, font=f, fill=c)
        y += h
    return im


def prompt_panel(messages, width, image_files, note=None, highlight=None, highlight_note=None):
    """把实际发出去的请求画成一个面板：文字照原文（等宽字体），图片画成缩略图，按发送顺序排。"""
    pad = 16
    inner = width - 2 * pad
    blocks = []
    if note:
        blocks.append(text_block([note], inner, F_S, fill=INK2, bg=PANEL))
    sys_text = messages[0]["content"][0]["text"]
    blocks.append(text_block([f"[system] {sys_text}"], inner, F_MONO, fill=INK3, bg=PANEL))
    cw = F_MONO.getlength("M")
    for part in messages[1]["content"]:
        if part["type"] == "image_url":
            m = re.search(r"\[图片 ([^\s\]]+)", part["image_url"]["url"])
            name = m.group(1) if m else "?"
            thumb = Image.open(image_files[name]).convert("RGB")
            thumb = thumb.resize((round(thumb.width * 120 / thumb.height), 120))
            cap = text_block([f"[图片] {name}"], inner - thumb.width - 12, F_S, fill=INK2, bg=PANEL)
            row = Image.new("RGB", (inner, max(thumb.height, cap.height)), PANEL)
            row.paste(thumb, (0, 0))
            row.paste(cap, (thumb.width + 12, (thumb.height - cap.height) // 2))
            blocks.append(row)
            continue
        txt = part["text"]
        wrapped = "\n".join("\n".join(textwrap.wrap(line, int(inner / cw)) or [""]) for line in txt.split("\n"))
        is_hl = highlight and highlight in txt
        b = text_block([wrapped], inner, F_MONO, fill=INK, bg=BAND if is_hl else PANEL, line_gap=4)
        blocks.append(b)
        if is_hl and highlight_note:
            blocks.append(text_block([highlight_note], inner, F_S, fill=INK2, bg=PANEL))
    h = sum(b.height for b in blocks) + pad * (len(blocks) + 1)
    panel = Image.new("RGB", (width, h), PANEL)
    y = pad
    for b in blocks:
        panel.paste(b, (pad, y))
        y += b.height + pad
    return panel


def T(b):
    return [(b[0] - CROP[0]) * SCALE, (b[1] - CROP[1]) * SCALE, (b[2] - CROP[0]) * SCALE, (b[3] - CROP[1]) * SCALE]


def scene_cell(case, boxes, base_image=None):
    """场景局部放大；boxes: [(框, 是否命中)]。先画没命中的，命中的压在上面。
    给了 base_image（模型实际看到的候选编号图）时原样放大，不再叠真值和干扰物，免得和候选框的颜色混在一起。"""
    src = Image.open(base_image or os.path.join(CASE, "scene.png")).convert("RGB")
    im = src.crop(CROP).resize((COL_W, CELL_H), Image.LANCZOS)
    if base_image:
        return im
    d = ImageDraw.Draw(im)
    dis = T(case["干扰物框"])
    d.rectangle(dis, outline=GRAY, width=1)
    d.text((dis[0] + 3, dis[1] - 22), "干扰物", font=F_S, fill=GRAY)
    d.rectangle(T(case["真值框"]), outline=TRUTH, width=3)
    for b, hit in sorted(boxes, key=lambda x: x[1]):
        d.rectangle(T(b), outline=HIT if hit else MISS, width=2)
    return im


def empty_cell(msg):
    im = Image.new("RGB", (COL_W, CELL_H), PANEL)
    d = ImageDraw.Draw(im)
    tw = d.textlength(msg, font=F_B)
    d.text(((COL_W - tw) / 2, CELL_H / 2 - 14), msg, font=F_B, fill=INK3)
    return im


def legend(width):
    im = Image.new("RGB", (width, 34), BG)
    d = ImageDraw.Draw(im)
    x = 0
    for color, label, w in [(TRUTH, "真值（目测）", 3), (HIT, "命中的框", 2), (MISS, "没命中的框", 2),
                            (GRAY, "干扰物：白色六角件套在黑圆柱上", 1)]:
        d.rectangle([x, 6, x + 30, 26], outline=color, width=w)
        d.text((x + 40, 4), label, font=F_B, fill=INK)
        x += 40 + d.textlength(label, font=F_B) + 36
    return im


def span(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    lo, hi = min(vals), max(vals)
    return f"{lo:.1f}" if abs(hi - lo) < 0.05 else f"{lo:.1f}~{hi:.1f}"


class Canvas:
    """从上往下排版：add(图, x) 放在当前行，next(高) 换行。"""

    def __init__(self):
        self.items, self.y = [], MARGIN

    def put(self, im, x=MARGIN, y=None):
        self.items.append((im, x, self.y if y is None else y))

    def row(self, im, x=MARGIN, gap=12):
        self.put(im, x)
        self.y += im.height + gap

    def save(self, path):
        h = self.y + MARGIN
        out = Image.new("RGB", (W, h), BG)
        for im, x, y in self.items:
            out.paste(im, (x, y))
        out.save(path)
        return out.size


def col_x(i):
    return MARGIN + i * (COL_W + GAP)


def column_headers(c):
    top = c.y
    hs = []
    for i, m in enumerate(MODELS):
        b = text_block([(m, F_H, INK), (MODEL_NOTE[m], F_S, INK2)], COL_W, F_B)
        c.put(b, col_x(i), top)
        hs.append(b.height)
    c.y = top + max(hs) + 8


def cell_row(c, cells, captions):
    top = c.y
    hs = []
    for i, (cell, cap) in enumerate(zip(cells, captions)):
        c.put(cell, col_x(i), top)
        capim = text_block(cap, COL_W, F_B, line_gap=4)
        c.put(capim, col_x(i), top + cell.height + 8)
        hs.append(cell.height + 8 + capim.height)
    c.y = top + max(hs) + 24


def header(c, title, subtitle):
    c.row(text_block([(title, F_TITLE, INK)], W - 2 * MARGIN, F_B), gap=4)
    c.row(text_block([subtitle], W - 2 * MARGIN, F_B, fill=INK2), gap=8)
    c.row(legend(W - 2 * MARGIN), gap=16)


SUBTITLE = ("案例01：参考图是腕部相机截出的灰色六角柱（没有通用名字），场景图是 335 俯视、另一次摆放。"
            "每格是场景图的局部放大，叠画这个模型所有次运行给出的框。命中 = 和真值的交并比 ≥ 0.5。")


# ---------- 方案一：一次请求 ----------

def plot_one_shot(runs, case):
    one = [r for r in runs if r["_方案"] == "一次请求"]
    sample = next(r for r in one if r["模型"] == "qwen3-vl-plus" and r["输入"] == "图片加外形")
    msgs = load_request(sample["_dir"], "请求.json")
    files = {"ref.png": os.path.join(CASE, "ref.png"), "scene.png": os.path.join(CASE, "scene.png")}

    c = Canvas()
    header(c, "方案一：一次请求，同时发参考图和场景图，直接要框", SUBTITLE)
    c.row(text_block([("实际发给模型的内容（英文原文，按发送顺序）", F_H, INK)], W - 2 * MARGIN, F_B), gap=8)
    c.row(prompt_panel(
        msgs, W - 2 * MARGIN, files,
        note="这是「参考在前」的写法。「场景在前」只是两张图对调，文字里的 Image 1 / Image 2 跟着对调，要求在 Image 1（场景图）上给坐标。"
             "Qwen3 系列按 0~999 相对坐标给框，画图前换成像素。",
        highlight="also describes the object's appearance",
        highlight_note="↑ 浅蓝底这一句只在「图片加外形说明」时有；只给图片时没有，其余完全一样。"), gap=24)
    column_headers(c)

    for order in ["参考在前", "场景在前"]:
        c.row(text_block([(f"{order}（{'参考图第 1 张、场景图第 2 张' if order == '参考在前' else '场景图第 1 张、参考图第 2 张'}）",
                           F_H, INK)], W - 2 * MARGIN, F_B), gap=8)
        cells, caps = [], []
        for m in MODELS:
            rs = [r for r in one if r["模型"] == m and r["_顺序"] == order]
            if not rs:
                cells.append(empty_cell("没测：参考在前已经全中"))
                caps.append([""])
                continue
            hits = sum(r["结论"] == "命中" for r in rs)
            boxes = [(r["_框"], r["结论"] == "命中") for r in rs if r["_框"]]
            cells.append(scene_cell(case, boxes))
            n_img = sum(r["输入"] == "图片" for r in rs)
            n_shape = len(rs) - n_img
            how = f"只给图片 {n_img} 次" + (f"、加外形说明 {n_shape} 次" if n_shape else "")
            labels = sorted({(r.get("解析结果") or {}).get("label") for r in rs
                             if isinstance(r.get("解析结果"), dict) and r["解析结果"].get("label")})
            cap = [(f"命中 {hits}/{len(rs)}", F_H, INK), how,
                   f"每次 {span([r['_耗时'] for r in rs])} 秒"]
            if labels:
                cap.append((f"模型对目标的描述：{' / '.join(labels[:3])}", F_S, INK2))
            caps.append(cap)
        cell_row(c, cells, caps)

    c.row(text_block([
        ("说明", F_H, INK),
        "· 开源模型这 18 次的描述都认对了目标，框也没落到别的物体上，只是落在目标旁边的空白处。"
        "原因是 Qwen3.5/3.6 一次请求里有两张图时坐标会整体偏（只发场景图时它们框得准，见坐标探针）。",
        "· qwen3-vl-plus 第 1 次和 qwen3.6-35b-a3b「参考在前」3 次用的是早期提示词：reason 没限定只描述外形。"
        "框在 reason 前面输出，不影响框。",
        "· 只有一个案例；真值框是目测的，误差约 3 像素。"], W - 2 * MARGIN, F_S, fill=INK2), gap=0)
    path = os.path.join(OUT, "方案一_一次请求.png")
    return path, c.save(path)


# ---------- 方案二：两步法 ----------

def plot_two_step(runs, case):
    two = [r for r in runs if r["_方案"] == "两步法"]
    sample = next(r for r in two if r["模型"] == "qwen3-vl-plus" and r["输入"] == "图片加外形")
    m1 = load_request(sample["_dir"], "第一步请求.json")
    m2 = load_request(sample["_dir"], "第二步请求.json")
    files = {"ref.png": os.path.join(CASE, "ref.png"), "scene.png": os.path.join(CASE, "scene.png"),
             "候选编号图.png": os.path.join(sample["_dir"], "候选编号图.png")}

    c = Canvas()
    header(c, "方案二：两步法 —— 先只发场景图框出所有物体，再把候选编号画上去，和参考图一起发，只问「是几号」",
           SUBTITLE.replace("叠画这个模型所有次运行给出的框", "第一行是第一步实际产出的候选编号图，第二行叠画 6 次最终选中的框"))
    half = (W - 2 * MARGIN - GAP) // 2
    top = c.y
    t1 = text_block([("第一步：只发场景图（坐标从单图拿）", F_H, INK)], half, F_B)
    t2 = text_block([("第二步：参考图 + 候选编号图，不要坐标", F_H, INK)], half, F_B)
    c.put(t1, MARGIN, top)
    c.put(t2, MARGIN + half + GAP, top)
    c.y = top + max(t1.height, t2.height) + 8
    top = c.y
    p1 = prompt_panel(m1, half, files, note="四个模型、两种输入的第一步完全一样。")
    p2 = prompt_panel(m2, half, files,
                      note="例图是 qwen3-vl-plus 某一次第一步的产出；每次运行发的是它自己第一步画出来的编号图。",
                      highlight="also describes the object's appearance",
                      highlight_note="↑ 浅蓝底这一句只在「图片加外形说明」时有。")
    c.put(p1, MARGIN, top)
    c.put(p2, MARGIN + half + GAP, top)
    c.y = top + max(p1.height, p2.height) + 24
    column_headers(c)

    # 第一行：各模型第一次（只给图片）运行实际产出的候选编号图
    c.row(text_block([("第一步实际产出的候选编号图（各取只给图片的第 1 次）", F_H, INK),
                      ("这就是第二步发给模型的图，原样放大；框的颜色只用来区分候选，和上面的图例无关。目标是几号写在图下面。",
                       F_S, INK2)], W - 2 * MARGIN, F_B), gap=8)
    cells, caps = [], []
    for m in MODELS:
        rs = [r for r in two if r["模型"] == m]
        first = next(r for r in rs if r["输入"] == "图片")
        cand_img = os.path.join(first["_dir"], "候选编号图.png")
        cells.append(scene_cell(case, [], base_image=cand_img))
        near = first.get("候选里最接近真值的") or {}
        target = f"目标是 {near['编号']} 号" if near.get("交并比", 0) >= 0.5 else "目标没被框进来"
        chosen = first.get("选中编号")
        caps.append([f"{len(first.get('候选（像素）') or {})} 个候选，{target}",
                     f"第二步选了 {chosen} 号 → {first['结论']}" if chosen else f"第二步 → {first['结论']}"])
    cell_row(c, cells, caps)

    # 第二行：6 次的最终框
    c.row(text_block([("6 次的最终框（只给图片 3 次 + 加外形说明 3 次）", F_H, INK)], W - 2 * MARGIN, F_B), gap=8)
    cells, caps = [], []
    for m in MODELS:
        rs = [r for r in two if r["模型"] == m]
        hits = sum(r["结论"] == "命中" for r in rs)
        miss1 = sum(r["结论"].startswith("未命中：第一步") for r in rs)
        dis = sum(r["结论"] == "选中干扰物" for r in rs)
        other = len(rs) - hits - miss1 - dis
        boxes = [(r["_框"], r["结论"] == "命中") for r in rs if r["_框"]]
        cells.append(scene_cell(case, boxes))
        parts = [f"选到干扰物 {dis}"] if dis else []
        parts += [f"第一步漏框 {miss1}"] if miss1 else []
        parts += [f"其他 {other}"] if other else []
        caps.append([(f"命中 {hits}/{len(rs)}", F_H, INK),
                     "，".join(parts) if parts else "全部命中",
                     f"每轮 {span([r['_耗时'] for r in rs])} 秒（两步合计）"])
    cell_row(c, cells, caps)

    c.row(text_block([
        ("说明", F_H, INK),
        "· 「选到干扰物」是第二步认错：qwen3.5-397b-a17b 6 次里 5 次认定白色六角件是「深色六角柱、白色圆孔」，这是真正的看错，不是坐标问题。",
        "· 「第一步漏框」是第一步没把只有 29×45 像素的六角柱列进候选，第二步再怎么选也选不中。"
        "qwen3.5-122b-a10b 最后一次第一步漏框后选到了干扰物，算在「选到干扰物」里。",
        "· 开源模型第一次跑两步法时因为脚本编号 bug 作废，图里是修好后重跑的结果；qwen3-vl-plus 那次不受影响。"],
        W - 2 * MARGIN, F_S, fill=INK2), gap=0)
    path = os.path.join(OUT, "方案二_两步法.png")
    return path, c.save(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    runs, case = load_runs(), load_case()
    for path, size in [plot_one_shot(runs, case), plot_two_step(runs, case)]:
        print(f"{path}  {size[0]}x{size[1]}")


if __name__ == "__main__":
    main()
