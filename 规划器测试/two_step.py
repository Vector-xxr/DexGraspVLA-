"""两步法：坐标只从单图请求里拿，绕开 Qwen3.5/3.6 一次发两张图坐标就乱的问题（10-17 坐标探针查出来的）。

    第一步  只发场景图，让模型框出台面上所有物体（单图坐标已验证是准的）
    第二步  把这些候选框编号画到场景图上，和参考图一起发，只问「参考图里的是几号」，不要坐标
    最后    用第一步里那个编号的框当结果，和真值比

用法（在 算法/DexGraspVLA- 目录下）：
    export DASHSCOPE_API_KEY=sk-xxxx
    PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python 规划器测试/two_step.py 规划器测试/测试集/案例01 --model qwen3.5-122b-a10b
    # --input 图片加外形：第二步多给一句外形说明（case.json 的「外形说明」）
    # --dry-run：不调接口，按 物体框.json 画一张「第二步会发的候选编号图」看看样子

结果写到 规划器测试/结果/两步_案例名_输入_模型_时间/：两步各自的请求、发出去的候选编号图、可视化、结果.json。
失败时分得清是哪一步错的：第一步没框到目标（候选里没有），还是第二步选错了号。
"""
import argparse
import json
import os
import re
import time
import urllib.error

from PIL import Image, ImageDraw, ImageFont

import ref_grounding_min as rg

STEP1_RULES = {
    "pixel": "in the image's pixel coordinates",
    "relative": "in coordinates relative to the image, normalized to the range 0-999 for both width and height",
}
STEP1_TASK = (
    "Detect every distinct physical object on the table in the image (ignore the table surface, the floor and "
    "the background). Return a bounding box for each object, COORD_RULE.\n"
    "Return strict JSON only, as a list: "
    "[{\"id\": 1, \"bbox_2d\": [x1, y1, x2, y2], \"label\": \"short name\"}, ...] with ids numbered 1, 2, 3, ..."
)
STEP2_SCENE = "Image 2 (current view from the robot's head camera; candidate objects are outlined and numbered):"
STEP2_TASK = (
    "Task: decide which numbered candidate in Image 2 is the object shown in Image 1.\n"
    "Rules:\n"
    "1. Answer with exactly one candidate number.\n"
    "2. Similar-looking objects may be present; choose the one that best matches the reference.\n"
    "3. If none of the candidates is the object, answer null. Do not guess.\n"
    "Return strict JSON only:\n"
    "{\"id\": <candidate number or null>, \"reason\": \"one sentence on which visual features matched; "
    "describe appearance only, not location\"}"
)
PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
           (240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128), (170, 110, 40), (128, 0, 0)]


def parse_candidates(text):
    """第一步的回复 → ({"1": 框, "2": 框, ...}, {"1": 名字, ...})，按列表顺序编号。
    不拿模型给的 id / label 当编号：Qwen3.5/3.6 常不给 id，label 又会重名（6 个都叫 physical object）。
    10-17 第一次跑就因此把 6 个候选并成 1 个，编号图上画的也是名字不是数字，第二步全部作废。"""
    s = re.sub(r"```(?:json)?", "", text).strip()
    items = []
    m = re.search(r"\[.*\]", s, re.S)
    if m:
        try:
            items = [it for it in json.loads(m.group(0)) if isinstance(it, dict) and it.get("bbox_2d")]
        except (json.JSONDecodeError, TypeError):
            items = []
    if items:
        boxes = [it["bbox_2d"] for it in items]
        names = [str(it.get("label") or it.get("id") or "") for it in items]
    else:  # JSON 坏了：按出现顺序抓所有四元组
        boxes = re.findall(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", s)
        names = [""] * len(boxes)
    cands = {str(i + 1): list(map(float, b)) for i, b in enumerate(boxes)}
    return cands, {str(i + 1): n for i, n in enumerate(names)}


def step1_messages(scene_path, coords):
    return [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": rg.data_url(scene_path)}},
                {"type": "text", "text": STEP1_TASK.replace("COORD_RULE", STEP1_RULES[coords])}]}]


def step2_messages(ref_path, marked_path, mode, shape):
    content = [{"type": "text", "text": rg.PROMPT_INTRO + "\nImage 1 (reference photo of the target object):"},
               {"type": "image_url", "image_url": {"url": rg.data_url(ref_path)}}]
    if mode == "图片加外形":
        content.append({"type": "text", "text": rg.PROMPT_SHAPE.replace("SHAPE", shape)})
    content += [{"type": "text", "text": STEP2_SCENE},
                {"type": "image_url", "image_url": {"url": rg.data_url(marked_path)}},
                {"type": "text", "text": STEP2_TASK}]
    return [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": content}]


def draw_candidates(scene_path, cands, out_path):
    """候选框 + 编号牌。编号牌放在框的左上角外侧，尽量不挡住物体本身。"""
    im = Image.open(scene_path).convert("RGB")
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=16)
    for i, (cid, b) in enumerate(cands.items()):
        color = PALETTE[i % len(PALETTE)]
        d.rectangle(b, outline=color, width=2)
        tw, th = d.textbbox((0, 0), str(cid), font=font)[2:]
        x = min(max(0, b[0]), im.width - tw - 6)
        y = b[1] - th - 6 if b[1] - th - 6 >= 0 else b[3] + 2
        d.rectangle([x, y, x + tw + 6, y + th + 4], fill=color)
        d.text((x + 3, y + 1), str(cid), fill=(255, 255, 255), font=font)
    im.save(out_path)


def parse_choice(text):
    s = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            cid = obj.get("id")
            return (None if cid in (None, "null", "") else str(cid).strip()), obj
        except json.JSONDecodeError:
            pass
    m = re.search(r"\"id\"\s*:\s*\"?(\d+|null)", s)
    if m:
        return (None if m.group(1) == "null" else m.group(1)), {"id": "正则兜底解析"}
    raise ValueError("回复里找不到 id")


def visualize(ref_path, scene_path, cands, chosen, case, lines, out_path):
    scene = Image.open(scene_path).convert("RGB")
    ref = Image.open(ref_path).convert("RGB")
    ref = ref.resize((round(ref.width * scene.height / ref.height), scene.height))
    canvas = Image.new("RGB", (ref.width + scene.width, scene.height + 18 * len(lines) + 8), "white")
    canvas.paste(ref, (0, 0))
    canvas.paste(scene, (ref.width, 0))
    d = ImageDraw.Draw(canvas)

    def box(b, color, width):
        d.rectangle([b[0] + ref.width, b[1], b[2] + ref.width, b[3]], outline=color, width=width)

    for cid, b in cands.items():
        box(b, (150, 150, 150), 1)
        d.text((b[0] + ref.width + 2, b[1] + 1), cid, fill=(150, 150, 150))
    if case.get("真值框"):
        box(case["真值框"], (0, 200, 0), 2)
    if chosen in cands:
        box(cands[chosen], (255, 0, 0), 3)
    for i, line in enumerate(lines):
        d.text((6, scene.height + 4 + 18 * i), line, fill="black")
    canvas.save(out_path)


def ask(args, api_key, messages):
    try:
        data, latency = rg.call_api(args.base_url, api_key, args.model, messages, args.timeout, args.thinking)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"接口返回 {e.code}：{e.read().decode(errors='replace')}")
    return (data["choices"][0]["message"].get("content") or ""), latency, data.get("usage")


def main():
    ap = argparse.ArgumentParser(description="两步法：先单图框全部物体，再按参考图选编号")
    ap.add_argument("case_dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", choices=["图片", "图片加外形"], default="图片")
    ap.add_argument("--base-url", default=rg.DEFAULT_BASE_URL)
    ap.add_argument("--coords", choices=["auto", "pixel", "relative"], default="auto")
    ap.add_argument("--thinking", choices=["off", "on", "unset"], default="off")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--dry-run", action="store_true", help="不调接口，按 物体框.json 画一张候选编号图")
    args = ap.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("没有设置 DASHSCOPE_API_KEY。先 export DASHSCOPE_API_KEY=sk-xxxx，或者加 --dry-run")
    coords = args.coords if args.coords != "auto" else ("relative" if "qwen3" in args.model.lower() else "pixel")

    case_dir = os.path.abspath(args.case_dir)
    ref_path, scene_path = os.path.join(case_dir, "ref.png"), os.path.join(case_dir, "scene.png")
    with open(os.path.join(case_dir, "case.json"), encoding="utf-8") as f:
        case = json.load(f)
    shape = case.get("外形说明")
    if args.input == "图片加外形" and not shape:
        raise SystemExit("--input 图片加外形 要用 case.json 里的「外形说明」，这个案例没写")
    sw, sh = Image.open(scene_path).size

    base = os.path.join(rg.HERE, "结果", f"两步_{os.path.basename(case_dir)}_{args.input}_{args.model}_"
                                         f"{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir, n = base, 1
    while os.path.exists(out_dir):
        n += 1
        out_dir = f"{base}_{n}"
    os.makedirs(out_dir)
    marked_path = os.path.join(out_dir, "候选编号图.png")

    def save_request(name, messages, images):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(rg.preview_messages(messages, images), f, ensure_ascii=False, indent=2)

    m1 = step1_messages(scene_path, coords)
    save_request("第一步请求.json", m1, [f"scene.png {sw}x{sh}"])
    result = {"案例": os.path.basename(case_dir), "模型": args.model, "输入": args.input,
              "外形说明": shape if args.input == "图片加外形" else None, "坐标约定": coords, "思考模式": args.thinking}

    if args.dry_run:
        with open(os.path.join(case_dir, "物体框.json"), encoding="utf-8") as f:
            demo = {str(i + 1): o["框"] for i, o in enumerate(json.load(f)["物体"].values())}
        draw_candidates(scene_path, demo, marked_path)
        save_request("第二步请求.json", step2_messages(ref_path, marked_path, args.input, shape),
                     ["ref.png", "候选编号图.png"])
        print(f"模拟运行完成，第二步会发的候选编号图在 {marked_path}")
        return

    # 第一步：单图框全部物体
    text1, lat1, use1 = ask(args, api_key, m1)
    raw, names = parse_candidates(text1)
    cands = {cid: rg.to_pixel(b, coords, sw, sh) for cid, b in raw.items()}
    result.update({"第一步耗时秒": round(lat1, 2), "第一步用量": use1, "第一步原始回复": text1,
                   "候选（像素）": cands, "候选名称": names})
    gt = case.get("真值框")
    best = max(((rg.iou(b, gt), cid) for cid, b in cands.items()), default=(0.0, None))
    result["候选里最接近真值的"] = {"编号": best[1], "交并比": round(best[0], 3)}

    chosen, verdict, lat2 = None, None, 0.0
    if not cands:
        verdict = "第一步解析失败"
    else:
        draw_candidates(scene_path, cands, marked_path)
        m2 = step2_messages(ref_path, marked_path, args.input, shape)
        save_request("第二步请求.json", m2, ["ref.png", "候选编号图.png"])
        text2, lat2, use2 = ask(args, api_key, m2)
        result.update({"第二步耗时秒": round(lat2, 2), "第二步用量": use2, "第二步原始回复": text2})
        try:
            chosen, parsed = parse_choice(text2)
            result["第二步解析结果"] = parsed
        except ValueError as e:
            verdict = f"第二步解析失败：{e}"
        if verdict is None:
            if chosen is None:
                verdict = "模型回答候选里没有"
            elif chosen not in cands:
                verdict = f"选了不存在的编号 {chosen}"
            else:
                iou_gt = rg.iou(cands[chosen], gt) if gt else None
                iou_dis = rg.iou(cands[chosen], case["干扰物框"]) if case.get("干扰物框") else None
                result.update({"选中编号": chosen, "最终框（像素）": cands[chosen],
                               "与真值交并比": iou_gt, "与干扰物交并比": iou_dis})
                if iou_gt is not None and iou_gt >= 0.5:
                    verdict = "命中"
                elif iou_dis is not None and iou_dis >= 0.5:
                    verdict = "选中干扰物"
                elif best[0] < 0.5:
                    verdict = "未命中：第一步就没框到目标"
                else:
                    verdict = f"未命中：第二步选错（目标是 {best[1]} 号）"
    result["结论"] = verdict
    result["总耗时秒"] = round(lat1 + lat2, 2)
    with open(os.path.join(out_dir, "结果.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    iou_s = "-" if result.get("与真值交并比") is None else f"{result['与真值交并比']:.2f}"
    lines = [f"two-step  model = {args.model}  input = {rg.INPUT_TAGS[args.input]}  "
             f"candidates = {len(cands)}  chosen = {chosen}  IoU gt = {iou_s}",
             f"gray = step-1 candidates, red = chosen, green = ground truth (eyeballed)   "
             f"latency {lat1:.1f}s + {lat2:.1f}s"]
    visualize(ref_path, scene_path, cands, chosen, case, lines, os.path.join(out_dir, "可视化.png"))

    print(f"结论：{verdict}（{args.model}，两步法，输入：{args.input}）")
    near = (f"最接近真值的是 {best[1]} 号（交并比 {best[0]:.2f}）" if best[0] >= 0.1
            else f"没有一个候选框到目标（最大交并比 {best[0]:.2f}）")
    print(f"第一步：{len(cands)} 个候选，{near}，{lat1:.1f} 秒")
    pick = f"选了 {chosen} 号" if chosen else "没选（回答候选里没有，或解析失败）"
    print(f"第二步：{pick}，最终交并比 {iou_s}，{lat2:.1f} 秒；合计 {lat1 + lat2:.1f} 秒")
    print(f"结果在 {out_dir}")


if __name__ == "__main__":
    main()
