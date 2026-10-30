"""坐标探针：让模型一次框出场景里多个已知位置的物体，按横纵两个方向分别拟合「模型坐标 = 斜率 × 像素 + 截距」，
查清它的坐标约定到底是什么，以及多发一张参考图会不会把坐标带偏。

用法（在 算法/DexGraspVLA- 目录下）：
    export DASHSCOPE_API_KEY=sk-xxxx
    PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python 规划器测试/coord_probe.py 规划器测试/测试集/案例01 \
        --models qwen3-vl-plus qwen3.7-flash qwen3.6-35b-a3b

每个模型默认跑下面四种问法（--variants 可以只选几种）：
    单图-相对    只发场景图，要求 0~999 相对坐标（和主测试的写法一样）
    单图-不指定  只发场景图，不提坐标约定，看模型自己默认给什么
    双图-相对    参考图 + 场景图（和最早的主测试布局一样），要求 0~999，看参考图会不会把坐标带偏
    双图-场景在前 场景图 + 参考图，要求 0~999，看把要框的图排第一能不能躲开这个偏差
10-17 结论：Qwen3.5/3.6 单图准、参考图在前就整体偏；qwen3-vl-plus 两种都准。
案例目录里要有 scene.png、ref.png、物体框.json；结果写到 规划器测试/结果/探针_时间/。

怎么看：相对坐标正确时，横向斜率 ≈ 999/图宽，纵向斜率 ≈ 999/图高，截距 ≈ 0；
像素坐标时斜率 ≈ 1。「等效宽/高」= 999 / 斜率，就是模型心里以为的图片尺寸。
"""
import argparse
import json
import os
import re
import time
import urllib.error

from PIL import Image, ImageDraw

import ref_grounding_min as rg

VARIANTS = ["单图-相对", "单图-不指定", "双图-相对", "双图-场景在前"]


def build_probe_messages(variant, objects, scene_path, ref_path):
    where = {"双图-相对": "Image 2", "双图-场景在前": "Image 1"}.get(variant, "the image")
    rule = "" if variant.endswith("不指定") else (
        f", in coordinates relative to {where}, normalized to the range 0-999 for both width and height")
    listing = "\n".join(f"- {oid}: {o['描述']}" for oid, o in objects.items())
    task = (f"Detect each of the following objects in {where} and return its bounding box{rule}.\n"
            f"Objects:\n{listing}\n"
            "Return strict JSON only, as a list: "
            "[{\"id\": \"tape\", \"bbox_2d\": [x1, y1, x2, y2]}, ...] with one entry per object id above.")
    scene = [{"type": "image_url", "image_url": {"url": rg.data_url(scene_path)}}]
    ref = [{"type": "image_url", "image_url": {"url": rg.data_url(ref_path)}}]
    if variant == "双图-相对":  # 参考图在前，和最早的主测试布局一样
        content = ([{"type": "text", "text": "Image 1 (a reference photo, not needed for this task):"}] + ref
                   + [{"type": "text", "text": "Image 2 (current view from the robot's head camera):"}] + scene)
    elif variant == "双图-场景在前":
        content = ([{"type": "text", "text": "Image 1 (current view from the robot's head camera):"}] + scene
                   + [{"type": "text", "text": "Image 2 (a reference photo, not needed for this task):"}] + ref)
    else:
        content = list(scene)
    content += [{"type": "text", "text": task}]
    return [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": content}]


def _item_id(it):
    """编号优先看 id；Qwen3.5/3.6 常把编号写进 label，还会写成「prism: small gray ...」，取冒号前那段。"""
    raw = it.get("id") or it.get("label") or ""
    return str(raw).split(":")[0].strip()


def parse_boxes(text):
    """返回 {id: [x1, y1, x2, y2]}。先当 JSON 解析，不行再按 id 逐个正则兜底。"""
    s = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\[.*\]", s, re.S)
    if m:
        try:
            items = json.loads(m.group(0))
            out = {_item_id(it): list(map(float, it["bbox_2d"])) for it in items
                   if isinstance(it, dict) and _item_id(it) and it.get("bbox_2d")}
            if out:
                return out
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    out = {}
    for mm in re.finditer(r"\"(?:id|label)\"\s*:\s*\"(\w+)[^\"]*\"[^\]]*?\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)", s):
        out.setdefault(mm.group(1), [float(v) for v in mm.groups()[1:]])
    return out


def fit(pairs):
    """最小二乘 模型坐标 = a × 像素 + b，返回 (a, b, 最大残差像素)。"""
    n = len(pairs)
    if n < 2:
        return None
    mx = sum(p for p, _ in pairs) / n
    my = sum(r for _, r in pairs) / n
    sxx = sum((p - mx) ** 2 for p, _ in pairs)
    if sxx == 0:
        return None
    a = sum((p - mx) * (r - my) for p, r in pairs) / sxx
    b = my - a * mx
    worst = max(abs((r - b) / a - p) for p, r in pairs) if a else float("inf")
    return a, b, worst


def analyse(boxes, objects, sw, sh):
    xs, ys = [], []
    for oid, o in objects.items():
        if oid in boxes:
            g, r = o["框"], boxes[oid]
            xs += [(g[0], r[0]), (g[2], r[2])]
            ys += [(g[1], r[1]), (g[3], r[3])]
    fx, fy = fit(xs), fit(ys)
    out = {"框出物体数": f"{len(boxes)}/{len(objects)}", "横向拟合": None, "纵向拟合": None}
    if fx:
        out["横向拟合"] = {"斜率": round(fx[0], 3), "截距": round(fx[1], 1), "等效宽": round(999 / fx[0]) if fx[0] else None,
                       "拟合后最大残差像素": round(fx[2], 1)}
    if fy:
        out["纵向拟合"] = {"斜率": round(fy[0], 3), "截距": round(fy[1], 1), "等效高": round(999 / fy[0]) if fy[0] else None,
                       "拟合后最大残差像素": round(fy[2], 1)}
    for conv in ["relative", "pixel"]:
        ious = [rg.iou(rg.to_pixel(boxes[oid], conv, sw, sh), o["框"]) for oid, o in objects.items() if oid in boxes]
        out[f"按 {conv} 的平均交并比"] = round(sum(ious) / len(ious), 2) if ious else None
    return out


def draw(scene_path, objects, boxes, sw, sh, title, out_path):
    im = Image.open(scene_path).convert("RGB")
    d = ImageDraw.Draw(im)
    for oid, o in objects.items():
        d.rectangle(o["框"], outline=(0, 200, 0), width=2)
        if oid in boxes:
            b = rg.to_pixel(boxes[oid], "relative", sw, sh)
            d.rectangle(b, outline=(255, 0, 0), width=2)
            d.text((b[0] + 2, b[1] + 2), oid, fill=(255, 0, 0))
    d.text((6, sh - 16), f"{title}   green = truth, red = model box read as relative 0-999", fill=(255, 255, 0))
    im.save(out_path)


def print_line(entry):
    fx, fy = entry.get("横向拟合"), entry.get("纵向拟合")
    line = f"{entry['模型']:18s} {entry['问法']:6s} "
    if "错误" in entry:
        line += f"出错 {entry['错误'][:120]}"
    else:
        line += (f"框出 {entry['框出物体数']}  "
                 f"横向 斜率 {fx['斜率'] if fx else '-'} 截距 {fx['截距'] if fx else '-'} 等效宽 {fx['等效宽'] if fx else '-'}  "
                 f"纵向 斜率 {fy['斜率'] if fy else '-'} 截距 {fy['截距'] if fy else '-'} 等效高 {fy['等效高'] if fy else '-'}  "
                 f"平均交并比 相对 {entry['按 relative 的平均交并比']} 像素 {entry['按 pixel 的平均交并比']}  "
                 f"{entry.get('耗时秒', '-')} 秒")
    print(line)


def main():
    ap = argparse.ArgumentParser(description="坐标探针")
    ap.add_argument("case_dir")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--reanalyze", metavar="结果目录", help="不调接口，用已存的原始回复重新解析和拟合（改了解析规则后用）")
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    ap.add_argument("--base-url", default=rg.DEFAULT_BASE_URL)
    ap.add_argument("--thinking", choices=["off", "on", "unset"], default="off")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--dry-run", action="store_true", help="只拼请求，不调接口")
    args = ap.parse_args()

    case_dir = os.path.abspath(args.case_dir)
    scene_path, ref_path = os.path.join(case_dir, "scene.png"), os.path.join(case_dir, "ref.png")
    with open(os.path.join(case_dir, "物体框.json"), encoding="utf-8") as f:
        objects = json.load(f)["物体"]
    sw, sh = Image.open(scene_path).size

    if args.reanalyze:
        rdir = os.path.abspath(args.reanalyze)
        with open(os.path.join(rdir, "结果.json"), encoding="utf-8") as f:
            report = json.load(f)
        print(f"重新解析 {rdir}（场景图 {sw}x{sh}，正确时等效宽 {sw}、等效高 {sh}）\n")
        for entry in report:
            if "原始回复" not in entry:
                continue
            boxes = parse_boxes(entry["原始回复"])
            entry["解析出的框"] = boxes
            entry.update(analyse(boxes, objects, sw, sh))
            draw(scene_path, objects, boxes, sw, sh, f"{entry['模型']}_{entry['问法']}",
                 os.path.join(rdir, f"{entry['模型']}_{entry['问法']}.png"))
            print_line(entry)
        with open(os.path.join(rdir, "结果.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return
    if not args.models:
        ap.error("要给 --models，或者用 --reanalyze")

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("没有设置 DASHSCOPE_API_KEY。先 export DASHSCOPE_API_KEY=sk-xxxx，或者加 --dry-run")

    out_dir = os.path.join(rg.HERE, "结果", f"探针_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"场景图 {sw}x{sh}：相对坐标正确时横向斜率应 ≈ {999 / sw:.3f}、纵向 ≈ {999 / sh:.3f}，截距 ≈ 0\n")

    report = []
    for model in args.models:
        for variant in args.variants:
            messages = build_probe_messages(variant, objects, scene_path, ref_path)
            tag = f"{model}_{variant}"
            if args.dry_run:
                with open(os.path.join(out_dir, f"{tag}_请求.json"), "w", encoding="utf-8") as f:
                    names = {"双图-相对": ["ref.png", "scene.png"],
                             "双图-场景在前": ["scene.png", "ref.png"]}.get(variant, ["scene.png"])
                    json.dump(rg.preview_messages(messages, names), f, ensure_ascii=False, indent=2)
                continue
            entry = {"模型": model, "问法": variant}
            try:
                data, latency = rg.call_api(args.base_url, api_key, model, messages, args.timeout, args.thinking)
                text = data["choices"][0]["message"].get("content") or ""
                boxes = parse_boxes(text)
                entry.update({"耗时秒": round(latency, 2), "原始回复": text, "解析出的框": boxes})
                entry.update(analyse(boxes, objects, sw, sh))
                draw(scene_path, objects, boxes, sw, sh, tag, os.path.join(out_dir, f"{tag}.png"))
            except urllib.error.HTTPError as e:
                entry["错误"] = f"{e.code}：{e.read().decode(errors='replace')[:300]}"
            report.append(entry)
            print_line(entry)

    if args.dry_run:
        print(f"模拟运行完成，请求在 {out_dir}")
        return
    with open(os.path.join(out_dir, "结果.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n结果在 {out_dir}")


if __name__ == "__main__":
    main()
