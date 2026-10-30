"""参考图指认物体的最小测试：给视觉语言模型一张物体参考图 + 一张场景图，让它在场景图里框出这个物体。
默认模型 qwen3-vl-plus（10-17 测试后定下的方案，结论见同目录 README.md）。

用法（在 算法/DexGraspVLA- 目录下）：
    export DASHSCOPE_API_KEY=sk-xxxx
    PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python 规划器测试/ref_grounding_min.py 规划器测试/测试集/案例01
    # 只拼请求、画真值框，不调接口：加 --dry-run
    # 国际账号：加 --base-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    # 控制台给的是带业务空间编号的新地址（…cn-beijing.maas.aliyuncs.com/compatible-mode/v1）就照抄给 --base-url
    # 看账号能调哪些模型：加 --list-models
    # 换模型：--model qwen3.6-35b-a3b 等。qwen2.5-vl 系列 2026 年已在百炼下线（返回 403 access_denied）

坐标约定：Qwen2.5-VL 输出像素坐标；Qwen3-VL、Qwen3.5~3.8 输出相对坐标，按百炼文档归一化到 0~999。
--coords auto 按模型名选，提示词里的坐标要求也跟着换；结果里另按「另一种约定」算一次交并比，约定配错时一眼能看出来。

思考模式：qwen3.5~3.8 默认开着、qwen3-vl-plus 默认关着。--thinking off（默认）统一关掉，各模型对比条件一致；
on 打开（输出上限放宽到 4096，思考过程也计费）；unset 不发这个参数（给非千问模型用）。

输入方式：--input 图片（默认）/ 图片加外形 / 文字，三种只在「怎么指认目标」上不同，输入里不带位置信息。
后两种用 case.json 的「外形说明」；文字是对照组，看参考图本身起多大作用。

案例目录里要有 ref.png、scene.png、case.json（真值框、干扰物框、外形说明）；结果写到 规划器测试/结果/案例名_输入_时间/。
只用标准库 + PIL，不依赖 openai 包；直连接口、不走 .bashrc 里的代理，和规划器的行为一致。
"""
import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-vl-plus"

# 提示词沿用规划器 bounding_box_prediction 的写法，只把文字指令换成参考图。
# 三种输入只在「怎么指认目标」上不同，其余措辞一致；输入里不带任何位置信息，reason 也只许描述外形：
#   图片        参考图
#   图片加外形  参考图 + 一句外形说明（case.json 的「外形说明」）
#   文字        只有外形说明，不给参考图（对照组：看参考图本身起多大作用）
INPUT_MODES = ["图片", "图片加外形", "文字"]
INPUT_TAGS = {"图片": "image", "图片加外形": "image+shape", "文字": "text"}  # 画图用，PIL 默认字体不认中文

# 图片顺序：参考在前（默认，和最早的测试一样）/ 场景在前。
# 10-17 坐标探针：Qwen3.5/3.6 只发场景图时框得准，场景图前面多一张参考图坐标就整体偏（qwen3-vl-plus 不受影响），
# 所以加「场景在前」，让要框的那张图排第一。
ORDERS = ["参考在前", "场景在前"]

PROMPT_INTRO = (
    "You are controlling a robotic arm. The user says \"Pick up this object\" and shows a reference photo of it. "
    "The object may not have a common name, so match it by visual appearance (shape, color, proportions, texture), "
    "not by category alone. The reference photo may be taken from a different camera, viewpoint and scale."
)
PROMPT_REF = "REF_NAME (reference photo of the target object):"
PROMPT_SHAPE = "The user also describes the object's appearance: \"SHAPE\"."
PROMPT_TEXT = "You are controlling a robotic arm. The user says \"Pick up the SHAPE.\""
PROMPT_SCENE = "SCENE_NAME (current view from the robot's head camera):"
PROMPT_TASK = (
    "Task: locate TARGET within SCENE_NAME.\n"
    "Rules:\n"
    "1. Return EXACTLY ONE bounding box, COORD_RULE.\n"
    "2. Similar-looking objects may be present; choose the one that best matches MATCH.\n"
    "3. If the object is not visible in SCENE_NAME, set \"bbox_2d\" to null. Do not guess.\n"
    "Return strict JSON only:\n"
    "{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"a short 2-4 word name\", "
    "\"reason\": \"one sentence on which visual features matched; describe appearance only, not location\"}"
)
COORD_RULES = {
    "pixel": "in SCENE_NAME's pixel coordinates",
    "relative": "in coordinates relative to SCENE_NAME, normalized to the range 0-999 for both width and height",
}
RELATIVE_SCALE = 999  # 百炼文档：Qwen3-VL 及 Qwen3.5~3.8「坐标值会归一化到 [0, 999]」


def data_url(path):
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def build_messages(ref_path, scene_path, coords, mode="图片", shape=None, order="参考在前"):
    text_only = mode == "文字"
    scene_first = text_only or order == "场景在前"
    scene_name, ref_name = ("Image 1", "Image 2") if scene_first else ("Image 2", "Image 1")

    def fill(s):
        return (s.replace("TARGET", "the described object" if text_only else "the object shown in REF_NAME")
                 .replace("SCENE_NAME", scene_name).replace("REF_NAME", ref_name)
                 .replace("MATCH", "the description" if text_only else "the reference"))

    scene_part = [{"type": "image_url", "image_url": {"url": data_url(scene_path)}}]
    task_part = [{"type": "text", "text": fill(PROMPT_TASK.replace("COORD_RULE", COORD_RULES[coords]))}]
    if text_only:
        content = [{"type": "text", "text": PROMPT_TEXT.replace("SHAPE", shape) + "\n" + fill(PROMPT_SCENE)}]
        content += scene_part + task_part
    else:
        ref_part = [{"type": "image_url", "image_url": {"url": data_url(ref_path)}}]
        if mode == "图片加外形":
            ref_part.append({"type": "text", "text": PROMPT_SHAPE.replace("SHAPE", shape)})
        if scene_first:
            content = [{"type": "text", "text": PROMPT_INTRO + "\n" + fill(PROMPT_SCENE)}] + scene_part
            content += [{"type": "text", "text": fill(PROMPT_REF)}] + ref_part
        else:
            content = [{"type": "text", "text": PROMPT_INTRO + "\n" + fill(PROMPT_REF)}] + ref_part
            content += [{"type": "text", "text": fill(PROMPT_SCENE)}] + scene_part
        content += task_part
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": content},
    ]


def preview_messages(messages, names):
    """把 base64 换成图片名，存档时能看清发了什么。"""
    out = json.loads(json.dumps(messages))
    it = iter(names)
    for part in out[1]["content"]:
        if part["type"] == "image_url":
            part["image_url"]["url"] = f"[图片 {next(it)}]"
    return out


def _request(url, api_key, timeout, body=None):
    req = urllib.request.Request(
        url, data=body, method="POST" if body else "GET",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 直连
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def is_local(base_url):
    host = urllib.parse.urlparse(base_url).hostname or ""
    return host in ("127.0.0.1", "localhost")


def call_api(base_url, api_key, model, messages, timeout, thinking):
    payload = {"model": model, "messages": messages, "max_tokens": 4096 if thinking == "on" else 512}
    if thinking != "unset":
        # 百炼认顶层的 enable_thinking；本机 vLLM 会忽略它（于是默认开思考），要放进 chat_template_kwargs
        if is_local(base_url):
            payload["chat_template_kwargs"] = {"enable_thinking": thinking == "on"}
        else:
            payload["enable_thinking"] = thinking == "on"
    body = json.dumps(payload).encode()
    t0 = time.time()
    data = _request(base_url.rstrip("/") + "/chat/completions", api_key, timeout, body)
    return data, time.time() - t0


def list_models(base_url, api_key, timeout):
    """全部列出：新的多模态模型（qwen3.7-plus 之类）名字里不带 vl，按名字筛会漏。"""
    data = _request(base_url.rstrip("/") + "/models", api_key, timeout)
    return sorted(m["id"] for m in data.get("data", []))


def to_pixel(box, coords, w, h):
    if coords == "relative":
        s = RELATIVE_SCALE
        return [box[0] * w / s, box[1] * h / s, box[2] * w / s, box[3] * h / s]
    return list(box)


def parse_box(text):
    """模型常见输出：裸 JSON、```json 代码块、或者套一层列表。返回 (bbox 或 None, 解析出的字典)。"""
    s = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"[\[{].*[\]}]", s, re.S)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        obj = obj[0]
    if isinstance(obj, dict) and "bbox_2d" in obj:
        box = obj["bbox_2d"]
        return (list(map(float, box)) if box else None), obj
    m = re.search(r"bbox_2d\"?\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)", s)
    if m:
        return [float(v) for v in m.groups()], {"bbox_2d": "正则兜底解析"}
    raise ValueError("回复里找不到 bbox_2d")


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def qwen_resized(w, h, factor=28):
    """Qwen2.5-VL 开源实现的默认缩放：边长取最近的 28 的倍数。服务端参数可能不同，只作参考。"""
    return round(w / factor) * factor, round(h / factor) * factor


def visualize(ref_path, scene_path, pred, case, lines, out_path):
    """ref_path 为 None（文字对照组没发参考图）时只画场景图。"""
    scene = Image.open(scene_path).convert("RGB")
    ref = None
    if ref_path:
        ref = Image.open(ref_path).convert("RGB")
        ref = ref.resize((round(ref.width * scene.height / ref.height), scene.height))
    off = ref.width if ref else 0
    canvas = Image.new("RGB", (off + scene.width, scene.height + 18 * len(lines) + 8), "white")
    if ref:
        canvas.paste(ref, (0, 0))
    canvas.paste(scene, (off, 0))
    d = ImageDraw.Draw(canvas)

    def box(b, color, width):
        d.rectangle([b[0] + off, b[1], b[2] + off, b[3]], outline=color, width=width)

    if case.get("干扰物框"):
        box(case["干扰物框"], (255, 200, 0), 1)
    if case.get("真值框"):
        box(case["真值框"], (0, 200, 0), 2)
    if pred:
        box(pred, (255, 0, 0), 3)
    for i, line in enumerate(lines):
        d.text((6, scene.height + 4 + 18 * i), line, fill="black")
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser(description="参考图指认物体的最小测试")
    ap.add_argument("case_dir", nargs="?", help="案例目录，含 ref.png、scene.png、case.json")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--input", choices=INPUT_MODES, default="图片",
                    help="怎么指认目标：图片 / 图片加外形 / 文字（后两种用 case.json 的「外形说明」）")
    ap.add_argument("--order", choices=ORDERS, default="参考在前",
                    help="两张图的顺序；Qwen3.5/3.6 要用 场景在前，参考在前时它们给的坐标会整体偏")
    ap.add_argument("--coords", choices=["auto", "pixel", "relative"], default="auto",
                    help="模型输出的坐标约定；auto = 模型名带 qwen3 用 relative（0~999），否则 pixel")
    ap.add_argument("--thinking", choices=["off", "on", "unset"], default="off",
                    help="思考模式；off 统一关掉（默认），unset 不发这个参数")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--dry-run", action="store_true", help="只拼请求和画真值框，不调接口")
    ap.add_argument("--list-models", action="store_true", help="列出这个密钥能调的全部模型后退出")
    args = ap.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("没有设置 DASHSCOPE_API_KEY。先 export DASHSCOPE_API_KEY=sk-xxxx，或者加 --dry-run")
    if args.list_models:
        try:
            models = list_models(args.base_url, api_key, args.timeout)
        except urllib.error.HTTPError as e:
            raise SystemExit(f"列模型失败（{e.code}）：{e.read().decode(errors='replace')}\n这个接口不支持列模型的话，去控制台模型广场看")
        print("\n".join(models) if models else "这个密钥下一个模型都没列出来")
        print("注意：能不能看图要查模型说明，名字里不带 vl 的也可能是多模态的")
        return
    if not args.case_dir:
        ap.error("要给案例目录")

    coords = args.coords
    if coords == "auto":
        coords = "relative" if "qwen3" in args.model.lower() else "pixel"
    other_coords = "pixel" if coords == "relative" else "relative"

    case_dir = os.path.abspath(args.case_dir)
    ref_path, scene_path = os.path.join(case_dir, "ref.png"), os.path.join(case_dir, "scene.png")
    with open(os.path.join(case_dir, "case.json"), encoding="utf-8") as f:
        case = json.load(f)
    sw, sh = Image.open(scene_path).size
    rw, rh = Image.open(ref_path).size
    shape = case.get("外形说明")
    if args.input != "图片" and not shape:
        raise SystemExit(f"--input {args.input} 要用 case.json 里的「外形说明」，这个案例没写")
    shown_ref = None if args.input == "文字" else ref_path

    order_tag = "_场景在前" if args.order == "场景在前" and args.input != "文字" else ""
    base = os.path.join(HERE, "结果",
                        f"{os.path.basename(case_dir)}_{args.input}{order_tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir, n = base, 1
    while os.path.exists(out_dir):  # 同一秒内连跑
        n += 1
        out_dir = f"{base}_{n}"
    os.makedirs(out_dir)
    messages = build_messages(ref_path, scene_path, coords, args.input, shape, args.order)
    names = [f"scene.png {sw}x{sh}"]
    if shown_ref is not None:
        names = names + [f"ref.png {rw}x{rh}"] if args.order == "场景在前" else [f"ref.png {rw}x{rh}"] + names
    with open(os.path.join(out_dir, "请求.json"), "w", encoding="utf-8") as f:
        json.dump(preview_messages(messages, names), f, ensure_ascii=False, indent=2)

    result = {"案例": os.path.basename(case_dir), "模型": args.model, "输入": args.input,
              "图片顺序": None if args.input == "文字" else args.order,
              "外形说明": shape if args.input != "图片" else None, "接口": args.base_url,
              "坐标约定": coords, "思考模式": args.thinking, "场景图尺寸": [sw, sh], "模型内部尺寸估算": list(qwen_resized(sw, sh))}

    if args.dry_run:
        lines = [f"dry run ({INPUT_TAGS[args.input]}): green = ground truth (eyeballed), yellow = distractor"]
        visualize(shown_ref, scene_path, None, case, lines, os.path.join(out_dir, "可视化.png"))
        print(f"模拟运行完成，请求和真值框图在 {out_dir}")
        return

    try:
        data, latency = call_api(args.base_url, api_key, args.model, messages, args.timeout, args.thinking)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        hint = ""
        if e.code == 401:
            hint = ("\n密钥无效，或者接口地址不对：国内账号用 dashscope.aliyuncs.com，国际账号用 dashscope-intl.aliyuncs.com；"
                    "控制台显示的是带业务空间编号的地址就照抄。都用 --base-url 传")
        elif e.code == 403:
            hint = (f"\n密钥是对的，但没有 {args.model} 的调用权限，或者这个模型已下线（qwen2.5-vl 系列 2026 年已在百炼下线）。"
                    "用 --list-models 看能调哪些，再用 --model 换")
        elif e.code == 400:
            hint = ("\n请求参数不被接受：可能这个模型不支持图片输入，或者不认 enable_thinking（试 --thinking unset），"
                    "或者开思考时要求流式调用")
        raise SystemExit(f"接口返回 {e.code}：{body}{hint}")

    message = data["choices"][0]["message"]
    text = message.get("content") or ""
    result.update({"耗时秒": round(latency, 2), "用量": data.get("usage"), "原始回复": text})
    if message.get("reasoning_content"):
        result["思考过程"] = message["reasoning_content"]

    raw, pred = None, None
    try:
        raw, parsed = parse_box(text)
        result["解析结果"] = parsed
    except ValueError as e:
        result["解析结果"] = f"失败：{e}"

    if raw is None:
        verdict = "解析失败" if isinstance(result["解析结果"], str) else "模型回答看不见"
    else:
        pred = to_pixel(raw, coords, sw, sh)
        result.update({"原始框": raw, "预测框（像素）": pred})
        limit_w, limit_h = (RELATIVE_SCALE, RELATIVE_SCALE) if coords == "relative" else (sw, sh)
        if raw[2] > limit_w * 1.02 or raw[3] > limit_h * 1.02:
            result["警告"] = f"原始坐标超出 {coords} 约定的范围：可能用了参考图的坐标系，或者坐标约定配错了"
        iou_gt = iou(pred, case["真值框"]) if case.get("真值框") else None
        iou_dis = iou(pred, case["干扰物框"]) if case.get("干扰物框") else None
        result.update({"与真值交并比": iou_gt, "与干扰物交并比": iou_dis})
        if case.get("真值框"):
            iou_other = iou(to_pixel(raw, other_coords, sw, sh), case["真值框"])
            result[f"按 {other_coords} 换算的与真值交并比"] = iou_other
            if iou_other >= 0.5 > iou_gt:
                result["警告"] = f"按 {other_coords} 换算才对得上真值，坐标约定可能配错了，试 --coords {other_coords}"
        if iou_gt is not None and iou_gt >= 0.5:
            verdict = "命中"
        elif iou_dis is not None and iou_dis >= 0.5:
            verdict = "选中干扰物"
        else:
            verdict = "未命中"
    result["结论"] = verdict

    with open(os.path.join(out_dir, "结果.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    def fmt(v):
        return "-" if v is None else f"{v:.2f}"
    pred_str = None if pred is None else [round(v) for v in pred]
    lines = [f"red = prediction {pred_str} (raw {raw}, {coords}), green = ground truth (eyeballed), yellow = distractor",
             f"IoU gt = {fmt(result.get('与真值交并比'))}   IoU distractor = {fmt(result.get('与干扰物交并比'))}"
             f"   latency = {latency:.1f}s   model = {args.model}   input = {INPUT_TAGS[args.input]}"]
    visualize(shown_ref, scene_path, pred, case, lines, os.path.join(out_dir, "可视化.png"))

    print(f"结论：{verdict}（{args.model}，输入：{args.input}{'，' + args.order if args.input != '文字' else ''}）")
    print(f"预测框（像素）：{pred_str}   原始框：{raw}（{coords}）   真值框：{case.get('真值框')}   耗时 {latency:.1f} 秒")
    print(f"回复：{text}")
    if "警告" in result:
        print(f"警告：{result['警告']}")
    print(f"结果在 {out_dir}")


if __name__ == "__main__":
    main()
