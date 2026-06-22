#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生物视觉启发的 CNN 卷积演示 — Python 版（与 HTML / MATLAB 三端联动）

=====================================================================
 黄金校验例（三端冻结，用于抓翻转/填充/off-by-one bug）
---------------------------------------------------------------------
 输入 5×5（外圈 0、内 3×3 为 1）：
   0 0 0 0 0
   0 1 1 1 0
   0 1 1 1 0
   0 1 1 1 0
   0 0 0 0 0
 核 = horiz_edge（3×3 Sobel）：
   -1 -2 -1
    0  0  0
    1  2  1
 参数：bias=0, stride=1, padding=valid, 无 ReLU（互相关，不翻转）
 期望输出 3×3：
    3  4  3
    0  0  0
   -3 -4 -3
 运行： python bio_cnn_demo.py --golden   （应打印上面这组整数）
=====================================================================

共享规范（见 demo/README.md 第四节）：
 - 互相关（cross-correlation），核不做 180° 翻转（与 PyTorch/TF 一致）
 - 4×4 偶数核 "same" 填充用 TF/Keras 约定：上/左=floor((k-1)/2)，下/右=ceil((k-1)/2) → k=4 时上1下2
 - 灰度统一 Rec.601：0.299R + 0.587G + 0.114B
 - 特征图零中心发散色图（负蓝/零白/正红），vmin=-vmax=max(|f|)
"""

import argparse
import sys
import numpy as np

# Windows GBK 控制台兼容：强制 stdout/stderr 用 UTF-8，避免 emoji/特殊字符崩溃
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ============================================================
#  预设卷积核（与 HTML / MATLAB 逐字一致）
#  concept 用于生物学概念归类；w4=None 表示仅 3×3
# ============================================================
PRESETS = {
    "identity":            dict(label="恒等 Identity",                       concept="identity",       scale=1.0,
                                w3=[[0,0,0],[0,1,0],[0,0,0]],
                                w4=[[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]),
    "horiz_edge":          dict(label="水平边缘 Horiz edge (V1 0°)",          concept="orientation",    scale=1.0,
                                w3=[[-1,-2,-1],[0,0,0],[1,2,1]],
                                w4=[[-2,-2,-2,-2],[-1,-1,-1,-1],[1,1,1,1],[2,2,2,2]]),
    "vert_edge":           dict(label="垂直边缘 Vert edge (V1 90°)",          concept="orientation",    scale=1.0,
                                w3=[[-1,0,1],[-2,0,2],[-1,0,1]],
                                w4=[[-2,-1,1,2],[-2,-1,1,2],[-2,-1,1,2],[-2,-1,1,2]]),
    "diag_45_edge":        dict(label="45° 对角 Diag 45°",                    concept="orientation",    scale=1.0,
                                w3=[[0,1,2],[-1,0,1],[-2,-1,0]],
                                w4=[[0,1,2,3],[-1,0,1,2],[-2,-1,0,1],[-3,-2,-1,0]]),
    "diag_135_edge":       dict(label="135° 对角 Diag 135°",                  concept="orientation",    scale=1.0,
                                w3=[[2,1,0],[1,0,-1],[0,-1,-2]],
                                w4=[[3,2,1,0],[2,1,0,-1],[1,0,-1,-2],[0,-1,-2,-3]]),
    "center_surround_dog": dict(label="中心-周边拮抗 Center-surround (LGN·RGC)", concept="center_surround", scale=1.0,
                                w3=[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],
                                w4=[[-1,-1,-1,-1],[-1,3,3,-1],[-1,3,3,-1],[-1,-1,-1,-1]]),
    "laplacian":           dict(label="拉普拉斯 Laplacian",                   concept="center_surround", scale=1.0,
                                w3=[[0,1,0],[1,-4,1],[0,1,0]], w4=None),
    "gabor_45":            dict(label="Gabor 朝向 Gabor (V1 调谐)",           concept="orientation",    scale=1.0,
                                w3=[[1,1,1],[1,-2,-1],[1,-1,-2]], w4=None),
    "box_blur":            dict(label="盒式模糊 Box blur",                    concept="sum",            scale=1.0/9.0,
                                w3=[[1,1,1],[1,1,1],[1,1,1]],
                                w4=[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]),
    "sharpen":             dict(label="锐化 Sharpen",                         concept="sum",            scale=1.0,
                                w3=[[0,-1,0],[-1,5,-1],[0,-1,0]],
                                w4=[[0,-1,-1,0],[-1,5,5,-1],[-1,5,5,-1],[0,-1,-1,0]]),
    "emboss":              dict(label="浮雕 Emboss",                          concept="sum",            scale=1.0,
                                w3=[[-2,-1,0],[-1,1,1],[0,1,2]],
                                w4=[[-2,-2,-1,0],[-2,-1,1,1],[-1,1,1,2],[0,1,2,2]]),
}

# 生物视觉 ↔ CNN 映射（演示主线，对应讲座 PPT 第4-6页）
MAPPING = [
    ("视网膜感光细胞",        "输入像素矩阵"),
    ("感受野 (局部连接)",     "滑动的卷积核窗口"),
    ("V1 简单细胞·朝向选择",  "朝向边缘/Gabor 核"),
    ("中心-周边拮抗 (LGN)",   "中心-周边/拉普拉斯核"),
    ("空间总和 (树突求和)",   "加权点积 Σ(w·x)+b"),
    ("权值共享 (朝向柱)",     "同一核扫过全图"),
    ("动作电位放电阈值",      "ReLU = max(0,x)"),
    ("V1 复杂细胞 (不变性)",  "最大池化 MaxPool"),
    ("腹侧流层级 (V1→IT)",    "堆叠多层卷积"),
]


# ============================================================
#  核心算法
# ============================================================
def pad_amounts(mode, k):
    """返回 (top, bot, left, right)。same 模式用 TF/Keras 约定（k=4 → 上1下2）。"""
    if mode == "same":
        top = (k - 1) // 2
        bot = (k - 1) - top          # = ceil((k-1)/2)
        return top, bot, top, bot
    return 0, 0, 0, 0


def conv2d(x, kernel, bias=0.0, stride=1, padding="valid", relu=False, scale=1.0):
    """
    二维互相关（cross-correlation，不翻转），与深度学习约定一致。
    x: HxW 灰度 [0,1]；kernel: kxk；返回 outHxoutW 特征图。
    """
    x = np.asarray(x, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    kh, kw = k.shape
    assert kh == kw, "仅支持方核"
    H, W = x.shape
    top, bot, left, right = pad_amounts(padding, kh)
    xp = np.pad(x, ((top, bot), (left, right)), mode="constant")
    pH, pW = xp.shape
    outH = (pH - kh) // stride + 1
    outW = (pW - kw) // stride + 1
    out = np.zeros((outH, outW), dtype=np.float64)
    for oy in range(outH):
        for ox in range(outW):
            by, bx = oy * stride, ox * stride
            win = xp[by:by + kh, bx:bx + kw]
            out[oy, ox] = float(np.sum(win * k)) * scale + bias
    if relu:
        out = np.maximum(out, 0.0)
    return out


def maxpool2x2(x):
    """2×2 最大池化，stride=2，奇数维截断。对应 V1 复杂细胞。"""
    x = np.asarray(x, dtype=np.float64)
    H, W = x.shape
    pH, pW = H // 2, W // 2
    out = np.zeros((pH, pW), dtype=np.float64)
    for y in range(pH):
        for ox in range(pW):
            out[y, ox] = np.max(x[y * 2:y * 2 + 2, ox * 2:ox * 2 + 2])
    return out


def to_gray(rgb):
    """Rec.601 灰度：0.299R+0.587G+0.114B，归一化到 [0,1]。与 Canvas 一致。"""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim == 2:
        return rgb / 255.0 if rgb.max() > 1.0 else rgb
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


# ============================================================
#  程序化样本刺激（与 HTML 同算法 → 跨端可精确比对）
# ============================================================
def make_stimulus(kind, size=48):
    if kind == "golden":
        size = 5
    img = np.zeros((size, size), dtype=np.float64)
    c = (size - 1) / 2.0
    for y in range(size):
        for x in range(size):
            if kind == "circle":
                dx, dy = x - c, y - c
                v = 0.5 * (1 - min(1.0, np.hypot(dx, dy) / (size * 0.42)))
            elif kind == "gradient":
                dx, dy = x - c, y - c
                v = max(0.0, 1 - np.hypot(dx, dy) / (size * 0.7))
            elif kind == "grid":
                v = 0.85 if (int(np.floor(x / (size / 8.0))) % 2) ^ (int(np.floor(y / (size / 8.0))) % 2) else 0.12
            elif kind in ("bars0", "bars45", "bars90", "bars135"):
                # 朝向 θ 的光棒：调制方向 = θ+90°（垂直于光棒长轴）
                ang = {"bars0": np.pi / 2, "bars45": 3 * np.pi / 4, "bars90": np.pi, "bars135": 5 * np.pi / 4}[kind]
                px = (x - c) * np.cos(ang) + (y - c) * np.sin(ang)
                freq = 2 * np.pi * 3 / size          # 3 个周期
                v = 0.5 + 0.5 * np.sin(px * freq)
            elif kind == "golden":
                v = 1.0 if (1 <= x <= 3 and 1 <= y <= 3) else 0.0
            else:
                v = 0.0
            img[y, x] = min(1.0, max(0.0, v))
    return img


def load_image(path, max_dim=256):
    """用 PIL 打开图片 → Rec.601 灰度 → 降采样到 max_dim。"""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    scale = min(1.0, max_dim / max(im.width, im.height))
    tw = max(1, int(round(im.width * scale)))
    th = max(1, int(round(im.height * scale)))
    im = im.resize((tw, th), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float64)
    return to_gray(arr)


# ============================================================
#  黄金例自检
# ============================================================
def golden_test(verbose=True):
    img = make_stimulus("golden", 5)
    ker = np.array(PRESETS["horiz_edge"]["w3"], dtype=np.float64)
    out = conv2d(img, ker, bias=0.0, stride=1, padding="valid", relu=False, scale=1.0)
    expected = np.array([[3, 4, 3], [0, 0, 0], [-3, -4, -3]], dtype=np.float64)
    ok = np.allclose(out, expected)
    if verbose:
        print("=== 黄金校验例 ===")
        print("输入 5×5:\n", img.astype(int))
        print("核 horiz_edge:\n", ker.astype(int))
        print("输出:\n", np.round(out, 4))
        print("期望:\n", expected)
        print("结果:", "[通过 / PASS]" if ok else "[失败 / FAIL]（检查翻转/填充/off-by-one）")
    return ok


# ============================================================
#  可视化
# ============================================================
def _setup_cjk_font():
    """让 matplotlib 能显示中文标题（按平台选常见 CJK 字体）。"""
    import matplotlib as mpl
    cjk = ["Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC", "Noto Sans CJK SC",
           "Source Han Sans SC", "WenQuanYi Zen Hei", "Arial Unicode MS"]
    existing = list(mpl.rcParams.get("font.sans-serif", []))
    mpl.rcParams["font.sans-serif"] = cjk + [f for f in existing if f not in cjk]
    mpl.rcParams["axes.unicode_minus"] = False


def visualize(img, kernel, out, preset_key, pooled=None, out_path=None, show=False):
    import matplotlib.pyplot as plt
    _setup_cjk_font()
    fig, axes = plt.subplots(1, 4 if pooled is not None else 3, figsize=(16, 4.5))
    if pooled is None:
        axes = list(axes)
    vmax_k = max(1e-9, float(np.max(np.abs(kernel))))
    vmax_f = max(1e-9, float(np.max(np.abs(out))))

    ax = axes[0]
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("输入图 Input\n(视网膜感光细胞阵列)", fontsize=10)

    ax = axes[1]
    ax.imshow(kernel, cmap="bwr", vmin=-vmax_k, vmax=vmax_k, interpolation="nearest")
    ax.set_title(f"卷积核 Kernel\n{PRESETS[preset_key]['label']}\n(感受野 / V1简单细胞)", fontsize=10)
    for (yy, xx), val in np.ndenumerate(kernel):
        ax.text(xx, yy, f"{val:g}", ha="center", va="center", fontsize=9,
                color="black" if abs(val) < vmax_k * 0.6 else "white")

    ax = axes[2]
    ax.imshow(out, cmap="bwr", vmin=-vmax_f, vmax=vmax_f, interpolation="nearest")
    relu_tag = " + ReLU" if False else ""
    ax.set_title(f"输出特征图 Feature Map\n({out.shape[0]}×{out.shape[1]})\n(空间总和·权值共享)", fontsize=10)

    if pooled is not None:
        vmax_p = max(1e-9, float(np.max(np.abs(pooled))))
        axes[3].imshow(pooled, cmap="bwr", vmin=-vmax_p, vmax=vmax_p, interpolation="nearest")
        axes[3].set_title("最大池化 MaxPool\n(V1 复杂细胞)", fontsize=10)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("生物视觉启发的卷积神经网络演示  ·  Bio-Inspired CNN Demo", fontsize=12)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"已保存图：{out_path}")
    if show:
        plt.show()
    plt.close(fig)


def animate_gif(img, kernel, out, out_path, preset_key, padding="valid", stride=1, bias=0.0, relu=False, scale=1.0):
    """生成滑动感受野动画 GIF（需要 Pillow）。"""
    try:
        from PIL import Image
    except ImportError:
        print("[注意] 未安装 Pillow，无法生成 GIF。请 pip install Pillow。")
        return
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    _setup_cjk_font()

    top, bot, left, right = pad_amounts(padding, kernel.shape[0])
    xp = np.pad(img, ((top, bot), (left, right)), mode="constant")
    vmax_f = max(1e-9, float(np.max(np.abs(out))))
    outH, outW = out.shape
    frames = []
    total = outH * outW
    step = max(1, total // 60)            # 约 60 帧
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    canvas = FigureCanvasAgg(fig)

    def render(reveal):
        partial = np.full_like(out, np.nan)
        flat = out.reshape(-1)
        partial.reshape(-1)[:reveal] = flat[:reveal]
        axes[0].clear(); axes[1].clear()
        axes[0].imshow(xp, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[0].set_title(f"含填充输入 ({xp.shape[0]}×{xp.shape[1]})", fontsize=10)
        if reveal > 0:
            idx = reveal - 1
            ox, oy = idx % outW, idx // outW
            by, bx = oy * stride, ox * stride
            k = kernel.shape[0]
            axes[0].add_patch(plt.Rectangle((bx - 0.5, by - 0.5), k, k, fill=False, edgecolor="#fbbf24", lw=2))
        axes[0].set_xticks([]); axes[0].set_yticks([])
        cmap = plt.cm.bwr.copy(); cmap.set_bad("#151c25")
        axes[1].imshow(partial, cmap=cmap, vmin=-vmax_f, vmax=vmax_f, interpolation="nearest")
        axes[1].set_title(f"特征图 {reveal}/{total}", fontsize=10)
        axes[1].set_xticks([]); axes[1].set_yticks([])
        fig.canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())
        frames.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE))

    r = 0
    while r < total:
        r = min(total, r + step)
        render(r)
    plt.close(fig)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=True)
    print(f"已保存动画：{out_path}（{len(frames)} 帧）")


# ============================================================
#  CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="生物视觉启发的 CNN 卷积演示（Python 版）")
    ap.add_argument("--image", help="输入图片路径（不传则用 --stimulus）")
    ap.add_argument("--stimulus", default="bars45",
                    choices=["circle", "bars0", "bars45", "bars90", "bars135", "gradient", "grid", "golden"],
                    help="过程化样本刺激（默认 bars45）")
    ap.add_argument("--size", type=int, default=48, help="样本尺寸（golden 固定为 5）")
    ap.add_argument("--preset", default="horiz_edge", choices=list(PRESETS.keys()))
    ap.add_argument("--ksize", type=int, default=3, choices=[3, 4], help="核尺寸 3 或 4")
    ap.add_argument("--bias", type=float, default=0.0)
    ap.add_argument("--stride", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--padding", default="valid", choices=["valid", "same"])
    ap.add_argument("--relu", action="store_true", help="启用 ReLU（=放电阈值）")
    ap.add_argument("--maxpool", action="store_true", help="启用 2×2 最大池化（=复杂细胞）")
    ap.add_argument("--out", default="feature_map.png", help="输出图路径")
    ap.add_argument("--show", action="store_true", help="弹窗显示")
    ap.add_argument("--animate", help="生成滑动感受野动画 GIF（需 Pillow）")
    ap.add_argument("--golden", action="store_true", help="运行黄金校验自检并退出")
    args = ap.parse_args()

    if args.golden:
        golden_test()
        return

    # 取图
    if args.image:
        img = load_image(args.image)
    else:
        img = make_stimulus(args.stimulus, args.size)

    # 取核
    p = PRESETS[args.preset]
    w = p["w3"] if args.ksize == 3 else p["w4"]
    if w is None:
        raise SystemExit(f"预设 '{args.preset}' 仅支持 3×3，请用 --ksize 3 或换预设。")
    kernel = np.array(w, dtype=np.float64)

    # 卷积 + 可选池化
    out = conv2d(img, kernel, bias=args.bias, stride=args.stride,
                 padding=args.padding, relu=args.relu, scale=p["scale"])
    pooled = maxpool2x2(out) if args.maxpool else None

    print(f"输入 {img.shape[1]}×{img.shape[0]} | 核 {args.ksize}×{args.ksize} ({args.preset}) | "
          f"stride {args.stride} | padding {args.padding} | bias {args.bias} | ReLU {args.relu}")
    print(f"输出特征图 {out.shape[1]}×{out.shape[0]}，值范围 [{out.min():.3f}, {out.max():.3f}]")

    visualize(img, kernel, out, args.preset, pooled=pooled, out_path=args.out, show=args.show)

    if args.animate:
        animate_gif(img, kernel, out, args.animate, args.preset,
                    padding=args.padding, stride=args.stride, bias=args.bias,
                    relu=args.relu, scale=p["scale"])


if __name__ == "__main__":
    main()