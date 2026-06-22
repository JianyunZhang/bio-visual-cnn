# Python 版 · 生物视觉 CNN 卷积演示

NumPy 手写卷积 + Matplotlib 可视化，依赖轻量（核心仅需 `numpy` + `matplotlib`）。与 HTML / MATLAB 三端共享同一套数值规范（见上级 [`../README.md`](../README.md) 第四节）。

## 安装

```bash
cd demo/python
pip install -r requirements.txt          # numpy + matplotlib（Pillow 可选）
```

> 仅 `numpy + matplotlib` 即可运行所有 `--stimulus` 样本模式与 `--golden` 自检。
> `Pillow` 仅在使用 `--image 上传图片` 或 `--animate 生成GIF` 时需要。

## 快速开始

```bash
# 1) 黄金校验自检（应输出 3 4 3 / 0 0 0 / -3 -4 -3，并打印 [通过 / PASS]）
python bio_cnn_demo.py --golden

# 2) 斜光棒刺激 + 水平边缘核（V1 0° 朝向选择），保存特征图
python bio_cnn_demo.py --stimulus bars45 --size 64 --preset horiz_edge --out out.png

# 3) 圆形 + 垂直边缘核，并叠加最大池化（=V1 复杂细胞）
python bio_cnn_demo.py --stimulus circle --size 48 --preset vert_edge --maxpool --out out_pool.png

# 4) 上传自己的图片（自动 Rec.601 灰度 + 降采样）
python bio_cnn_demo.py --image my_photo.jpg --preset sharpen --padding same --relu --out out.png

# 5) 生成滑动感受野动画 GIF（需 Pillow）
python bio_cnn_demo.py --stimulus circle --size 48 --preset gabor_45 --animate slide.gif
```

## 命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--image PATH` | — | 输入图片路径（不传则用 `--stimulus`） |
| `--stimulus` | `bars45` | 过程化刺激：`circle / bars0 / bars45 / bars90 / bars135 / gradient / grid / golden` |
| `--size N` | `48` | 样本尺寸（`golden` 固定为 5） |
| `--preset` | `horiz_edge` | 预设核：见下表 |
| `--ksize` | `3` | 核尺寸 3 或 4（4×4 仅部分预设支持） |
| `--bias` | `0` | 偏置 b |
| `--stride` | `1` | 步长 1/2/3 |
| `--padding` | `valid` | `valid` 无填充 / `same` 同尺寸（TF/Keras） |
| `--relu` | off | 启用 ReLU（=放电阈值） |
| `--maxpool` | off | 启用 2×2 最大池化（=复杂细胞） |
| `--out` | `feature_map.png` | 输出图路径 |
| `--show` | off | 弹窗显示 |
| `--animate PATH` | — | 生成滑动感受野 GIF（需 Pillow） |
| `--golden` | — | 运行黄金自检并退出 |

## 预设核（生物学归类）

| 预设 | 中文 | 生物学概念 | 3×3 | 4×4 |
|---|---|---|---|---|
| `horiz_edge` | 水平边缘 | V1 简单细胞 0° 朝向 | ✓ | ✓ |
| `vert_edge` | 垂直边缘 | V1 简单细胞 90° 朝向 | ✓ | ✓ |
| `diag_45_edge` | 45° 对角 | V1 朝向 | ✓ | ✓ |
| `diag_135_edge` | 135° 对角 | V1 朝向 | ✓ | ✓ |
| `gabor_45` | Gabor 朝向 | V1 调谐曲线 | ✓ | ✗ |
| `center_surround_dog` | 中心-周边拮抗 | LGN·RGC 给光中心 | ✓ | ✓ |
| `laplacian` | 拉普拉斯 | 零交叉检测 | ✓ | ✗ |
| `box_blur` | 盒式模糊 | 局部平均 | ✓ | ✓ |
| `sharpen` | 锐化 | 高频增强 | ✓ | ✓ |
| `emboss` | 浮雕 | 方向差分 | ✓ | ✓ |
| `identity` | 恒等 | 对照 | ✓ | ✓ |

## 作为模块导入

```python
import numpy as np
from bio_cnn_demo import conv2d, PRESETS, make_stimulus, maxpool2x2

img = make_stimulus("bars45", 64)                      # 64×64 斜光棒
ker = np.array(PRESETS["horiz_edge"]["w3"])            # 3×3 水平边缘
feat = conv2d(img, ker, padding="valid", relu=True)    # 特征图
pooled = maxpool2x2(feat)                              # 复杂细胞
```

## 黄金校验例（三端冻结）

```
输入 5×5（外0内1）+ horiz_edge 3×3 + valid + 无ReLU → 输出 3×3：
   3  4  3
   0  0  0
  -3 -4 -3
```
`python bio_cnn_demo.py --golden` 必须打印 `[通过 / PASS]`。任何一端算不出这组整数即有 bug。

## 与其它端的一致性

- **互相关**（不翻转），与 PyTorch/TF 一致。
- **灰度** 统一 Rec.601（0.299/0.587/0.114），与 Canvas 一致。
- **4×4 same 填充** 用 TF/Keras 约定（上1下2）。
- 过程化刺激（`bars45` 等）与 HTML 同算法，可做像素级比对（`max(|A−B|)<1e-4`）。