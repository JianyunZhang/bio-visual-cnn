# MATLAB / Octave 版 · 生物视觉 CNN 卷积演示

与 HTML / Python 三端共享同一套数值规范（见上级 [`../README.md`](../README.md) 第四节）。手写互相关循环（保证 4×4 偶数核 `same` 填充与 TF/Keras 约定一致），不依赖任何工具箱即可跑刺激/黄金模式。

## 运行

### 方式一：MATLAB
1. 打开 `bio_cnn_demo.m`。
2. 修改顶部 **「配置区」**：`mode` / `preset` / `ksize` / `stimulus` / `imgSize` / `padding` / `useReLU` / `usePool` 等。
3. 按 **F5** 运行。图保存到 `outFile`（默认 `feature_map.png`）。

### 方式二：GNU Octave（免费）
```bash
octave bio_cnn_demo.m
```
> Octave 需 `image` 包（仅当 `imgSrc='file'` 用到 `imresize`/`im2double`）。纯刺激模式只需 Octave 核心。

## 黄金自检

把顶部 `mode = 'golden'` 后运行，应打印：
```
输出:
     3.00   4.00   3.00
     0.00   0.00   0.00
    -3.00  -4.00  -3.00
期望:
     3.00   4.00   3.00
     ...
结果: [通过 / PASS]
```

## 配置区常用项

| 变量 | 取值 | 说明 |
|---|---|---|
| `mode` | `'demo'` / `'golden'` | 演示 / 黄金自检 |
| `preset` | 见下表 | 预设核 |
| `ksize` | `3` / `4` | 核尺寸（4×4 仅部分预设） |
| `imgSrc` | `'stimulus'` / `'file'` | 过程化刺激 / 读取图片 |
| `stimulus` | `circle/bars0/bars45/bars90/bars135/gradient/grid/golden` | 刺激种类 |
| `imgSize` | 整数 | 样本尺寸（`golden` 固定 5） |
| `imgFile` | 路径 | `imgSrc='file'` 时使用 |
| `bias` | 数 | 偏置 b |
| `stride` | 1/2/3 | 步长 |
| `padding` | `'valid'` / `'same'` | 无填充 / 同尺寸(TF/Keras) |
| `useReLU` | `true/false` | ReLU = 放电阈值 |
| `usePool` | `true/false` | 2×2 最大池化 = 复杂细胞 |
| `outFile` | 路径 | 输出 PNG |

## 预设核（生物学归类）

| 预设 | 中文 | 生物学概念 | 3×3 | 4×4 |
|---|---|---|---|---|
| `horiz_edge` | 水平边缘 | V1 简单细胞 0° | ✓ | ✓ |
| `vert_edge` | 垂直边缘 | V1 简单细胞 90° | ✓ | ✓ |
| `diag_45_edge` | 45° 对角 | V1 朝向 | ✓ | ✓ |
| `diag_135_edge` | 135° 对角 | V1 朝向 | ✓ | ✓ |
| `gabor_45` | Gabor 朝向 | V1 调谐 | ✓ | ✗ |
| `center_surround_dog` | 中心-周边拮抗 | LGN·RGC | ✓ | ✓ |
| `laplacian` | 拉普拉斯 | 零交叉 | ✓ | ✗ |
| `box_blur` | 盒式模糊 | 局部平均 | ✓ | ✓ |
| `sharpen` | 锐化 | 高频增强 | ✓ | ✓ |
| `emboss` | 浮雕 | 方向差分 | ✓ | ✓ |
| `identity` | 恒等 | 对照 | ✓ | ✓ |

## 已知注意点（Octave / 工具箱）

- **卷积用手写循环**而非 `imfilter`：因 `imfilter` 对 4×4 偶数核的 padding 与本约定的 TF/Keras 方式不同，会破坏跨端一致性。代码内附 `imfilter(...,'corr')` 等价注释（仅奇核 same 近似）。
- **灰度手写 Rec.601**（0.299/0.587/0.114），**不用 `rgb2gray`**（它用 Rec.709，会与 Canvas/Python 不一致）。
- **依赖**：刺激/黄金模式只需核心；`imgSrc='file'` 的 `imread/im2double/imresize` 在 MATLAB 需 Image Processing Toolbox，在 Octave 需 `image` 包。
- **`sgtitle`/`exportgraphics`** 仅新版 MATLAB 有，已用 `try/catch` 兜底（Octave 自动回退）。

## 三端一致性

`max(|matlab − python|) < 1e-9`（同为 double，过程化刺激同算法）。