%% 生物视觉启发的 CNN 卷积演示 — MATLAB / Octave 版（与 HTML / Python 三端联动）
% =========================================================================
% 黄金校验例（三端冻结，用于抓翻转/填充/off-by-one bug）
% -------------------------------------------------------------------------
%  输入 5×5（外圈 0、内 3×3 为 1）+ horiz_edge（3×3 Sobel）+ bias0 + stride1 + valid + 无ReLU
%  期望输出 3×3：
%     3  4  3
%     0  0  0
%    -3 -4 -3
%  运行：把 mode 设为 'golden' 后按 F5，应打印上面这组整数。
%
% 共享规范（见 demo/README.md 第四节）：
%  - 互相关（不翻转）；与 PyTorch/TF 一致
%  - 4×4 偶数核 "same" 用 TF/Keras 约定：上/左=floor((k-1)/2)，下/右=ceil((k-1)/2) → k=4 时 上1下2
%  - 灰度统一 Rec.601：0.299R+0.587G+0.114B（注意：MATLAB 自带 rgb2gray 用 Rec.709，会不一致，这里手写覆盖）
%  - 特征图零中心发散色图（负蓝/零白/正红），caxis=[-vmax,vmax], vmax=max(|f|)
% =========================================================================

clear; clc;
% Octave 下尽量用 qt 工具箱以获得较好的 imagesc 渲染
try, if exist('OCTAVE_VERSION','builtin'); graphics_toolkit('qt'); end; catch; end

%% ===================== 顶部配置区（讲台临时修改这里）=====================
mode     = 'demo';        % 'demo' 演示  |  'golden' 黄金自检
preset   = 'horiz_edge';  % 预设核名（见下方 getKernel）
ksize    = 3;             % 核尺寸 3 或 4
imgSrc   = 'stimulus';    % 'stimulus' 或 'file'
stimulus = 'bars45';      % circle/bars0/bars45/bars90/bars135/gradient/grid/golden
imgSize  = 48;            % 样本尺寸（golden 固定 5）
imgFile  = 'my_photo.jpg';% imgSrc='file' 时使用
bias     = 0;             % 偏置 b
stride   = 1;             % 1/2/3
padding  = 'valid';       % 'valid' 无填充  |  'same' 同尺寸(TF/Keras)
useReLU  = false;         % ReLU = 放电阈值
usePool  = false;         % 2×2 最大池化 = V1 复杂细胞
outFile  = 'feature_map.png';
%% =======================================================================

%% 主程序
if strcmp(mode, 'golden')
    golden_test();
    return;   % 脚本中 return 直接退出
end

% --- 1) 取图（Rec.601 灰度）---
if strcmp(imgSrc, 'file')
    gray = load_gray(imgFile);
else
    gray = make_stimulus(stimulus, imgSize);
end

% --- 2) 取核 ---
[kern, scale, label] = getKernel(preset, ksize);
if isempty(kern)
    error('bio_cnn_demo:badPreset', '预设 ''%s'' 不支持 %dx%d，请换预设或用 ksize=3。', preset, ksize, ksize);
end

% --- 3) 卷积 + 可选池化 ---
feat = conv2d_bio(gray, kern, bias, stride, padding, useReLU, scale);
if usePool
    pooled = maxpool2x2(feat);
else
    pooled = [];
end

fprintf('输入 %dx%d | 核 %dx%d (%s) | stride %d | %s | bias %g | ReLU %d\n', ...
        size(gray, 2), size(gray, 1), ksize, ksize, preset, stride, padding, bias, useReLU);
fprintf('输出特征图 %dx%d，值范围 [%.3f, %.3f]\n', size(feat, 2), size(feat, 1), min(feat(:)), max(feat(:)));

% --- 4) 可视化 ---
visualize(gray, kern, feat, label, pooled, usePool, outFile);
fprintf('已保存图：%s\n', outFile);


% =========================================================================
%  本地函数（脚本末尾，R2016b+/Octave 6+ 支持）
% =========================================================================

function [k, scale, label] = getKernel(preset, ksize)
%GETKERNEL 预设卷积核（与 HTML / Python 逐字一致）。
% 返回 k=[] 表示该预设不支持此尺寸。
switch preset
    case 'identity'
        if ksize==3, k=[0 0 0;0 1 0;0 0 0]; else, k=zeros(4); k(2,2)=1; end
        scale=1; label='恒等 Identity';
    case 'horiz_edge'
        if ksize==3, k=[-1 -2 -1;0 0 0;1 2 1]; else, k=[-2 -2 -2 -2;-1 -1 -1 -1;1 1 1 1;2 2 2 2]; end
        scale=1; label='水平边缘 Horiz (V1 0°)';
    case 'vert_edge'
        if ksize==3, k=[-1 0 1;-2 0 2;-1 0 1]; else, k=[-2 -1 1 2;-2 -1 1 2;-2 -1 1 2;-2 -1 1 2]; end
        scale=1; label='垂直边缘 Vert (V1 90°)';
    case 'diag_45_edge'
        if ksize==3, k=[0 1 2;-1 0 1;-2 -1 0]; else, k=[0 1 2 3;-1 0 1 2;-2 -1 0 1;-3 -2 -1 0]; end
        scale=1; label='45° 对角 Diag 45°';
    case 'diag_135_edge'
        if ksize==3, k=[2 1 0;1 0 -1;0 -1 -2]; else, k=[3 2 1 0;2 1 0 -1;1 0 -1 -2;0 -1 -2 -3]; end
        scale=1; label='135° 对角 Diag 135°';
    case 'center_surround_dog'
        if ksize==3, k=[-1 -1 -1;-1 8 -1;-1 -1 -1]; else, k=[-1 -1 -1 -1;-1 3 3 -1;-1 3 3 -1;-1 -1 -1 -1]; end
        scale=1; label='中心-周边拮抗 Center-surround (LGN·RGC)';
    case 'laplacian'
        if ksize==3, k=[0 1 0;1 -4 1;0 1 0]; else, k=[]; end
        scale=1; label='拉普拉斯 Laplacian';
    case 'gabor_45'
        if ksize==3, k=[1 1 1;1 -2 -1;1 -1 -2]; else, k=[]; end
        scale=1; label='Gabor 朝向 (V1 调谐)';
    case 'box_blur'
        if ksize==3, k=ones(3); else, k=ones(4); end
        scale=1/size(k,1)^2; label='盒式模糊 Box blur';
    case 'sharpen'
        if ksize==3, k=[0 -1 0;-1 5 -1;0 -1 0]; else, k=[0 -1 -1 0;-1 5 5 -1;-1 5 5 -1;0 -1 -1 0]; end
        scale=1; label='锐化 Sharpen';
    case 'emboss'
        if ksize==3, k=[-2 -1 0;-1 1 1;0 1 2]; else, k=[-2 -2 -1 0;-2 -1 1 1;-1 1 1 2;0 1 2 2]; end
        scale=1; label='浮雕 Emboss';
    otherwise
        k=[]; scale=1; label='';
end
end


function out = conv2d_bio(x, k, bias, stride, padding, relu, scale)
%CONV2D_BIO 二维互相关（不翻转），与 Python/JS 算法完全一致，保证三端像素级一致。
% 注意：为保证 4×4 偶数核 "same" 填充与 TF/Keras 约定一致，这里手写循环，
%       而非用 imfilter（后者对偶数核的 padding 与本约定不同，会破坏跨端一致性）。
%  ---- 等价的 imfilter 写法（仅奇核 + same 模式近似）----
%    feat = imfilter(x, k, 0, 'corr') * scale + bias;   % 'corr' 显式：互相关，不翻转
  [H, W] = size(x); [kh, kw] = size(k);
  if strcmp(padding, 'same')
      top = floor((kh-1)/2);    bot = (kh-1) - top;
      lp  = floor((kw-1)/2);    rp  = (kw-1) - lp;
  else
      top=0; bot=0; lp=0; rp=0;
  end
  xp = zeros(H+top+bot, W+lp+rp);
  xp(top+1:top+H, lp+1:lp+W) = x;
  pH = size(xp, 1); pW = size(xp, 2);
  outH = floor((pH-kh)/stride) + 1;
  outW = floor((pW-kw)/stride) + 1;
  out = zeros(outH, outW);
  for oy = 1:outH
      for ox = 1:outW
          by = (oy-1)*stride; bx = (ox-1)*stride;
          win = xp(by+1:by+kh, bx+1:bx+kw);
          out(oy, ox) = sum(win(:) .* k(:)) * scale + bias;
      end
  end
  if relu, out(out < 0) = 0; end
end


function out = maxpool2x2(x)
%MAXPOOL2X2 2×2 最大池化，stride=2，奇数维截断。对应 V1 复杂细胞。
  [H, W] = size(x); pH = floor(H/2); pW = floor(W/2);
  out = zeros(pH, pW);
  for y = 1:pH
      for ox = 1:pW
          blk = x(y*2-1:y*2, ox*2-1:ox*2);
          out(y, ox) = max(blk(:));
      end
  end
end


function g = load_gray(path)
%LOAD_GRAY 读取图片 → Rec.601 灰度 → 降采样到 ≤256。
% 注意：不用 rgb2gray（它用 Rec.709 0.21/0.72/0.07，会与 Canvas/Python 不一致）。
  im = imread(path);
  d = im2double(im);
  if size(d, 3) == 3
      g = 0.299*d(:,:,1) + 0.587*d(:,:,2) + 0.114*d(:,:,3);
  else
      g = d;
  end
  mx = max(size(g,1), size(g,2));
  if mx > 256
      sc = 256 / mx;
      g = imresize(g, [round(size(g,1)*sc), round(size(g,2)*sc)]);
  end
end


function img = make_stimulus(kind, sz)
%MAKE_STIMULUS 过程化样本刺激（与 Python/HTML 同算法）。
  if strcmp(kind, 'golden'), sz = 5; end
  img = zeros(sz);
  c = (sz-1)/2;
  for y = 1:sz
      for x = 1:sz
          dx = (x-1)-c; dy = (y-1)-c;
          switch kind
              case 'circle'
                  v = 0.5*(1 - min(1, hypot(dx,dy)/(sz*0.42)));
              case 'gradient'
                  v = max(0, 1 - hypot(dx,dy)/(sz*0.7));
              case 'grid'
                  v = 0.85*(mod(floor((x-1)/(sz/8))+floor((y-1)/(sz/8)), 2)~=0) + 0.12;
              case {'bars0','bars45','bars90','bars135'}
                  % 朝向 θ 的光棒：调制方向 = θ+90°（垂直于光棒长轴）
                  switch kind
                      case 'bars0', ang=pi/2; case 'bars45', ang=3*pi/4;
                      case 'bars90', ang=pi; case 'bars135', ang=5*pi/4; end
                  px = dx*cos(ang) + dy*sin(ang);
                  freq = 2*pi*3/sz;       % 3 个周期
                  v = 0.5 + 0.5*sin(px*freq);
              case 'golden'
                  if x>=2 && x<=4 && y>=2 && y<=4, v=1; else, v=0; end
              otherwise
                  v = 0;
          end
          img(y, x) = min(1, max(0, v));
      end
  end
end


function lut = bwr_lut()
%BWR_LUT 256×3 蓝-白-红 发散色图（Octave 无 blue2red，手建）。
  n = 128;
  neg = [linspace(0,1,n)', linspace(0,1,n)', ones(n,1)];   % 蓝→白
  pos = [ones(n,1), linspace(1,0,n)', linspace(1,0,n)'];   % 白→红
  lut = [neg; pos];
end


function visualize(gray, kern, feat, label, pooled, usePool, outFile)
%VISUALIZE 绘制 输入 | 核 | 特征图（+池化）。
  bwr = bwr_lut();
  np = 3 + usePool;
  fig = figure('Color','w','Position',[100 100 360*np 420]);
  set(0, 'DefaultAxesFontSize', 10);

  % 输入
  ax1 = subplot(1, np, 1);
  imagesc(gray); axis image off; colormap(ax1, gray(256));
  caxis(ax1, [0 1]);
  title(ax1, {'输入图 Input','(视网膜感光细胞阵列)'});

  % 核
  ax2 = subplot(1, np, 2);
  imagesc(kern); axis image off; colormap(ax2, bwr);
  vk = max(1e-9, max(abs(kern(:)))); caxis(ax2, [-vk vk]);
  [kr, kc] = size(kern);
  for r = 1:kr, for c = 1:kc
      col = [0 0 0]; if abs(kern(r,c)) >= vk*0.6, col = [1 1 1]; end
      text(ax2, c, r, sprintf('%g', kern(r,c)), ...
           'HorizontalAlignment','center','FontSize',9,'Color',col);
  end, end
  title(ax2, {'卷积核 Kernel', label, '(感受野 / V1 简单细胞)'});

  % 特征图
  ax3 = subplot(1, np, 3);
  imagesc(feat); axis image off; colormap(ax3, bwr);
  vf = max(1e-9, max(abs(feat(:)))); caxis(ax3, [-vf vf]);
  title(ax3, {'输出特征图 Feature Map', ...
              sprintf('(%dx%d)', size(feat,2), size(feat,1)), '(空间总和·权值共享)'});

  % 池化
  if usePool
      ax4 = subplot(1, np, 4);
      imagesc(pooled); axis image off; colormap(ax4, bwr);
      vp = max(1e-9, max(abs(pooled(:)))); caxis(ax4, [-vp vp]);
      title(ax4, {'最大池化 MaxPool','(V1 复杂细胞)'});
  end

  % sgtitle 仅 MATLAB R2018b+ 有；Octave 无，用 try/catch 兜底（面板标题已含信息）
  try
      sgtitle(fig, '生物视觉启发的卷积神经网络演示 · Bio-Inspired CNN Demo');
  catch
      annotation(fig, 'textbox', [0.3 0.93 0.4 0.06], 'String', ...
          '生物视觉启发的卷积神经网络演示 · Bio-Inspired CNN Demo', ...
          'EdgeColor','none','HorizontalAlignment','center','FontSize',11);
  end
  try, exportgraphics(fig, outFile, 'Resolution', 120); catch, print(fig, outFile, '-dpng', '-r120'); end
end


function golden_test()
%GOLDEN_TEST 黄金校验自检。
  img = make_stimulus('golden', 5);
  k = [-1 -2 -1; 0 0 0; 1 2 1];
  out = conv2d_bio(img, k, 0, 1, 'valid', false, 1);
  expected = [3 4 3; 0 0 0; -3 -4 -3];
  ok = max(abs(out(:) - expected(:))) < 1e-9;
  fprintf('=== 黄金校验例 ===\n');
  fprintf('输出: \n'); print_mat(out);
  fprintf('期望: \n'); print_mat(expected);
  if ok, fprintf('结果: [通过 / PASS]\n'); else, fprintf('结果: [失败 / FAIL]（检查翻转/填充/off-by-one）\n'); end
end


function print_mat(M)
%PRINT_MAT 紧凑打印整数矩阵。
  for r = 1:size(M,1)
    fprintf('  ');
    for c = 1:size(M,2)
      fprintf('%6.2f ', M(r,c));
    end
    fprintf('\n');
  end
end