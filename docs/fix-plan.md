# 训练管线 Bug 修复计划

> **状态：已全部修复并验证 ✅** (2026-05-18)
>
> 训练 1000 次后 loss 从 5.48 降至 0.93，OCR 输出 `001.png` → `:40A:` 正确。

## 问题背景

导出的 `.traineddata` 能被 Tesseract 正确加载（无 assert 错误），但 OCR 输出质量差：`001.png` 期望 `:40A:`，实际输出 `oll Eh`。

经调查发现训练管线存在 7 个 bug，其中 3 个是致命的。

**与 Tesseract 官方源码（`/tmp/tesseract`）对比验证结果**：

| Bug | 计划准确性 | 修复状态 | 说明 |
|-----|-----------|---------|------|
| Bug 1 | ✅ 正确 | ✅ 已修复 | Convolve 确认无权重（`convolve.h:63-66` 只有 `half_x_`, `half_y_`） |
| Bug 2 | ✅ 正确 | ✅ 已修复 | Tesseract 的 `Lrx` = plain LSTM + Reversed wrapper，LSTM 无 reverse 标志 |
| Bug 3 | ✅ 已修正 | ✅ 已修复 | 原计划 `percentile(5/95)` 不正确，已改为**中间行局部极值**算法 |
| Bug 4 | ✅ 正确 | ✅ 已修复 | Tesseract 用 Leptonica `pixScale`，非 PIL 的 BILINEAR/LANCZOS |
| Bug 5 | ✅ 正确 | ✅ 已修复 | 缺少 import |
| Bug 6 | ✅ 正确 | ✅ 已修复 | 未知字符发出 warnings.warn 而非静默替换 |
| Bug 7 | ✅ 新增 | ✅ 已修复 | Tesseract 用随机噪声填充，非零值（`networkio.cpp:247-249`） |

---

## Bug 列表

### Bug 1（致命）：ConvolveLayer 权重从未从 traineddata 加载

**文件**：`src/tesseract_cuda/network/weight_mapper.py:31-36`，`src/tesseract_cuda/network/model.py:111-196`

**原因**：Tesseract 的 `Ct3,3,16` 在二进制中存储为 `Series(Convolve[无权重], Tanh[有权重])`。`_load_recursive` 有特殊路径处理此模式，但只在 PyTorch 模型的**第一层**是 ConvolveLayer 时触发。实际第一层是 InputLayer，特殊路径永远不触发。ConvolveLayer 权重保持随机初始化。

**修复**：新增 `_try_load_convolve()` 函数，在 SeriesLayer 遍历中检测 `Series(Convolve, Activation)` → ConvolveLayer 映射，将 Activation 的权重加载到 ConvolveLayer.fc。修复后验证权重 std=0.27（非随机）。

### Bug 2（致命）：双向反转相互抵消

**文件**：`src/tesseract_cuda/network/model.py:155-166`

**原因**：对于反向 LSTM（`Lrx96`），代码同时做了：
- `LSTMLayer(reverse=True)` — 内部翻转输入/输出
- `ReversedLayer(dim="x")` — 外部再次翻转

4 次翻转互相抵消，反向 LSTM 等价于前向 LSTM。模型完全没有右→左的上下文能力。

**修复**：所有 LSTMLayer 统一传 `reverse=False`，方向翻转完全由 `ReversedLayer` 处理。`_build_series_children` 和 `_build_parallel` 均已修正。

### Bug 3（致命）：图像归一化不一致

**文件**：`src/tesseract_cuda/training/dataset.py:54`，`src/tesseract_cuda/recognizer.py:42-49`

**原因**：

| | 修复前（训练） | 修复后（训练/推理统一） |
|---|---|---|
| 归一化范围 | [-0.5, 0.5] | [-1.0, 1.0] |
| 方法 | `pixel/255 - 0.5` | 中间行局部极值对比度拉伸（ComputeBlackWhite） |

**修复**：在 `dataset.py` 中新增 `compute_black_white()` 和 `tesseract_normalize()` 函数，实现 Tesseract 官方的 `ComputeBlackWhite` 算法（`src/lstm/networkio.cpp:122-158`）。`recognizer.py` 也改为调用同一函数，确保训练/推理一致。

**Tesseract 官方算法**（`src/lstm/networkio.cpp:122-158`, `ComputeBlackWhite`）：

扫描图像中间行，收集所有局部最小值和局部最大值，取局部最小值的 25th percentile 为 black，取局部最大值的 75th percentile 为 white。归一化公式：`contrast = max((white - black) / 2.0, 1.0)`，`pixel = (pixel - black) / contrast - 1.0`。

### Bug 4（中等）：重采样方法不一致

**文件**：`src/tesseract_cuda/training/dataset.py:50`

**原因**：训练用 BILINEAR，推理用 LANCZOS。

**修复**：`PILImage.Resampling.BILINEAR` → `PILImage.Resampling.LANCZOS`。

### Bug 5（中等）：recognizer.py 缺少 `TESSDATA_UNICHARSET` 导入

**文件**：`src/tesseract_cuda/recognizer.py:5`，`src/tesseract_cuda/recognizer.py:21`

**原因**：第 21 行引用 `TESSDATA_UNICHARSET` 但未在 import 中包含。

**修复**：在 import 行添加 `TESSDATA_UNICHARSET`。

### Bug 6（轻微）：未知字符退化为 CTC blank

**文件**：`src/tesseract_cuda/training/dataset.py:70-73`

**原因**：当字符无法通过 recoder 编码时，退回 `null_char_id`（0），即 CTC blank。模型学习到未知字符应预测为空。

**修复**：改为 `warnings.warn()` 发出警告，而非静默替换为 blank。

### Bug 7（轻微）：宽度填充方式不一致

**文件**：`src/tesseract_cuda/training/dataset.py:89`

**原因**：Tesseract 在图像宽度不足时填充 **[-1, +1] 随机噪声**（`networkio.cpp:247-249`, `Randomize`），而我们的 dataset.py 用零填充。随机噪声是 Tesseract 的设计选择，避免模型过拟合到零值边界。

**修复**：`zeros(h, max_w - w)` → `torch.rand(h, max_w - w) * 2.0 - 1.0`。

---

## 修复步骤

| 步骤 | 操作 | 文件 | 状态 |
|------|------|------|------|
| 1 | 修复 ConvolveLayer 权重加载 | `weight_mapper.py` | ✅ 已完成 |
| 2 | 修复反向 LSTM 双重翻转 | `model.py` | ✅ 已完成 |
| 3 | 统一图像归一化（ComputeBlackWhite） | `dataset.py`, `recognizer.py` | ✅ 已完成 |
| 4 | 统一重采样方法为 LANCZOS | `dataset.py` | ✅ 已完成 |
| 5 | 修复缺少的导入 | `recognizer.py` | ✅ 已完成 |
| 6 | 未知字符改为 warnings.warn | `dataset.py` | ✅ 已完成 |
| 7 | 修正宽度填充为随机噪声 | `dataset.py` | ✅ 已完成 |
| 8 | 重新训练（1000 iterations, GPU） | — | ✅ 已完成 |
| 9 | 重新导出并测试 | — | ✅ 已完成 |

---

## 实际训练结果

| 指标 | 修复前（v3） | 修复后（v4） |
|------|------------|------------|
| 训练迭代 | 5000 | 1000 |
| 最终 loss | 1.03 | 0.93 |
| OCR `001.png` | `oll Eh` | `:40A:` ✅ |
| 训练时间 | ~34 min | ~7 min (422s) |

**训练日志**（`output/training_v4.log`）：

```
iteration  100/100,  Mean loss=5.476575, elapsed=43s
iteration  200/200,  Mean loss=3.977799, elapsed=84s
iteration  300/300,  Mean loss=2.990287, elapsed=129s
iteration  400/400,  Mean loss=2.306429, elapsed=172s
iteration  500/500,  Mean loss=1.857045, elapsed=215s
iteration  600/600,  Mean loss=1.552600, elapsed=257s
iteration  700/700,  Mean loss=1.332705, elapsed=299s
iteration  800/800,  Mean loss=1.167003, elapsed=340s
iteration  900/900,  Mean loss=1.037556, elapsed=381s
iteration 1000/1000, Mean loss=0.933913, elapsed=422s
```

**输出文件**：
- `output/finetuned_v4_checkpoint.pt` — PyTorch checkpoint
- `output/finetuned_v4_1000.traineddata` — 最终模型
- 已部署至 `/data/traineddata/eng.traineddata`

**Java Tess4J 验证**：
```
LD_LIBRARY_PATH=/data/tesseract/lib mvn compile exec:java -Dexec.mainClass=org.example.App
输出：:40A:  ✅
```

---

## 修改文件摘要

| 文件 | 变更 |
|------|------|
| `src/tesseract_cuda/network/weight_mapper.py` | 新增 `_try_load_convolve()`，修复 Series 层遍历逻辑 |
| `src/tesseract_cuda/network/model.py` | 所有 LSTMLayer 传 `reverse=False`，方向由 ReversedLayer 处理 |
| `src/tesseract_cuda/training/dataset.py` | 新增 `compute_black_white()` + `tesseract_normalize()`；BILINEAR→LANCZOS；zeros→随机噪声填充；unknown char→warnings |
| `src/tesseract_cuda/recognizer.py` | 添加 `TESSDATA_UNICHARSET` import；归一化改用 `compute_black_white()` |
| `tests/test_weight_mapper.py` | 修复 `_extract_recursive()` 调用缺少 `preserved_convs` 参数 |

---

## 后续优化方向（非必需）

- 增加训练迭代次数（1000 → 5000+），进一步降低 loss
- 数据增强（旋转、噪声、弹性变形）
- 增加训练样本数量和多样性
- 对所有 15 张图片做端到端验证，而非仅 `001.png`
