# Tesseract `.traineddata` 模型文件格式详解

本文档详细描述 Tesseract OCR 的 `.traineddata` 文件的二进制结构，包括容器格式、LSTM 网络序列化、权重矩阵布局、网络描述语言、以及所有组件类型。

---

## 目录

1. [容器格式](#1-容器格式)
2. [组件类型列表](#2-组件类型列表)
3. [二进制序列化基础（TFile）](#3-二进制序列化基础tfile)
4. [LSTM 网络二进制格式](#4-lstm-网络二进制格式)
5. [网络节点通用头](#5-网络节点通用头)
6. [各层类型的序列化细节](#6-各层类型的序列化细节)
7. [WeightMatrix 二进制格式](#7-weightmatrix-二进制格式)
8. [LSTM 单元结构](#8-lstm-单元结构)
9. [网络描述语言（Spec Language）](#9-网络描述语言spec-language)
10. [Unicharset 文本格式](#10-unicharset-文本格式)
11. [Recoder（UnicharCompress）格式](#11-recoderunicharcompress-格式)
12. [DAWG（有向无环词图）格式](#12-dawg有向无环词图格式)
13. [LSTM 训练数据格式（.lstmf）](#13-lstm-训练数据格式lstmf)
14. [示例：eng.traineddata 完整解析](#14-示例engtraineddata-完整解析)

---

## 1. 容器格式

`.traineddata` 是一个简单的二进制容器，包含最多 24 个组件（component）。文件使用小端序（little-endian）。

### 文件布局

```
┌─────────────────────────────────────┐
│  uint32_t  num_entries (固定为 24)   │  4 bytes
├─────────────────────────────────────┤
│  int64_t   offset_table[24]         │  192 bytes
│  (每项为组件起始偏移量, -1=不存在)   │
├─────────────────────────────────────┤
│  组件 0 的数据                       │
│  组件 1 的数据                       │
│  ...                                │
│  组件 23 的数据                      │
└─────────────────────────────────────┘
```

### 头部细节

| 字段 | 类型 | 大小 | 说明 |
|------|------|------|------|
| `num_entries` | uint32 | 4 字节 | 固定为 24（`TESSDATA_NUM_ENTRIES`）。若值 > 1000 则视为大端序存储 |
| `offset_table[0..23]` | int64 × 24 | 192 字节 | 各组件的字节偏移量。-1 表示该组件不存在 |
| **头部总大小** | | **196 字节** | |

### 组件大小计算

组件 `i` 的大小 = 下一个存在组件的偏移量 - 当前组件偏移量。若 `i` 是最后一个存在的组件，则到文件末尾。

### 大端序检测

C++ 源码中，若 `num_entries > kMaxNumEntries(1000)`，则认为文件是大端序存储，对所有 `int64` 做字节交换。

> **参考**：`src/ccutil/tessdatamanager.cpp`，`src/tesseract_cuda/formats/tessdata.py`

---

## 2. 组件类型列表

共 24 个槽位（index 0-23）：

| 索引 | 枚举名 | 文件后缀 | 说明 |
|------|--------|----------|------|
| 0 | `TESSDATA_LANG_CONFIG` | `config` | 语言配置（文本） |
| 1 | `TESSDATA_UNICHARSET` | `unicharset` | 字符集（文本，用于传统引擎） |
| 2 | `TESSDATA_AMBIGS` | `unicharambigs` | 字符歧义映射（文本） |
| 3 | `TESSDATA_INTTEMP` | `inttemp` | 整数字符模板（传统分类器） |
| 4 | `TESSDATA_PFFMTABLE` | `pffmtable` | 原型特征频率表（传统） |
| 5 | `TESSDATA_NORMPROTO` | `normproto` | 归一化原型（传统） |
| 6 | `TESSDATA_PUNC_DAWG` | `punc-dawg` | 标点 DAWG（传统） |
| 7 | `TESSDATA_SYSTEM_DAWG` | `word-dawg` | 系统词典 DAWG（传统） |
| 8 | `TESSDATA_NUMBER_DAWG` | `number-dawg` | 数字模式 DAWG（传统） |
| 9 | `TESSDATA_FREQ_DAWG` | `freq-dawg` | 高频词 DAWG（传统） |
| 10 | `TESSDATA_FIXED_LENGTH_DAWGS` | (已废弃) | 固定长度 DAWG |
| 11 | `TESSDATA_CUBE_UNICHARSET` | (已废弃) | Cube 字符集 |
| 12 | `TESSDATA_CUBE_SYSTEM_DAWG` | (已废弃) | Cube 词典 |
| 13 | `TESSDATA_SHAPE_TABLE` | `shapetable` | 形状表（自适应分类器） |
| 14 | `TESSDATA_BIGRAM_DAWG` | `bigram-dawg` | 双字组 DAWG |
| 15 | `TESSDATA_UNAMBIG_DAWG` | `unambig-dawg` | 无歧义词 DAWG |
| 16 | `TESSDATA_PARAMS_MODEL` | `params-model` | 参数模型 |
| **17** | **`TESSDATA_LSTM`** | **`lstm`** | **LSTM 神经网络（二进制）** |
| 18 | `TESSDATA_LSTM_PUNC_DAWG` | `lstm-punc-dawg` | LSTM 标点 DAWG |
| 19 | `TESSDATA_LSTM_SYSTEM_DAWG` | `lstm-word-dawg` | LSTM 系统词典 DAWG |
| 20 | `TESSDATA_LSTM_NUMBER_DAWG` | `lstm-number-dawg` | LSTM 数字 DAWG |
| 21 | `TESSDATA_LSTM_UNICHARSET` | `lstm-unicharset` | LSTM 字符集（文本） |
| 22 | `TESSDATA_LSTM_RECODER` | `lstm-recoder` | 字符重编码器（二进制） |
| 23 | `TESSDATA_VERSION` | `version` | 版本字符串（原始 UTF-8 文本，无长度前缀） |

### 可用性检测

- `IsBaseAvailable()` = 存在组件 1 + 3（unicharset + inttemp）
- `IsLSTMAvailable()` = 存在组件 17（lstm）

---

## 3. 二进制序列化基础（TFile）

所有二进制数据使用 TFile 格式序列化，统一小端序。

### 基本类型

| 类型 | 大小 | Python struct | 说明 |
|------|------|---------------|------|
| int8 | 1 字节 | `<b` | 有符号字节 |
| uint8 | 1 字节 | `<B` | 无符号字节 |
| int32 | 4 字节 | `<i` | 32 位有符号整数 |
| uint32 | 4 字节 | `<I` | 32 位无符号整数 |
| int64 | 8 字节 | `<q` | 64 位有符号整数 |
| float | 4 字节 | `<f` | 32 位浮点 |
| double | 8 字节 | `<d` | 64 位双精度浮点 |

### 复合类型

| 类型 | 序列化方式 | 说明 |
|------|-----------|------|
| **string** | `uint32 length` + `length` 字节 UTF-8 | 无 null 终止符 |
| **bytes_vector** | `uint32 length` + `length` 字节原始数据 | 字节数组 |
| **int32_vector** | `uint32 count` + `count × int32` | 整数数组 |
| **GENERIC_2D_ARRAY\<T\>** | `uint32 dim1` + `uint32 dim2` + `T empty` + `T[dim1×dim2]` | 二维数组，行优先 |
| **pointer_list** | `uint32 count`，每项: `uint8 non_null`，若非空则反序列化该项 | 可空指针列表 |

> **参考**：`src/tesseract_cuda/formats/tfile.py`

---

## 4. LSTM 网络二进制格式

组件索引 17（`TESSDATA_LSTM`）包含完整的 LSTM 模型。

### 顶层结构

```
┌──────────────────────────────┐
│  network（递归序列化的网络树）  │  变长
├──────────────────────────────┤
│  network_str（网络描述字符串）  │  string
├──────────────────────────────┤
│  training_flags              │  int32
│  training_iteration          │  int32
│  sample_iteration            │  int32
│  null_char                   │  int32
│  adam_beta                   │  float
│  learning_rate               │  float
│  momentum                    │  float
└──────────────────────────────┘
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `network` | 递归 NetworkLayer | 完整网络树（见下文） |
| `network_str` | string | 网络描述字符串，如 `[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]` |
| `training_flags` | int32 | 训练标志 |
| `training_iteration` | int32 | 训练迭代次数 |
| `sample_iteration` | int32 | 样本迭代次数 |
| `null_char` | int32 | 空白字符 ID（CTC 的 blank token） |
| `adam_beta` | float | Adam 优化器 beta 参数 |
| `learning_rate` | float | 学习率 |
| `momentum` | float | 动量 |

> **参考**：`src/lstm/lstmrecognizer.cpp`（C++），`src/tesseract_cuda/formats/network_ser.py`（Python）

---

## 5. 网络节点通用头

每个网络节点以一个通用头开始，然后根据类型读取特定数据。

### 通用头格式

```
int8    type_id          类型 ID（0=新格式用字符串名称，非0=旧格式直接用数字）
string  type_name        [仅当 type_id==0] 类型名称字符串
int8    training         训练状态：0=禁用, 1=启用
int8    needs_backprop   是否需要反向传播：0=false, 1=true
int32   network_flags    网络标志位
int32   ni               输入维度
int32   no               输出维度
int32   num_weights      总权重数
string  name             层名称
```

### 网络类型名称表

| 索引 | 枚举名 | 字符串名称 | 类别 |
|------|--------|-----------|------|
| 0 | `NT_NONE` | `"Invalid"` | 哨兵 |
| 1 | `NT_INPUT` | `"Input"` | 输入 |
| 2 | `NT_CONVOLVE` | `"Convolve"` | 卷积 |
| 3 | `NT_MAXPOOL` | `"Maxpool"` | 池化 |
| 4 | `NT_PARALLEL` | `"Parallel"` | 管道 |
| 5 | `NT_REPLICATED` | `"Replicated"` | 管道 |
| 6 | `NT_PAR_RL_LSTM` | `"ParBidiLSTM"` | 管道 |
| 7 | `NT_PAR_UD_LSTM` | `"DepParUDLSTM"` | 管道 |
| 8 | `NT_PAR_2D_LSTM` | `"Par2dLSTM"` | 管道 |
| 9 | `NT_SERIES` | `"Series"` | 管道 |
| 10 | `NT_RECONFIG` | `"Reconfig"` | 空间 |
| 11 | `NT_XREVERSED` | `"RTLReversed"` | 管道 |
| 12 | `NT_YREVERSED` | `"TTBReversed"` | 管道 |
| 13 | `NT_XYTRANSPOSE` | `"XYTranspose"` | 管道 |
| 14 | `NT_LSTM` | `"LSTM"` | LSTM |
| 15 | `NT_LSTM_SUMMARY` | `"SummLSTM"` | LSTM |
| 16 | `NT_LOGISTIC` | `"Logistic"` | 全连接 |
| 17 | `NT_POSCLIP` | `"LinLogistic"` | 全连接 |
| 18 | `NT_SYMCLIP` | `"LinTanh"` | 全连接 |
| 19 | `NT_TANH` | `"Tanh"` | 全连接 |
| 20 | `NT_RELU` | `"Relu"` | 全连接 |
| 21 | `NT_LINEAR` | `"Linear"` | 全连接 |
| 22 | `NT_SOFTMAX` | `"Softmax"` | 输出 |
| 23 | `NT_SOFTMAX_NO_CTC` | `"SoftmaxNoCTC"` | 输出 |
| 24 | `NT_LSTM_SOFTMAX` | `"LSTMSoftmax"` | LSTM 变体 |
| 25 | `NT_LSTM_SOFTMAX_ENCODED` | `"LSTMBinarySoftmax"` | LSTM 变体 |
| 26 | `NT_TENSORFLOW` | `"TensorFlow"` | 外部 |

### NetworkFlags 标志位

| 标志 | 值 | 说明 |
|------|-----|------|
| `NF_LAYER_SPECIFIC_LR` | 64 | 每层使用不同学习率 |
| `NF_ADAM` | 128 | 使用 Adam 优化器 |

### TrainingState

| 状态 | 值 | 说明 |
|------|-----|------|
| `TS_DISABLED` | 0 | 永久禁用 |
| `TS_ENABLED` | 1 | 训练激活 |
| `TS_TEMP_DISABLE` | 2 | 临时禁用 |
| `TS_RE_ENABLE` | 3 | 从临时禁用恢复 |

---

## 6. 各层类型的序列化细节

通用头之后，每种层类型有各自的数据。

### 6.1 Input 层

```
int32  batch       批次大小（通常为 1）
int32  height      图像高度
int32  width       图像宽度（通常为 0，表示可变）
int32  depth       特征深度（通常为 1）
int32  loss_type   损失函数类型
```

**LossType 枚举**：

| 值 | 名称 | 说明 |
|----|------|------|
| 0 | `LT_NONE` | 无损失 |
| 1 | `LT_CTC` | CTC 损失 |
| 2 | `LT_SOFTMAX` | Softmax 损失 |
| 3 | `LT_LOGISTIC` | Logistic 损失 |

**关键注意**：Input 层的 `ni`（在通用头中）= `height`（不是 depth）。`no` = `depth`。当 height > 1 时，Input 层将 height 维"折叠"进 ni，即将每列像素视为一个输入向量。

### 6.2 Plumbing 层（Series, Parallel, Replicated, ParBidiLSTM 等）

包括：`Series`, `Parallel`, `Replicated`, `ParBidiLSTM`, `DepParUDLSTM`, `Par2dLSTM`, `RTLReversed`, `TTBReversed`, `XYTranspose`

```
uint32              stack_size          子网络数量
NetworkLayer[stack_size]  children     递归序列化的子网络
[float[stack_size]]       learning_rates  [仅当 network_flags & 64]
```

### 6.3 LSTM 层（LSTM, SummLSTM, LSTMSoftmax, LSTMBinarySoftmax）

```
int32           na              填充输入大小 = ni + ns [+ ns（2D时）] + nf
WeightMatrix    gate_ci         Cell Input 门（tanh 激活）
WeightMatrix    gate_gi         Input Gate（sigmoid 激活）
WeightMatrix    gate_gf1        Forget Gate 1（sigmoid 激活）
WeightMatrix    gate_go         Output Gate（sigmoid 激活）
[WeightMatrix   gate_gfs]       Forget Gate Spatial（仅 2D LSTM）
[NetworkLayer   softmax]       Softmax 子网络（仅 LSTMSoftmax/LSTMBinarySoftmax）
```

**关键计算值**：

- `ns`（隐藏状态数）= `gate_ci.NumOutputs()`
- `is_2d` = `(na - nf) == ni + 2 * ns`
- `nf` 计算：
  - `LSTMSoftmax`: `nf = no`
  - `LSTMBinarySoftmax`: `nf = ceil(log2(no))`
  - 其他: `nf = 0`
- `na` = `ni + ns`（1D）或 `ni + 2*ns`（2D）+ `nf`

**每个门的 WeightMatrix 形状**：`[ns, na+1]`（最后一列是偏置）

### 6.4 Convolve 层

```
int32  half_x    卷积核半宽（全宽 = 2*half_x+1）
int32  half_y    卷积核半高（全高 = 2*half_y+1）
```

**输出维度**：`no = ni × (2×half_x+1) × (2×half_y+1)`

**注意**：Tesseract 的 Convolve 层是无参数的 patch 提取操作（im2col）。实际的卷积权重存储在紧跟其后的激活层（Tanh/Relu/Logistic）中。在我们的 PyTorch 实现中，`ConvolveLayer` 将两者合并为一个带权重的模块。

### 6.5 Maxpool 层

```
int32  x_scale    水平缩放因子
int32  y_scale    垂直缩放因子
```

输出维度 `no = ni`（深度不变，空间尺寸缩小）。

### 6.6 Reconfig 层

```
int32  x_scale    水平重排因子
int32  y_scale    垂直重排因子
```

输出维度 `no = ni × x_scale × y_scale`（空间转深度）。

### 6.7 全连接/激活层（Tanh, Relu, Logistic, Linear, Softmax 等）

```
WeightMatrix  weights    单个权重矩阵 [no, ni+1]
```

包括：`Softmax`, `SoftmaxNoCTC`, `Logistic`, `LinLogistic`, `LinTanh`, `Tanh`, `Relu`, `Linear`

---

## 7. WeightMatrix 二进制格式

### 格式标记

```
uint8  mode    位标志组合
```

| 标志 | 位 | 值 | 说明 |
|------|-----|-----|------|
| `kInt8Flag` | bit 0 | 1 | Int8 量化权重 |
| `kAdamFlag` | bit 2 | 4 | 包含 Adam 优化器数据 |
| `kDoubleFlag` | bit 7 | 128 | 双精度格式（vs 单精度） |

### 7.1 浮点模式（mode & kInt8Flag == 0）

```
GENERIC_2D_ARRAY<double>  weights     uint32 dim1, uint32 dim2, double empty, double[dim1×dim2]
[GENERIC_2D_ARRAY<double> updates_]   [仅训练模式] 同形状
[GENERIC_2D_ARRAY<double> dw_sq_sum_] [仅训练模式 & Adam] 同形状
```

`GENERIC_2D_ARRAY<double>` 详细布局：

```
uint32    dim1           行数（输出维度）
uint32    dim2           列数（输入维度 + 1，最后一列为偏置）
double    empty          始终 0.0（填充值）
double[dim1×dim2]        行优先权重数据
```

**权重矩阵布局**：

```
         input_0  input_1  ...  input_{ni-1}  bias
out_0    w[0,0]   w[0,1]   ...  w[0,ni-1]     w[0,ni]
out_1    w[1,0]   w[1,1]   ...  w[1,ni-1]     w[1,ni]
...      ...      ...      ...  ...           ...
out_{no-1} w[no-1,0] ...         ...           w[no-1,ni]
```

即：`dim1 = no`（输出数），`dim2 = ni + 1`（输入数 + 1 偏置列）。

### 7.2 Int8 量化模式（mode & kInt8Flag != 0）

```
GENERIC_2D_ARRAY<int8>  wi_       int32 dim1, int32 dim2, int8 empty, int8[dim1×dim2]
uint32                  num_scales 缩放因子数量
double/float[num_scales]  scales  每行一个缩放因子（double 若 kDoubleFlag，否则 float）
```

**权重还原公式**：

```python
weight_float = (int8_val / 127.0) * scale
```

其中 `scale` = `scales[row]`（对应输出行）。

### 7.3 我们导出时使用的格式

导出时始终使用**双精度浮点模式**（`mode = 0x80`，即 `kDoubleFlag`），不包含训练数据（`training=False`）。序列化为：

```python
mode = 128  # WM_DOUBLE
GENERIC_2D_ARRAY<double> with weights only
```

> **参考**：`src/lstm/weightmatrix.cpp`（C++），`src/tesseract_cuda/formats/network_ser.py`（Python）

---

## 8. LSTM 单元结构

### 门方程

Tesseract 使用 5 门 LSTM（第 5 门仅 2D 时使用）：

```
CI  = tanh(W_ci × source)           # Cell Input
GI  = sigmoid(W_gi × source)        # Input Gate
GF1 = sigmoid(W_gf1 × source)       # Forget Gate 1（时间方向）
GO  = sigmoid(W_go × source)        # Output Gate
GFS = sigmoid(W_gfs × source)       # Forget Gate Spatial（仅 2D，空间方向）

state = GF1 ⊙ state_prev + CI ⊙ GI  (+ GFS ⊙ state_y, 如果 2D)
state = clamp(state, -100, 100)
output = tanh(state) ⊙ GO
```

### 输入向量构造

```
source = [input(t), prev_output, (prev_output_y, 如果 2D), (softmax_feedback, 如果有)]
```

总长度 `na` = `ni + ns + ns? + nf?`

### 门权重矩阵维度

| 门 | 行（输出）| 列（输入 + 1 偏置）|
|----|----------|-------------------|
| CI (gate_ci) | ns | na + 1 |
| GI (gate_gi) | ns | na + 1 |
| GF1 (gate_gf1) | ns | na + 1 |
| GO (gate_go) | ns | na + 1 |
| GFS (gate_gfs) | ns | na + 1 |

其中：
- `ns` = 隐藏状态数
- `na` = `ni + ns`（1D）或 `ni + 2*ns`（2D）+ `nf`
- `+1` 是偏置列

### 权重布局示例

以一个 `ni=48, ns=64` 的 1D LSTM 为例：

```
na = 48 + 64 = 112
每个门: [64, 113] = 7232 个 double
4 个门总计: 4 × 7232 = 28928 个 double = 231424 字节
```

> **参考**：`src/lstm/lstm.cpp`（C++），`src/tesseract_cuda/network/lstm_cell.py`（Python）

---

## 9. 网络描述语言（Spec Language）

网络描述字符串定义了网络的结构，使用紧凑的文本语法。

### 语法规则

```
spec       := '[' series_content ']' | layer
series_content := (input_spec layer*) | layer+
input_spec := batch ',' height ',' width ',' depth
```

### 层类型语法

| 语法 | 示例 | 说明 |
|------|------|------|
| `C<act><y>,<x>,<d>` | `Ct3,3,16` | Convolve: 激活函数(核y,核x) → 深度 d |
| `Mp<y>,<x>` | `Mp3,3` | Maxpool: 缩小 (y,x) |
| `S<y>,<x>` | `S2,4` | Reconfig: 空间转深度 (y,x) |
| `L<dir><dim>[s]<n>` | `Lfx64` / `Lfys48` / `Lrx96` | LSTM: 方向,维度,[summary],状态数 |
| `F<act><n>` | `Ft64` | 全连接: 激活函数,输出数 |
| `O<dims><type><n>` | `O1c1` | 输出: 维度,损失类型,输出数 |
| `[...]` | `[...]` | Series（顺序连接） |
| `(...)` | `(...)` | Parallel（并行） |

### 激活函数编码

| 代码 | 类型 |
|------|------|
| `s` | sigmoid / Logistic |
| `t` | tanh |
| `r` | ReLU |
| `l` | Linear |
| `m` | Softmax |
| `p` | PosClip (LinLogistic) |
| `n` | SymClip (LinTanh) |

### LSTM 方向/维度编码

| 代码 | 含义 |
|------|------|
| `f` | Forward（前向） |
| `r` | Reverse（反向） |
| `b` | Bidirectional（双向） |
| `x` | 沿 x 轴（宽度方向） |
| `y` | 沿 y 轴（高度方向） |
| `s` | Summary（仅在序列末端输出） |
| `S` | 带 Softmax 反馈的 LSTM |
| `E` | 带 Binary-Encoded Softmax 反馈的 LSTM |

### 输出编码

| 代码 | 含义 |
|------|------|
| `0` | 0-d 输出（分类，折叠所有空间维度） |
| `1` | 1-d 输出（序列，y 维折叠进深度） |
| `2` | 2-d 输出（逐位置，保留空间） |
| `c` | CTC 损失 |
| `s` | Softmax 损失 |
| `l` | Logistic 损失 |

### 完整示例解析

**描述字符串**：`[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]`

```
[                               ← Series 开始
  1,36,0,1                      ← Input: batch=1, height=36, width=0, depth=1
  Ct3,3,16                      ← Convolve: tanh, kernel 3×3, depth 16
  Mp3,3                         ← Maxpool: 3×3
  Lfys64                        ← LSTM: forward, y-dimension, summary, 64 states
  Lfx96                         ← LSTM: forward, x-dimension, 96 states
  Lrx96                         ← LSTM: reverse, x-dimension, 96 states
  Lfx512                        ← LSTM: forward, x-dimension, 512 states
  O1c1                          ← Output: 1-d sequence, CTC loss, 1 output
]                               ← Series 结束
```

**注意**：`O1c1` 中的 `1` 是占位符，实际输出类别数在运行时由 unicharset 大小决定。

### 各层维度变化

以 `eng.traineddata`（unicharset 112 字符）为例：

| 层 | 操作 | ni | no | 说明 |
|----|------|----|----|------|
| Input | — | 36 | 1 | 36 像素高的列，1 通道灰度 |
| Convolve | im2col | 1 | 9 | 1×(3×3) = 9 patches |
| Tanh | FC 9→16 | 9 | 16 | 卷积权重在此层 |
| Maxpool | 3×3 | 16 | 16 | 空间缩小 3 倍 |
| SummLSTM | forward+y+summary | 16 | 64 | 垂直方向摘要 |
| LSTM | forward+x | 64 | 96 | 前向水平扫描 |
| LSTM | reverse+x | 64 | 96 | 反向水平扫描 |
| LSTM | forward+x | 192 | 512 | 前向（双向拼接后 96+96=192） |
| Softmax | FC 512→112 | 512 | 112 | 输出 112 个字符类别 |

> **参考**：`src/training/common/networkbuilder.cpp`（C++），`src/tesseract_cuda/network/spec_parser.py`（Python）

---

## 10. Unicharset 文本格式

组件索引 21（`TESSDATA_LSTM_UNICHARSET`）为文本格式。

### 文件结构

```
<count>
<char_entry_0>
<char_entry_1>
...
<char_entry_{count-1}>
```

### 字符条目格式（完整）

```
<unichar> <properties_hex> <min_bot>,<max_bot>,<min_top>,<max_top>,<width>,<width_sd>,<bearing>,<bearing_sd>,<advance>,<advance_sd> <script> <other_case> <direction> <mirror> <normed>
```

**示例**：

```
标 1 0,255,0,255,0,0,0,0,0,0 Han 68 0 68 标
```

### 空格字符

空格字符使用特殊标记 `NULL`：

```
NULL <properties_hex> <bbox_fields> <script> <other_case> ...
```

### 属性位掩码

| 属性 | 说明 |
|------|------|
| ISALPHA_MASK | 字母 |
| ISLOWER_MASK | 小写 |
| ISUPPER_MASK | 大写 |
| ISDIGIT_MASK | 数字 |
| ISPUNCTUATION_MASK | 标点 |

### 解析兼容级别

解析器支持从完整到简化的多种格式：

| 级别 | 格式 |
|------|------|
| 0 | 完整 10 字段 bbox + script + other_case + direction + mirror + normed |
| 1 | 完整 bbox + script + other_case + direction + mirror |
| 2 | 短 bbox（3 逗号）+ script + other_case + direction + mirror |
| 3 | 短 bbox + script + other_case |
| 4 | script + other_case（无 bbox） |
| 5 | 仅 script |

> **参考**：`src/ccutil/unicharset.cpp`（C++），`src/tesseract_cuda/formats/unicharset.py`（Python）

---

## 11. Recoder（UnicharCompress）格式

组件索引 22（`TESSDATA_LSTM_RECODER`）为二进制格式。用于压缩大字符集。

### 二进制布局

```
uint32    count                    编码表条目数
For each entry:
  int8    self_normalized          1 = 自归一化编码
  int32   length                   编码序列长度
  int32[length]  codes             重编码值序列
```

### 用途

- **CJK 语言**：韩文用 Jamo 分解（每字符 3 个编码），汉字用部首-笔画索引
- **印度语言**：Unicode 分解为字素片段
- **其他**：连字分解（如 "fi" → "f","i"），相似形状合并

---

## 12. DAWG（有向无环词图）格式

用于存储词典（组件 18/19/20）。

### DAWG 类型

| 枚举 | 值 | 说明 |
|------|-----|------|
| `DAWG_TYPE_PUNCTUATION` | 0 | 标点模式 |
| `DAWG_TYPE_WORD` | 1 | 词典词 |
| `DAWG_TYPE_NUMBER` | 2 | 数字模式 |
| `DAWG_TYPE_PATTERN` | 3 | 通用模式 |

### 二进制布局（SquishedDawg）

```
int16     magic_number            固定 42，用于字节序检测
int32     unicharset_size         字符集大小
int32     num_edges               前向边数量
uint64[num_edges]  edges          边记录数组
```

### EDGE_RECORD 格式（uint64，64 位）

位域根据 `unicharset_size` 动态计算：

```
Bits [0 .. flag_start-1]           = UNICHAR_ID（字符 ID）
Bits [flag_start .. flag_start+2]  = 标志位（3 bits）
Bits [next_node_start .. 63]       = 下一个节点引用

标志位:
  bit 0: MARKER_FLAG    = 该节点边列表的最后一条边
  bit 1: DIRECTION_FLAG = 0=前向, 1=后向
  bit 2: WERD_END_FLAG  = 边标记词的结束
```

### 位掩码计算

```python
bits_for_unichar = ceil(log2(unicharset_size))
flag_start = bits_for_unichar
next_node_start = flag_start + 3  # 3 flag bits
```

> **参考**：`src/dict/dawg.h`（C++）

---

## 13. LSTM 训练数据格式（.lstmf）

### 二进制布局

```
uint32     total_pages              页面总数
For each page:
  uint8      non_null               0=空, 1=存在
  [if non_null]:
    string   imagefilename          源文件名
    int32    page_number            页码
    bytes    image_data             原始图像数据（PNG/TIFF）
    string   language               语言代码
    string   transcription          标注文本
    uint32   box_count              边界框数量
    int32×4×box_count  boxes        每框: x_min, y_min, x_max, y_max
    uint32   text_count             标签文本数量
    string[text_count] box_texts    每框对应的文本标签
    int8     vertical_text          0=水平, 1=垂直
```

> **参考**：`src/tesseract_cuda/formats/lstmf.py`（Python）

---

## 14. 示例：eng.traineddata 完整解析

### 组件列表

```
索引  名称               大小         说明
──────────────────────────────────────────────────────
17    lstm               11,689,032   LSTM 网络权重
18    lstm-punc-dawg     432          标点词典
19    lstm-word-dawg     3,694,794    系统词典
20    lstm-number-dawg   4,738        数字模式词典
21    lstm-unicharset    6,360        112 个字符
22    lstm-recoder       1,012        字符压缩编码
23    version            80           版本字符串
```

### 网络结构树

```
Series [ni=36, no=112]
├── Input [ni=36, no=1]
│   └── shape: batch=1, height=36, width=0, depth=1, loss=CTC
├── Series [ni=1, no=16]                    ← Convolve+Tanh 展开
│   ├── Convolve [ni=1, no=9]
│   │   └── half_x=1, half_y=1 (kernel 3×3)
│   └── Tanh [ni=9, no=16]
│       └── WeightMatrix [16, 10]
├── Maxpool [ni=16, no=16]
│   └── x_scale=3, y_scale=3
├── SummLSTM [ni=16, no=64]
│   └── 4 gates × WeightMatrix [64, 81]    (na=16+64=80, +1 bias=81)
├── LSTM [ni=64, no=96]
│   └── 4 gates × WeightMatrix [96, 161]   (na=64+96=160, +1 bias=161)
├── LSTM [ni=64, no=96]
│   └── 4 gates × WeightMatrix [96, 161]
├── LSTM [ni=192, no=512]
│   └── 4 gates × WeightMatrix [512, 705]  (na=192+512=704, +1 bias=705)
└── Softmax [ni=512, no=112]
    └── WeightMatrix [112, 513]            (512+1 bias=513)
```

### 关键数字

- **总参数量** ≈ 1,970,000（约 2M 权重）
- **LSTM 组件大小** ≈ 11.7 MB（以 double 精度存储）
- **字符集**：112 个字符（包括字母、数字、标点、空格等）
- **空白字符 ID**：null_char 通常为字符集的最后一个索引

### 权重映射注意事项

在我们的 PyTorch 实现中：

1. **Convolve + Tanh 合并**：Tesseract 的 `Convolve`（无权重 patch 提取）和 `Tanh`（FC 权重）在代码中合并为单个 `ConvolveLayer`。加载时跳过 Convolve 权重，只加载 Tanh 权重。导出时拆分为 `Series(Convolve, Tanh)`。

2. **Input 层的 ni**：`ni = height`（36），`no = depth`（1）。这是因为 Input 层将每列像素展平为向量。

3. **双向 LSTM 拼接**：前向 LSTM（Lfx96）和反向 LSTM（Lrx96）的输出在 Series 中自动拼接，所以下一个 LSTM 的输入是 `96 + 96 = 192`。

4. **Summary LSTM**：`Lfys64` 是沿 y 方向的 Summary LSTM，只在序列末端输出（将垂直方向的 16 维输入压缩为 64 维）。

---

## 附录 A：本项目的 Python 实现

| 文件 | 说明 |
|------|------|
| `src/tesseract_cuda/formats/tfile.py` | TFile 二进制读写器 |
| `src/tesseract_cuda/formats/tessdata.py` | .traineddata 容器管理 |
| `src/tesseract_cuda/formats/network_ser.py` | 网络序列化/反序列化 |
| `src/tesseract_cuda/formats/unicharset.py` | Unicharset 解析 |
| `src/tesseract_cuda/formats/recoder.py` | Recoder 解析 |
| `src/tesseract_cuda/formats/lstmf.py` | .lstmf 训练数据格式 |
| `src/tesseract_cuda/network/spec_parser.py` | 网络描述语言解析器 |
| `src/tesseract_cuda/network/weight_mapper.py` | PyTorch ↔ Tesseract 权重映射 |
| `src/tesseract_cuda/network/model.py` | PyTorch 模型定义 |

## 附录 B：C++ 源码参考

| 文件 | 说明 |
|------|------|
| `src/ccutil/tessdatamanager.h/cpp` | .traineddata 容器 |
| `src/ccutil/serialis.h` | 二进制序列化原语 |
| `src/lstm/lstmrecognizer.cpp` | LSTM 识别器（顶层序列化） |
| `src/lstm/network.cpp` | 网络基类（节点序列化） |
| `src/lstm/lstm.cpp` | LSTM 层 |
| `src/lstm/plumbing.cpp` | Series/Parallel 等管道层 |
| `src/lstm/weightmatrix.cpp` | 权重矩阵 |
| `src/lstm/convolve.cpp` | 卷积层 |
| `src/lstm/maxpool.cpp` | 池化层 |
| `src/lstm/input.cpp` | 输入层 |
| `src/lstm/fullyconnected.cpp` | 全连接层 |
| `src/training/common/networkbuilder.cpp` | 网络构建器（描述语言） |
| `src/ccutil/unicharset.cpp` | 字符集 |
| `src/ccutil/unicharcompress.cpp` | 字符压缩 |
| `src/dict/dawg.h/cpp` | DAWG 词典 |
