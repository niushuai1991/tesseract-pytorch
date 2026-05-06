# Tesseract CUDA 训练项目实施计划

## Context

Tesseract OCR 的 LSTM 训练目前使用自研 C++ 框架，纯 CPU 运行，训练一个完整模型需要数天到数周。目标是用 Python + PyTorch 重写训练流程，支持 GPU 加速，同时保持与 Tesseract 现有 `traineddata` 格式的完全兼容，训练出的模型可以直接被 Tesseract 加载使用。

项目放在 `/home/ns/code/training-tesseract/tesseract-pytorch/` 目录，使用 UV 管理。

## 项目结构

```
tesseract-pytorch/
  pyproject.toml
  src/
    tesseract_cuda/
      __init__.py
      cli.py                        # CLI 入口
      formats/
        __init__.py
        tfile.py                    # TFile 二进制读写原语
        tessdata.py                 # traineddata 容器读写
        network_ser.py              # 网络权重序列化/反序列化
        lstmf.py                    # .lstmf 训练数据读取
        unicharset.py               # unicharset 解析
        recoder.py                  # UnicharCompress (recoder) 读写
      network/
        __init__.py
        spec_parser.py              # 网络规格语言解析器
        lstm_cell.py                # 5门 LSTM 单元（兼容 Tesseract）
        layers.py                   # PyTorch 层实现
        model.py                    # 顶层 TessLSTMModel
        weight_mapper.py            # PyTorch ↔ Tesseract 权重映射
      training/
        __init__.py
        dataset.py                  # .lstmf → PyTorch Dataset
        trainer.py                  # 训练循环（GPU）
      export/
        __init__.py
        exporter.py                 # 导出为 traineddata
```

## 实施步骤（按顺序）

### Phase 1: 项目骨架 + 二进制格式层

**1.1 UV 项目初始化**
- 创建 `tesseract-pytorch/` 目录
- `pyproject.toml`: python>=3.10, 依赖 torch, numpy, Pillow
- `src/tesseract_cuda/` 包结构

**1.2 `formats/tfile.py` — TFile 读写原语**
- `TFileReader`: 从 bytes 读取 int8/uint8/int32/uint32/int64/float/double/string/bytes/2d_array
- `TFileWriter`: 对应的写入方法
- 所有序列化使用小端序（struct `<`）
- 关键格式: string = uint32 len + bytes; vector<T> = uint32 count + T[]; GENERIC_2D_ARRAY<double> = uint32 dim1 + uint32 dim2 + double empty + double[dim1*dim2]
- 参考源文件: `src/ccutil/serialis.h`, `src/ccutil/serialis.cpp`

**1.3 `formats/tessdata.py` — traineddata 容器**
- 读取: uint32 num_entries(24) + int64[24] offset_table + 各组件数据
- 写入: 计算偏移表后序列化
- 组件索引: LSTM=17, LSTM_UNICHARSET=21, LSTM_RECODER=22, VERSION=23
- 大端检测: num_entries > 1000 则字节交换
- 参考源文件: `src/ccutil/tessdatamanager.cpp:110-154`(读取), `177-200`(写入)

**1.4 `formats/unicharset.py` — unicharset 解析**
- 文本格式: 第一行是字符数，之后每行一个字符条目
- 需要: id_to_unichar(id), unichar_to_id(string), size
- 参考源文件: `src/ccutil/unicharset.cpp`

**1.5 `formats/recoder.py` — recoder 读写**
- 二进制格式: int32 num_codes, 每 code: int8 self_normalized + int32 length + int32[length] codes
- 方法: encode(unichar_id) -> list[int], decode(codes) -> unichar_id, num_codes
- 参考源文件: `src/ccutil/unicharcompress.cpp`

**1.6 `formats/network_ser.py` — 网络序列化**
- 读取 LSTM 组件: network(递归) + network_str + training_flags + iterations + null_char + adam_beta + lr + momentum
- 网络层递归读取: int8 type(0=NT_NONE) + string type_name + int8 training_state + bool needs_backprop + int32 flags + int32 ni + int32 no + int32 num_weights + string name + 层数据
- LSTM 层数据: int32 na + 4-5 个 WeightMatrix (CI, GI, GF1, GO, [GFS])
- WeightMatrix: uint8 mode(128=double) + GENERIC_2D_ARRAY<double> wf + [training时: updates + adam_sq_sum]
- Series/Parallel: uint32 stack_size + 子层数组
- 参考源文件: `src/lstm/network.cpp`(Serialize/DeSerialize), `src/lstm/lstm.cpp`(LSTM::Serialize/DeSerialize), `src/lstm/weightmatrix.cpp`(WeightMatrix::Serialize/DeSerialize), `src/lstm/lstmrecognizer.cpp:93-130`(顶层序列化顺序)

**验证方式**: 用现有 eng.traineddata 读取 → 重新写入 → 二进制对比完全一致

### Phase 2: 网络构建

**2.1 `network/spec_parser.py` — 网络规格解析**
- 解析 Tesseract 的网络描述语言，如 `[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]`
- 支持: Input(b,h,w,d), Series[...], Parallel(...), Conv(Cx/y/d), Maxpool(Mp), LSTM(Lf/Lr/Lb/Lfys), FC(F), Output(O)
- 返回层描述列表，用于构建 PyTorch 模型
- 参考源文件: `src/training/common/networkbuilder.cpp`

**2.2 `network/lstm_cell.py` — 5门 LSTM**
- 5个独立的 nn.Linear: CI(tanh), GI(sigmoid), GF1(sigmoid), GO(sigmoid), GFS(可选,2D)
- source = [input, prev_output] 或 [input, prev_output_x, prev_output_y]
- 标准前向: CI=tanh(W*src), GI=sig(W*src), state = GF1*prev_state + CI*GI, output = tanh(state)*GO
- 状态裁剪 [-100, 100]，梯度裁剪 [-1, 1]
- 参考源文件: `src/lstm/lstm.cpp:291-503`

**2.3 `network/layers.py` — 各层实现**
- ConvLayer: nn.Conv2d + 激活函数
- MaxPoolLayer: nn.MaxPool2d
- ReconfigLayer: reshape + concat blocks
- FCLayer: nn.Linear + 激活函数
- ReversedLayer: 翻转时间维度的包装器
- SeriesLayer: 顺序执行（类似 nn.Sequential）
- ParallelLayer: 并行执行 + depth 拼接
- SoftmaxCTC: nn.Linear + log_softmax

**2.4 `network/weight_mapper.py` — 权重映射**
- Tesseract WeightMatrix [no, ni+1] (最后1列是bias) → PyTorch nn.Linear weight[no,ni] + bias[no]
- 递归遍历网络树，按 Tesseract 序列化顺序映射每层权重
- 支持双向: 从 traineddata 加载权重到 PyTorch，从 PyTorch 导出权重到 Tesseract 格式

**2.5 `network/model.py` — TessLSTMModel**
- 组合 spec_parser + layers + lstm_cell 构建完整模型
- forward(image_tensor) → output_logits
- load_from_traineddata(path): 读权重映射到 PyTorch
- 参考源文件: `src/lstm/lstmrecognizer.cpp`

**验证方式**: 从已有 traineddata 加载权重 → 同一输入 → 对比 Tesseract C++ 的输出 logits

### Phase 3: 训练管线

**3.1 `formats/lstmf.py` — 训练数据读取**
- 格式: uint32 total_pages + 每页 ImageData (imagefilename, page_number, image_data(PNG), language, transcription, boxes, box_texts, vertical_text)
- 参考源文件: `src/ccstruct/imagedata.cpp:92-116`(Serialize), `src/ccstruct/imagedata.cpp`(DeSerialize)

**3.2 `training/dataset.py` — PyTorch Dataset**
- 从 .lstmf 文件读取图像+转录文本
- 图像预处理: 缩放到目标高度(如36像素)、灰度、归一化到 [-0.5, 0.5]
- 文本编码: transcription → unicharset IDs → recoder codes → label sequence
- 返回 (image_tensor, labels, input_lengths, target_lengths)
- 由于 Tesseract 逐行处理且宽度可变，初期以单图处理为主

**3.3 `training/trainer.py` — 训练循环**
- 使用 torch.nn.CTCLoss (blank=null_char)
- Adam 优化器 (beta1=0.5, beta2=0.999, lr=0.001)
- 梯度裁剪
- 定期保存 checkpoint (Tesseract 格式 + PyTorch state_dict)
- 训练指标: CTC loss, 字符错误率
- 支持从 checkpoint 恢复训练
- 支持从已有 traineddata fine-tune

### Phase 4: 导出 + CLI

**4.1 `export/exporter.py` — 导出 traineddata**
- 从 PyTorch 模型提取权重
- 通过 network_ser 写入网络二进制
- 通过 tessdata 替换 starter traineddata 中的 LSTM 组件
- 关键: 导出时 training_state=DISABLED(0), 只写权重不写 adam 状态
- 参考源文件: `src/training/lstmtraining.cpp`(stop_training 路径)

**4.2 `cli.py` — 命令行接口**
```bash
# 从头训练
tesseract-pytorch train --traineddata eng.traineddata --train-list list.txt \
  --net-spec "[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]" \
  --model-output output/checkpoint --gpu 0

# Fine-tune
tesseract-pytorch fine-tune --continue-from eng_best.traineddata \
  --train-list list.txt --model-output output/finetuned --gpu 0

# 导出
tesseract-pytorch export --checkpoint output/checkpoint.pt \
  --starter eng.traineddata --output eng_new.traineddata
```

## 关键参考源文件

| 文件 | 用途 |
|------|------|
| `src/ccutil/serialis.h/.cpp` | TFile 序列化原语 |
| `src/ccutil/tessdatamanager.cpp` | traineddata 容器格式 |
| `src/ccutil/unicharcompress.cpp` | recoder 序列化 |
| `src/ccstruct/imagedata.cpp` | .lstmf 格式 |
| `src/lstm/network.cpp` | 网络层序列化 |
| `src/lstm/lstm.cpp` | LSTM 前向/序列化 |
| `src/lstm/weightmatrix.cpp` | 权重矩阵格式 |
| `src/lstm/lstmrecognizer.cpp` | 顶层模型序列化顺序 |
| `src/training/common/networkbuilder.cpp` | 网络规格解析 |
| `src/training/lstmtraining.cpp` | 训练 CLI 参考 |

## 验证计划

1. **Phase 1 验证**: 读取 eng.traineddata → 重新序列化 → 二进制 diff 完全一致
2. **Phase 2 验证**: 从 traineddata 加载权重 → 相同输入 → 输出与 C++ Tesseract 在浮点精度内一致
3. **Phase 3 验证**: 小数据集训练 → loss 下降 → 导出
4. **Phase 4 验证**: 导出 traineddata → Tesseract 命令行加载识别 → 正常工作
