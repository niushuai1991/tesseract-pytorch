# 项目当前状态与下一步任务

## 已完成的工作

### 项目结构
```
tesseract-pytorch/
  pyproject.toml              # UV 项目配置 (CPU 版 PyTorch)
  src/tesseract_cuda/
    formats/
      tfile.py                # TFile 二进制读写原语
      tessdata.py             # traineddata 容器读写
      unicharset.py           # unicharset 文本解析
      recoder.py              # recoder (UnicharCompress) 读写
      network_ser.py          # 网络权重序列化/反序列化
      lstmf.py                # .lstmf 训练数据读取
    network/
      spec_parser.py          # 网络规格语言解析器
      lstm_cell.py            # 5门 LSTM 单元
      layers.py               # PyTorch 层实现
      weight_mapper.py        # PyTorch ↔ Tesseract 权重映射
      model.py                # 顶层 TessLSTMModel
    training/
      dataset.py              # .lstmf → PyTorch Dataset
      trainer.py              # 训练循环
    export/
      exporter.py             # 导出为 traineddata
    cli.py                    # CLI 入口
  tests/
    test_tfile.py             # 18 tests - 二进制原语 round-trip
    test_tessdata.py          # 9 tests - traineddata 容器
    test_network_ser.py       # 13 tests - 网络序列化 + 权重映射
    test_spec_and_model.py    # 13 tests - 规格解析 + 模型构建 + 前向传播
```

### 测试状态
- **53 个单元测试全部通过**
- 覆盖了：TFile 读写、traineddata 容器、网络序列化、权重映射、spec 解析、模型构建、LSTM cell
- Pyright 静态类型检查 0 错误

### 已验证的能力
- 读取 traineddata 文件，解析出所有 24 个组件
- 解析 unicharset（文本格式）和 recoder（二进制格式）
- 网络规格解析正确（如 `[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]`）
- 自建模型的序列化/反序列化 round-trip 正确
- 自建模型的导出/导入 round-trip 正确
- LSTM cell 前向传播正确（单步、序列、batch）

## 当前阻塞问题

### 反序列化真实 eng.traineddata 失败

用 `/tmp/eng.traineddata`（23MB）测试时，LSTM 组件的反序列化在读取 Series 的子层时报错：

```
IndexError: list index out of range (TYPE_NAMES)
```

**原因分析：** 读取子层时 `type_id` 超出范围，说明在读取第一个子层（Input）之后，字节偏移不对，导致第二个子层的 header 读到了错误位置。

**可能的具体原因：**
1. **Input 层的 StaticShape 字段**：Tesseract 的 Input 层序列化了 `StaticShape`（4个int32），反序列化时也读了4个int32。但 Input 层可能还有额外字段（如 `loss_type`，对应 `StaticShape` 实际有5个字段）
2. **needs_backprop 字段**：C++ 代码中 `needs_backprop` 作为 `int8` 读取但存为 bool，可能有对齐差异
3. **WeightMatrix 的 mode 字节**：真实的 traineddata 的 WeightMatrix mode 可能不是 `WM_DOUBLE(128)`，需要检查实际值

## 下一步任务

### 任务 1：修复真实 traineddata 的反序列化 [高优先级]

1. 逐字节对比读取 eng.traineddata 的 LSTM 组件，定位偏移出错的确切位置
2. 对照 C++ 源码 `src/lstm/input.cpp` 的 `Input::DeSerialize` 确认 Input 层序列化的完整字段
3. 对照 `src/lstm/network.cpp` 的 `Network::CreateFromFile` 确认 header 的每个字段大小
4. 修复 `network_ser.py` 中的反序列化逻辑
5. 验证：读取 eng.traineddata → 重新序列化 → 二进制 diff 完全一致

**关键参考文件：**
- `tesseract/src/lstm/network.cpp:191-313` — `CreateFromFile` 工厂方法
- `tesseract/src/lstm/input.cpp:45-47` — Input::DeSerialize
- `tesseract/src/lstm/lstm.cpp:253-287` — LSTM::DeSerialize
- `tesseract/src/lstm/weightmatrix.cpp:280-338` — WeightMatrix::DeSerialize
- `tesseract/src/lstm/lstmrecognizer.cpp:132-170` — 顶层 LSTM 组件读取顺序

### 任务 2：从真实模型加载权重到 PyTorch [依赖任务1]

1. 修复后从 eng.traineddata 加载 LSTM 网络到 PyTorch
2. 验证 PyTorch 模型参数数量与 C++ 模型一致
3. 对比关键权重值（如第一个 LSTM 层的 CI 门权重）

### 任务 3：前向传播输出对比 [依赖任务2]

1. 构造一个简单的测试图像输入
2. 分别用 C++ Tesseract 和 Python PyTorch 模型运行前向传播
3. 对比输出 logits，确认在浮点精度内一致
4. 不一致则排查 LSTM cell 的前向传播逻辑

### 任务 4：端到端训练 + 导出 + 识别验证 [依赖任务3]

1. 用 tesstrain 生成少量 .lstmf 训练数据
2. 用本项目 fine-tune eng 模型若干步
3. 导出为 traineddata
4. 用 `tesseract` 命令行加载导出的模型做识别
5. 确认识别结果正常

## 技术备注

- PyTorch 为 CPU 版（`torch==2.11.0+cpu`），因本地无 GPU
- Python 3.14.2
- 所有训练参数默认匹配 Tesseract（Adam, beta1=0.5, beta2=0.999, lr=0.001）
