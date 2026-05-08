# Tesseract PyTorch

A PyTorch reimplementation of [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)'s LSTM engine, compatible with Tesseract's `.traineddata` model format.

## Features

- **Load & run** existing Tesseract LSTM models (e.g. `eng.traineddata`) directly in PyTorch
- **Train** new models with standard PyTorch workflows (CTC loss, Adam optimizer, DataLoader)
- **Export** trained models back to Tesseract's binary format for use with the original engine
- **End-to-end recognition**: image → preprocessing → LSTM forward pass → CTC decode → text

## Architecture

Faithfully reimplements Tesseract's C++ network layers:

- Input, Convolve (patch stacking + FC), Maxpool, Reconfig
- LSTM / Summary LSTM with state clipping
- XYTranspose, Reversed wrappers for directional processing
- Series, Parallel composition
- Full spec parser supporting Tesseract's network specification language (e.g. `[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]`)

## Installation

```bash
git clone https://github.com/niushuai1991/tesseract-pytorch.git
cd tesseract-pytorch
uv sync
```

## Quick Start

### Recognition

```python
from tesseract_cuda.recognizer import TesseractRecognizer

recognizer = TesseractRecognizer("eng.traineddata")
text = recognizer.recognize(image)  # PIL Image
```

### Training

```python
from tesseract_cuda.network.model import TessLSTMModel
from tesseract_cuda.training.trainer import LSTMTrainer
from tesseract_cuda.training.dataset import LSTMFDataset

model = TessLSTMModel.from_spec("[1,0,0,1Lfx16O1c3]", num_classes=3)
trainer = LSTMTrainer(model, max_iterations=1000)

dataset = LSTMFDataset(["train.lstmf"], unicharset)
trainer.train(dataset)
```

### Export

```python
from tesseract_cuda.export.exporter import export_model

export_model(model, "eng.traineddata", "output.traineddata",
             training_iteration=1000, sample_iteration=5000)
```

## Project Structure

```
src/tesseract_cuda/
├── formats/          # Tesseract binary format parsers
│   ├── tfile.py      # TFile reader/writer
│   ├── tessdata.py   # .traineddata container
│   ├── network_ser.py # Network serialization
│   ├── unicharset.py  # Character set
│   ├── recoder.py     # Label recoder
│   └── lstmf.py       # Training data format
├── network/          # Neural network layers
│   ├── model.py      # Top-level model (from_traineddata, from_spec)
│   ├── layers.py     # Convolve, Maxpool, Reconfig, LSTM, etc.
│   ├── lstm_cell.py  # LSTM cell implementations
│   ├── spec_parser.py # Network spec language parser
│   └── weight_mapper.py # Weight conversion between formats
├── training/         # Training pipeline
│   ├── trainer.py    # Training loop with CTC loss
│   └── dataset.py    # .lstmf dataset with collation
├── export/           # Model export
│   └── exporter.py   # Export to .traineddata
└── recognizer.py     # End-to-end recognition
```

## Testing

```bash
pytest tests/ -v
```

215 tests covering format parsing, network layers, model construction, training, export, and recognition.

## Requirements

- Python 3.12+
- PyTorch
- NumPy
- Pillow

## License

Apache License 2.0
