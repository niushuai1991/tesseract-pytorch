"""CLI entry point for tesseract-pytorch training tools."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="tesseract-pytorch",
        description="PyTorch-based LSTM training for Tesseract OCR",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new LSTM model")
    train_parser.add_argument("--traineddata", required=True,
                              help="Path to starter traineddata file")
    train_parser.add_argument("--train-list", required=True,
                              help="File listing training .lstmf files")
    train_parser.add_argument("--eval-list", default=None,
                              help="File listing evaluation .lstmf files")
    train_parser.add_argument("--net-spec", default=None,
                              help="Network specification string")
    train_parser.add_argument("--model-output", default="output/checkpoint",
                              help="Base path for output checkpoints")
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--momentum", type=float, default=0.5)
    train_parser.add_argument("--adam-beta", type=float, default=0.999)
    train_parser.add_argument("--max-iterations", type=int, default=0)
    train_parser.add_argument("--target-error-rate", type=float, default=0.01)
    train_parser.add_argument("--checkpoint-interval", type=int, default=10000)
    train_parser.add_argument("--target-height", type=int, default=36)
    train_parser.add_argument("--gpu", type=int, default=-1,
                              help="GPU device ID (-1 for CPU)")

    # Fine-tune command
    ft_parser = subparsers.add_parser("fine-tune", help="Fine-tune an existing model")
    ft_parser.add_argument("--continue-from", required=True,
                           help="Path to existing traineddata or checkpoint")
    ft_parser.add_argument("--traineddata", required=True,
                           help="Path to starter traineddata file")
    ft_parser.add_argument("--train-list", required=True,
                           help="File listing training .lstmf files")
    ft_parser.add_argument("--model-output", default="output/finetuned")
    ft_parser.add_argument("--learning-rate", type=float, default=0.0001)
    ft_parser.add_argument("--max-iterations", type=int, default=0)
    ft_parser.add_argument("--target-error-rate", type=float, default=0.01)
    ft_parser.add_argument("--checkpoint-interval", type=int, default=10000)
    ft_parser.add_argument("--target-height", type=int, default=36)
    ft_parser.add_argument("--gpu", type=int, default=-1)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to traineddata")
    export_parser.add_argument("--checkpoint", required=True,
                               help="Path to PyTorch checkpoint .pt file")
    export_parser.add_argument("--starter", required=True,
                               help="Path to starter traineddata")
    export_parser.add_argument("--output", required=True,
                               help="Output traineddata path")

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect traineddata file")
    inspect_parser.add_argument("path", help="Path to traineddata file")

    # Make-lstmf command
    lstmf_parser = subparsers.add_parser("make-lstmf", help="Convert images to .lstmf format")
    lstmf_parser.add_argument("--image", help="Single image file")
    lstmf_parser.add_argument("--text", help="Text transcription for the image")
    lstmf_parser.add_argument("--output", help="Output .lstmf file path")
    lstmf_parser.add_argument("--input", help="Input file with lines: image_path<TAB>text")
    lstmf_parser.add_argument("--output-dir", default=".", help="Output directory for .lstmf files")
    lstmf_parser.add_argument("--language", default="eng", help="Language code (default: eng)")

    args = parser.parse_args()

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "fine-tune":
        _cmd_finetune(args)
    elif args.command == "export":
        _cmd_export(args)
    elif args.command == "inspect":
        _cmd_inspect(args)
    elif args.command == "make-lstmf":
        _cmd_makelstmf(args)
    else:
        parser.print_help()
        sys.exit(1)


def _read_file_list(path: str) -> list[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _get_device(gpu_id: int) -> str:
    if gpu_id >= 0:
        import torch
        if torch.cuda.is_available():
            return f"cuda:{gpu_id}"
    return "cpu"


def _cmd_train(args):
    import torch
    from .formats.tessdata import TessdataManager, TESSDATA_LSTM_UNICHARSET, TESSDATA_LSTM_RECODER
    from .formats.unicharset import Unicharset
    from .formats.recoder import Recoder
    from .formats.tfile import TFileReader
    from .network.model import TessLSTMModel
    from .training.dataset import LSTMFDataset
    from .training.trainer import LSTMTrainer

    device = _get_device(args.gpu)
    print(f"Using device: {device}")

    # Load unicharset and recoder from starter traineddata
    mgr = TessdataManager.from_file(args.traineddata)
    unicharset_data = mgr.get_component(TESSDATA_LSTM_UNICHARSET)
    recoder_data = mgr.get_component(TESSDATA_LSTM_RECODER)

    unicharset = Unicharset.from_bytes(unicharset_data) if unicharset_data else Unicharset()
    recoder = Recoder.from_bytes(recoder_data) if recoder_data else None

    # Build or load model
    if args.net_spec:
        num_classes = recoder.code_range if recoder else unicharset.size
        model = TessLSTMModel.from_spec(args.net_spec, num_classes)
    else:
        model = TessLSTMModel.from_traineddata(args.traineddata)

    model.null_char = 0  # First code is null/blank

    # Load training data
    train_files = _read_file_list(args.train_list)
    dataset = LSTMFDataset(
        train_files, unicharset, recoder,
        target_height=args.target_height,
        null_char_id=model.null_char,
    )
    print(f"Loaded {len(dataset)} training samples")

    # Train
    trainer = LSTMTrainer(
        model=model, device=device,
        learning_rate=args.learning_rate,
        adam_beta1=args.momentum,
        adam_beta2=args.adam_beta,
        max_iterations=args.max_iterations,
        target_error_rate=args.target_error_rate,
        checkpoint_interval=args.checkpoint_interval,
        model_output=args.model_output,
    )
    trainer.train(dataset, starter_traineddata=args.traineddata)


def _cmd_finetune(args):
    import torch
    from .formats.tessdata import TessdataManager, TESSDATA_LSTM_UNICHARSET, TESSDATA_LSTM_RECODER
    from .formats.unicharset import Unicharset
    from .formats.recoder import Recoder
    from .network.model import TessLSTMModel
    from .training.dataset import LSTMFDataset
    from .training.trainer import LSTMTrainer

    device = _get_device(args.gpu)
    print(f"Using device: {device}")

    # Load model from existing traineddata
    model = TessLSTMModel.from_traineddata(args.continue_from)

    # Load unicharset from starter
    mgr = TessdataManager.from_file(args.traineddata)
    unicharset_data = mgr.get_component(TESSDATA_LSTM_UNICHARSET)
    recoder_data = mgr.get_component(TESSDATA_LSTM_RECODER)

    unicharset = Unicharset.from_bytes(unicharset_data) if unicharset_data else Unicharset()
    recoder = Recoder.from_bytes(recoder_data) if recoder_data else None

    # Load training data
    train_files = _read_file_list(args.train_list)
    dataset = LSTMFDataset(
        train_files, unicharset, recoder,
        null_char_id=model.null_char,
    )
    print(f"Loaded {len(dataset)} training samples")

    # Train
    trainer = LSTMTrainer(
        model=model, device=device,
        learning_rate=args.learning_rate,
        max_iterations=args.max_iterations,
        target_error_rate=args.target_error_rate,
        checkpoint_interval=args.checkpoint_interval,
        model_output=args.model_output,
    )
    trainer.train(dataset, starter_traineddata=args.traineddata)


def _cmd_export(args):
    import torch
    from .network.model import TessLSTMModel
    from .export.exporter import export_model

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"Checkpoint at iteration {checkpoint.get('training_iteration', '?')}")

    # Need to load model architecture first from starter
    model = TessLSTMModel.from_traineddata(args.starter)
    model.load_state_dict(checkpoint["model_state_dict"])

    export_model(
        model, args.starter, args.output,
        training_iteration=checkpoint.get("training_iteration", 0),
        sample_iteration=checkpoint.get("sample_iteration", 0),
    )
    print(f"Exported to {args.output}")


def _cmd_inspect(args):
    from .formats.tessdata import TessdataManager
    mgr = TessdataManager.from_file(args.path)
    components = mgr.list_components()
    print(f"Traineddata: {args.path}")
    print(f"Components ({len(components)}):")
    for idx, name, size in components:
        print(f"  [{idx:2d}] {name:20s} {size:>10d} bytes")


def _cmd_makelstmf(args):
    import os
    from io import BytesIO
    from PIL import Image as PILImage
    from .formats.tfile import TFileWriter
    from .formats.lstmf import ImageData

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input:
        _makelstmf_from_file(args.input, args.output_dir, args.language)
    elif args.image and args.text and args.output:
        _makelstmf_single(args.image, args.text, args.output, args.language)
    else:
        print("Error: either --image/--text/--output or --input required")
        sys.exit(1)


def _makelstmf_single(image_path: str, text: str, output_path: str, language: str):
    import os
    from io import BytesIO
    from PIL import Image as PILImage
    from .formats.lstmf import ImageData
    from .formats.tfile import TFileWriter
    img = PILImage.open(image_path).convert("L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    basename = os.path.basename(image_path)

    page = ImageData(
        imagefilename=basename,
        page_number=0,
        image_data=image_bytes,
        language=language,
        transcription=text,
        boxes=[],
        box_texts=[],
        vertical_text=False,
    )

    _write_lstmf([page], output_path)
    print(f"Created: {output_path}")


def _makelstmf_from_file(input_path: str, output_dir: str, language: str):
    import os
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                print(f"Skipping invalid line: {line}")
                continue
            image_path, text = parts
            basename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{basename}.lstmf")
            _makelstmf_single(image_path, text, output_path, language)


def _write_lstmf(pages, output_path: str):
    from .formats.tfile import TFileWriter
    writer = TFileWriter()
    writer.write_uint32(len(pages))
    for page in pages:
        writer.write_uint8(1)
        writer.write_string(page.imagefilename)
        writer.write_int32(page.page_number)
        writer.write_bytes_vector(page.image_data)
        writer.write_string(page.language)
        writer.write_string(page.transcription)
        writer.write_uint32(len(page.boxes))
        writer.write_uint32(len(page.box_texts))
        writer.write_int8(1 if page.vertical_text else 0)
    with open(output_path, "wb") as f:
        f.write(writer.get_bytes())


if __name__ == "__main__":
    main()
