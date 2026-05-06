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

    args = parser.parse_args()

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "fine-tune":
        _cmd_finetune(args)
    elif args.command == "export":
        _cmd_export(args)
    elif args.command == "inspect":
        _cmd_inspect(args)
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


if __name__ == "__main__":
    main()
