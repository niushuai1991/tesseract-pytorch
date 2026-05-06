"""Export PyTorch model to Tesseract traineddata format."""

from ..formats.tessdata import TessdataManager, TESSDATA_LSTM
from ..network.model import TessLSTMModel


def export_model(
    model: TessLSTMModel,
    starter_traineddata_path: str,
    output_path: str,
    training_iteration: int = 0,
    sample_iteration: int = 0,
) -> None:
    """Export a trained PyTorch model to a Tesseract traineddata file.

    This reads the starter traineddata, replaces the LSTM component with
    the trained model weights, and writes the result.
    """
    # Load starter traineddata
    mgr = TessdataManager.from_file(starter_traineddata_path)

    # Generate LSTM component bytes
    lstm_bytes = model.export_lstm_component(
        training_iteration=training_iteration,
        sample_iteration=sample_iteration,
    )

    # Replace LSTM component
    mgr.set_component(TESSDATA_LSTM, lstm_bytes)

    # Save
    mgr.save(output_path)
