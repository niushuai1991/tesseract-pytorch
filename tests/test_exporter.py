import pytest
import os

from tesseract_cuda.network.model import TessLSTMModel
from tesseract_cuda.export.exporter import export_model
from tesseract_cuda.formats.tessdata import TessdataManager, TESSDATA_LSTM
from tesseract_cuda.formats.tfile import TFileReader
from tesseract_cuda.formats.network_ser import deserialize_lstm_component

ENG_TRAINEDDATA = "/tmp/eng.traineddata"


@pytest.mark.skipif(not os.path.exists(ENG_TRAINEDDATA),
                    reason="eng.traineddata not available")
class TestExporter:
    def test_export_creates_file(self, tmp_path):
        model = TessLSTMModel.from_spec("[1,0,0,1Lfx16O1c3]", num_classes=3)
        model.null_char = 0
        out_path = str(tmp_path / "output.traineddata")
        export_model(model, ENG_TRAINEDDATA, out_path,
                     training_iteration=42, sample_iteration=10)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0

    def test_export_roundtrip(self, tmp_path):
        model = TessLSTMModel.from_spec("[1,0,0,1Lfx16O1c3]", num_classes=3)
        model.null_char = 0
        out_path = str(tmp_path / "output.traineddata")
        export_model(model, ENG_TRAINEDDATA, out_path,
                     training_iteration=100, sample_iteration=50)

        mgr = TessdataManager.from_file(out_path)
        lstm_data = mgr.get_component(TESSDATA_LSTM)
        assert lstm_data is not None

        reader = TFileReader(lstm_data)
        parsed = deserialize_lstm_component(reader)
        assert parsed.training_iteration == 100
        assert parsed.sample_iteration == 50

    def test_export_preserves_other_components(self, tmp_path):
        model = TessLSTMModel.from_spec("[1,0,0,1Lfx16O1c3]", num_classes=3)
        model.null_char = 0
        out_path = str(tmp_path / "output.traineddata")
        export_model(model, ENG_TRAINEDDATA, out_path)

        orig_mgr = TessdataManager.from_file(ENG_TRAINEDDATA)
        new_mgr = TessdataManager.from_file(out_path)

        from tesseract_cuda.formats.tessdata import TESSDATA_LSTM_UNICHARSET
        orig_uni = orig_mgr.get_component(TESSDATA_LSTM_UNICHARSET)
        new_uni = new_mgr.get_component(TESSDATA_LSTM_UNICHARSET)
        if orig_uni is not None:
            assert new_uni is not None
            assert orig_uni == new_uni
