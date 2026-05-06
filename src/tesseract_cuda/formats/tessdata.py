"""Read and write Tesseract traineddata container files."""

from .tfile import TFileReader, TFileWriter

TESSDATA_NUM_ENTRIES = 24

# Component indices
TESSDATA_LANG_CONFIG = 0
TESSDATA_UNICHARSET = 1
TESSDATA_AMBIGS = 2
TESSDATA_INTTEMP = 3
TESSDATA_PFFMTABLE = 4
TESSDATA_NORMPROTO = 5
TESSDATA_PUNC_DAWG = 6
TESSDATA_SYSTEM_DAWG = 7
TESSDATA_NUMBER_DAWG = 8
TESSDATA_FREQ_DAWG = 9
TESSDATA_SHAPE_TABLE = 13
TESSDATA_BIGRAM_DAWG = 14
TESSDATA_LSTM = 17
TESSDATA_LSTM_PUNC_DAWG = 18
TESSDATA_LSTM_SYSTEM_DAWG = 19
TESSDATA_LSTM_NUMBER_DAWG = 20
TESSDATA_LSTM_UNICHARSET = 21
TESSDATA_LSTM_RECODER = 22
TESSDATA_VERSION = 23

_K_MAX_NUM_ENTRIES = 1000


class TessdataManager:
    """Manages Tesseract traineddata container files."""

    def __init__(self):
        self._entries: dict[int, bytes] = {}

    @classmethod
    def from_file(cls, path: str) -> "TessdataManager":
        with open(path, "rb") as f:
            data = f.read()
        return cls.from_bytes(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> "TessdataManager":
        mgr = cls()
        reader = TFileReader(data)

        num_entries = reader.read_uint32()
        swap = False
        if num_entries > _K_MAX_NUM_ENTRIES:
            # Byteswap needed
            import struct
            num_entries = struct.unpack('>I', struct.pack('<I', num_entries))[0]
            swap = True

        if num_entries > _K_MAX_NUM_ENTRIES:
            raise ValueError(f"Invalid traineddata: num_entries={num_entries}")

        offset_table = []
        for i in range(num_entries):
            offset_table.append(reader.read_int64())

        for i in range(min(num_entries, TESSDATA_NUM_ENTRIES)):
            if offset_table[i] >= 0:
                end = len(data)
                for j in range(i + 1, num_entries):
                    if offset_table[j] >= 0:
                        end = offset_table[j]
                        break
                entry_size = end - offset_table[i]
                mgr._entries[i] = data[offset_table[i]:offset_table[i] + entry_size]

        return mgr

    def has_component(self, index: int) -> bool:
        return index in self._entries

    def get_component(self, index: int) -> bytes:
        return self._entries.get(index, b"")

    def set_component(self, index: int, data: bytes) -> None:
        self._entries[index] = data

    def remove_component(self, index: int) -> None:
        self._entries.pop(index, None)

    def save(self, path: str) -> None:
        data = self.to_bytes()
        with open(path, "wb") as f:
            f.write(data)

    def to_bytes(self) -> bytes:
        writer = TFileWriter()
        num_entries = TESSDATA_NUM_ENTRIES

        # Compute offset table
        offset_table = [0] * TESSDATA_NUM_ENTRIES
        header_size = 4 + 8 * TESSDATA_NUM_ENTRIES  # uint32 + int64[24]
        offset = header_size

        entries_data = {}
        for i in range(TESSDATA_NUM_ENTRIES):
            if i in self._entries and len(self._entries[i]) > 0:
                offset_table[i] = offset
                entries_data[i] = self._entries[i]
                offset += len(self._entries[i])
            else:
                offset_table[i] = -1

        writer.write_uint32(num_entries)
        for off in offset_table:
            writer.write_int64(off)

        for i in range(TESSDATA_NUM_ENTRIES):
            if i in entries_data:
                writer.write_bytes(entries_data[i])

        return writer.get_bytes()

    def list_components(self) -> list[tuple[int, str, int]]:
        names = {
            0: "config", 1: "unicharset", 2: "unicharambigs",
            3: "inttemp", 4: "pffmtable", 5: "normproto",
            6: "punc-dawg", 7: "word-dawg", 8: "number-dawg",
            9: "freq-dawg", 13: "shapetable", 14: "bigram-dawg",
            17: "lstm", 18: "lstm-punc-dawg", 19: "lstm-word-dawg",
            20: "lstm-number-dawg", 21: "lstm-unicharset",
            22: "lstm-recoder", 23: "version",
        }
        result = []
        for idx, data in sorted(self._entries.items()):
            name = names.get(idx, f"unknown-{idx}")
            result.append((idx, name, len(data)))
        return result
