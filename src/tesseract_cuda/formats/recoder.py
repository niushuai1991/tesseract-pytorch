"""Read and write Tesseract UnicharCompress (recoder) binary format."""

from .tfile import TFileReader, TFileWriter


class Recoder:
    """Maps unicharset IDs to compact code sequences.

    The recoder compresses large character sets (e.g. CJK) into shorter
    sequences of codes from a small alphabet, which is more efficient
    for LSTM training.
    """

    def __init__(self):
        # encoder_[unichar_id] -> (self_normalized, codes)
        self._encoder: list[tuple[bool, list[int]]] = []
        self._code_range: int = 0

    @property
    def num_codes(self) -> int:
        return len(self._encoder)

    @property
    def code_range(self) -> int:
        return self._code_range

    def encode(self, unichar_id: int) -> list[int]:
        if unichar_id < 0 or unichar_id >= len(self._encoder):
            return []
        _, codes = self._encoder[unichar_id]
        return codes

    def decode(self, code_seq: list[int]) -> int:
        """Decode a code sequence back to a unichar ID. Returns -1 if not found."""
        # Build reverse map lazily
        if not hasattr(self, '_decoder'):
            self._decoder = {}
            for uid, (_, codes) in enumerate(self._encoder):
                self._decoder[tuple(codes)] = uid
        result = self._decoder.get(tuple(code_seq), -1)
        return result

    @classmethod
    def from_reader(cls, reader: TFileReader) -> "Recoder":
        recoder = cls()
        count = reader.read_uint32()
        max_code = 0
        for _ in range(count):
            self_normalized = reader.read_int8() != 0
            length = reader.read_int32()
            codes = []
            for _ in range(length):
                c = reader.read_int32()
                codes.append(c)
                if c > max_code:
                    max_code = c
            recoder._encoder.append((self_normalized, codes))
        recoder._code_range = max_code + 1
        return recoder

    def write(self, writer: TFileWriter) -> None:
        writer.write_uint32(len(self._encoder))
        for self_normalized, codes in self._encoder:
            writer.write_int8(1 if self_normalized else 0)
            writer.write_int32(len(codes))
            for c in codes:
                writer.write_int32(c)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Recoder":
        return cls.from_reader(TFileReader(data))

    def to_bytes(self) -> bytes:
        writer = TFileWriter()
        self.write(writer)
        return writer.get_bytes()
