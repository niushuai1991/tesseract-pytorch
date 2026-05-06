"""TFile-compatible binary serialization primitives for Tesseract formats."""

import struct
from io import BytesIO
from typing import Optional


class TFileReader:
    """Reads Tesseract's binary serialization format."""

    def __init__(self, data: bytes):
        self._buf = BytesIO(data)
        self._swap = False

    @property
    def offset(self) -> int:
        return self._buf.tell()

    def seek(self, offset: int) -> None:
        self._buf.seek(offset)

    def remaining(self) -> int:
        pos = self._buf.tell()
        self._buf.seek(0, 2)
        end = self._buf.tell()
        self._buf.seek(pos)
        return end - pos

    def read_bytes(self, count: int) -> bytes:
        data = self._buf.read(count)
        if len(data) < count:
            raise EOFError(f"Expected {count} bytes, got {len(data)}")
        return data

    def skip(self, count: int) -> None:
        self._buf.seek(count, 1)

    # --- Primitive type readers ---

    def read_int8(self) -> int:
        return struct.unpack('<b', self.read_bytes(1))[0]

    def read_uint8(self) -> int:
        return struct.unpack('<B', self.read_bytes(1))[0]

    def read_int32(self) -> int:
        return struct.unpack('<i', self.read_bytes(4))[0]

    def read_uint32(self) -> int:
        return struct.unpack('<I', self.read_bytes(4))[0]

    def read_int64(self) -> int:
        return struct.unpack('<q', self.read_bytes(8))[0]

    def read_float(self) -> float:
        return struct.unpack('<f', self.read_bytes(4))[0]

    def read_double(self) -> float:
        return struct.unpack('<d', self.read_bytes(8))[0]

    # --- Compound type readers ---

    def read_string(self) -> str:
        length = self.read_uint32()
        if length == 0:
            return ""
        data = self.read_bytes(length)
        return data.decode("utf-8", errors="replace")

    def read_bytes_vector(self) -> bytes:
        length = self.read_uint32()
        if length == 0:
            return b""
        return self.read_bytes(length)

    def read_int32_vector(self) -> list[int]:
        count = self.read_uint32()
        if count == 0:
            return []
        data = self.read_bytes(count * 4)
        return list(struct.unpack(f'<{count}i', data))

    def read_double_array(self, count: int) -> list[float]:
        if count == 0:
            return []
        data = self.read_bytes(count * 8)
        return list(struct.unpack(f'<{count}d', data))

    def read_2d_double_array(self) -> tuple[int, int, list[float]]:
        """Read GENERIC_2D_ARRAY<double>: uint32 dim1, uint32 dim2, double empty, double[dim1*dim2]."""
        dim1 = self.read_uint32()
        dim2 = self.read_uint32()
        _empty = self.read_double()  # always 0.0
        total = dim1 * dim2
        if total == 0:
            return dim1, dim2, []
        return dim1, dim2, self.read_double_array(total)

    def read_pointer_list(self, deserializer):
        """Read a vector of nullable pointers: uint32 count, then for each:
        uint8 non_null, if non_null: deserialize item."""
        count = self.read_uint32()
        items = []
        for _ in range(count):
            non_null = self.read_uint8()
            if non_null:
                items.append(deserializer(self))
            else:
                items.append(None)
        return items

    def read_size(self) -> int:
        """Read DeSerializeSize: uint32 that may indicate endian swap."""
        size = self.read_uint32()
        return size


class TFileWriter:
    """Writes Tesseract's binary serialization format."""

    def __init__(self):
        self._buf = BytesIO()

    def get_bytes(self) -> bytes:
        return self._buf.getvalue()

    def write_bytes(self, data: bytes) -> None:
        self._buf.write(data)

    def skip(self, count: int) -> None:
        self._buf.write(b'\x00' * count)

    # --- Primitive type writers ---

    def write_int8(self, val: int) -> None:
        self._buf.write(struct.pack('<b', val))

    def write_uint8(self, val: int) -> None:
        self._buf.write(struct.pack('<B', val))

    def write_int32(self, val: int) -> None:
        self._buf.write(struct.pack('<i', val))

    def write_uint32(self, val: int) -> None:
        self._buf.write(struct.pack('<I', val))

    def write_int64(self, val: int) -> None:
        self._buf.write(struct.pack('<q', val))

    def write_float(self, val: float) -> None:
        self._buf.write(struct.pack('<f', val))

    def write_double(self, val: float) -> None:
        self._buf.write(struct.pack('<d', val))

    # --- Compound type writers ---

    def write_string(self, val: str) -> None:
        encoded = val.encode("utf-8")
        self.write_uint32(len(encoded))
        if encoded:
            self._buf.write(encoded)

    def write_bytes_vector(self, val: bytes) -> None:
        self.write_uint32(len(val))
        if val:
            self._buf.write(val)

    def write_int32_vector(self, vals: list[int]) -> None:
        self.write_uint32(len(vals))
        if vals:
            self._buf.write(struct.pack(f'<{len(vals)}i', *vals))

    def write_double_array(self, vals: list[float]) -> None:
        if vals:
            self._buf.write(struct.pack(f'<{len(vals)}d', *vals))

    def write_2d_double_array(self, dim1: int, dim2: int, data: list[float],
                              empty: float = 0.0) -> None:
        """Write GENERIC_2D_ARRAY<double>."""
        self.write_uint32(dim1)
        self.write_uint32(dim2)
        self.write_double(empty)
        self.write_double_array(data)

    def write_pointer_list(self, items: list, serializer) -> None:
        """Write a vector of nullable pointers."""
        self.write_uint32(len(items))
        for item in items:
            if item is not None:
                self.write_uint8(1)
                serializer(self, item)
            else:
                self.write_uint8(0)
