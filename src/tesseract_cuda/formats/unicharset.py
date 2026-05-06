"""Parse Tesseract unicharset text format."""

from dataclasses import dataclass


@dataclass
class CharEntry:
    unichar: str
    properties: int
    script: str
    other_case: int
    direction: int
    mirror: int
    normed: str


class Unicharset:
    """Tesseract unicharset: maps between characters and integer IDs."""

    def __init__(self):
        self._id_to_char: list[str] = []
        self._char_to_id: dict[str, int] = {}
        self._entries: list[CharEntry] = []

    @property
    def size(self) -> int:
        return len(self._id_to_char)

    def id_to_unichar(self, idx: int) -> str:
        return self._id_to_char[idx]

    def unichar_to_id(self, ch: str) -> int:
        return self._char_to_id[ch]

    def has_unichar(self, ch: str) -> bool:
        return ch in self._char_to_id

    def encode_string(self, text: str) -> list[int]:
        ids = []
        i = 0
        while i < len(text):
            # Try multi-byte match (longest first)
            matched = False
            for length in range(min(4, len(text) - i), 0, -1):
                substr = text[i:i + length]
                if substr in self._char_to_id:
                    ids.append(self._char_to_id[substr])
                    i += length
                    matched = True
                    break
            if not matched:
                i += 1
        return ids

    @classmethod
    def from_text(cls, text: str) -> "Unicharset":
        ucs = cls()
        lines = text.strip().split("\n")
        if not lines:
            return ucs

        count = int(lines[0].strip())
        for i in range(1, min(count + 1, len(lines))):
            line = lines[i].strip()
            if not line:
                continue
            ucs._parse_entry(line)
        return ucs

    @classmethod
    def from_file(cls, path: str) -> "Unicharset":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_text(f.read())

    @classmethod
    def from_bytes(cls, data: bytes) -> "Unicharset":
        return cls.from_text(data.decode("utf-8", errors="replace"))

    def _parse_entry(self, line: str) -> None:
        # Remove comment after tab#
        if "\t#" in line:
            line = line[:line.index("\t#")]

        parts = line.split()
        if not parts:
            return

        unichar = parts[0]
        idx = len(self._id_to_char)

        self._id_to_char.append(unichar)
        self._char_to_id[unichar] = idx

        props = int(parts[1], 16) if len(parts) > 1 else 0
        script = parts[4] if len(parts) > 4 else "Common"
        other_case = int(parts[5]) if len(parts) > 5 else 0
        direction = int(parts[6]) if len(parts) > 6 else 0
        mirror = int(parts[7]) if len(parts) > 7 else 0
        normed = parts[8] if len(parts) > 8 else ""

        self._entries.append(CharEntry(
            unichar=unichar,
            properties=props,
            script=script,
            other_case=other_case,
            direction=direction,
            mirror=mirror,
            normed=normed,
        ))
