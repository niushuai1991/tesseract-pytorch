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
        rest = parts[2:]

        script, other_case, direction, mirror, normed = _parse_rest(rest)

        self._entries.append(CharEntry(
            unichar=unichar,
            properties=props,
            script=script,
            other_case=other_case,
            direction=direction,
            mirror=mirror,
             normed=normed,
        ))


def _parse_comma_fields(token: str) -> list[int | float] | None:
    """Parse a comma-separated token like '0,255,0,255,0,0,0,0,0,0'.
    Returns None if the token doesn't contain commas or parsing fails."""
    if ',' not in token:
        return None
    try:
        return [int(v) if "." not in v else float(v) for v in token.split(",")]
    except ValueError:
        return None


def _try_parse(rest: list[str], expected_commas: int, has_normed: bool,
               ) -> tuple[str, int, int, int, str] | None:
    """Try parsing `rest` in a specific format.

    Format: bbox(script other_case [direction [mirror [normed]]])
    bbox is expected to have `expected_commas` commas.
    Returns (script, other_case, direction, mirror, normed) or None.
    """
    if not rest:
        return None
    bbox = _parse_comma_fields(rest[0])
    if bbox is None or len(bbox) - 1 != expected_commas:
        return None

    min_fields = 2  # bbox + script
    if has_normed:
        min_fields = 6  # bbox + script + other_case + direction + mirror + normed

    if len(rest) < min_fields:
        return None

    script = rest[1]
    other_case = int(rest[2]) if len(rest) > 2 else 0
    direction = int(rest[3]) if len(rest) > 3 else 0
    mirror = int(rest[4]) if len(rest) > 4 else 0
    normed = rest[5] if has_normed and len(rest) > 5 else ""
    return script, other_case, direction, mirror, normed


def _parse_rest(rest: list[str]) -> tuple[str, int, int, int, str]:
    """Parse the fields after unichar and properties, with fallback.

    Mirrors the C++ UNICHARSET::load_via_fgets fallback logic:
      Level 0: full bbox (9 commas) + script + other_case + direction + mirror + normed
      Level 1: full bbox (9 commas) + script + other_case + direction + mirror
      Level 2: short bbox (3 commas) + script + other_case + direction + mirror
      Level 3: short bbox (3 commas) + script + other_case
      Level 4: script + other_case
      Level 5: script only
    """
    defaults: tuple[str, int, int, int, str] = ("Common", 0, 0, 0, "")

    # Level 0: full bbox with 9 commas + normed
    result = _try_parse(rest, 9, has_normed=True)
    if result is not None:
        return result

    # Level 1: full bbox with 9 commas, no normed
    result = _try_parse(rest, 9, has_normed=False)
    if result is not None:
        return result

    # Level 2: short bbox with 3 commas + direction + mirror
    result = _try_parse(rest, 3, has_normed=False)
    if result is not None:
        return result

    # Level 3: short bbox with 3 commas + other_case only
    if rest and ',' in rest[0]:
        bbox = _parse_comma_fields(rest[0])
        if bbox is not None and len(bbox) - 1 == 3 and len(rest) >= 3:
            script = rest[1]
            other_case = int(rest[2])
            return script, other_case, 0, 0, ""

    # Level 4: script + other_case (no bbox)
    if len(rest) >= 2:
        try:
            script = rest[0]
            other_case = int(rest[1])
            return script, other_case, 0, 0, ""
        except ValueError:
            pass

    # Level 5: script only
    if rest:
        return rest[0], 0, 0, 0, ""

    return defaults
