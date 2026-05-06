"""Parse Tesseract's network specification language.

Example specs:
  [1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]
  [1,0,0,1Ct1,1,48Mp2,2Lfys48Lfx96Lrx96Lfx192O1c1]
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LayerDesc:
    type: str  # "input", "conv", "maxpool", "reconfig", "lstm", "fc", "output", "series", "parallel", "reversed"
    # Input
    batch: int = 1
    height: int = 0
    width: int = 0
    depth: int = 0
    # Conv
    kernel_y: int = 0
    kernel_x: int = 0
    activation: str = "tanh"
    # Maxpool / Reconfig
    y_scale: int = 0
    x_scale: int = 0
    # LSTM
    direction: str = "forward"  # forward, reverse, bidirectional
    dim: str = "x"  # x or y
    summary: bool = False
    num_states: int = 0
    # FC / Output
    num_outputs: int = 0
    output_type: str = ""  # "sequence", "heatmap", "category"
    loss_type: str = ""  # "ctc", "softmax", "logistic"
    # Series/Parallel children
    children: Optional[list["LayerDesc"]] = field(default=None)


def parse_network_spec(spec: str) -> LayerDesc:
    """Parse a complete Tesseract network spec string."""
    parser = _SpecParser(spec)
    return parser.parse()


class _SpecParser:
    def __init__(self, spec: str):
        self.spec = spec
        self.pos = 0

    def parse(self) -> LayerDesc:
        return self._parse_expr()

    def _peek(self) -> str:
        if self.pos < len(self.spec):
            return self.spec[self.pos]
        return ""

    def _advance(self) -> str:
        ch = self.spec[self.pos]
        self.pos += 1
        return ch

    def _parse_expr(self) -> LayerDesc:
        ch = self._peek()
        if ch == '[':
            return self._parse_series()
        elif ch == '(':
            return self._parse_parallel()
        else:
            return self._parse_layer()

    def _parse_series(self) -> LayerDesc:
        children = []
        self._advance()  # skip '['
        # First element might be input spec
        if self._peek().isdigit():
            children.append(self._parse_input())
        while self.pos < len(self.spec) and self._peek() != ']':
            children.append(self._parse_layer())
        if self.pos < len(self.spec):
            self._advance()  # skip ']'
        desc = LayerDesc(type="series", children=children)
        return desc

    def _parse_parallel(self) -> LayerDesc:
        children = []
        self._advance()  # skip '('
        while self.pos < len(self.spec) and self._peek() != ')':
            children.append(self._parse_layer())
        if self.pos < len(self.spec):
            self._advance()  # skip ')'
        desc = LayerDesc(type="parallel", children=children)
        return desc

    def _parse_input(self) -> LayerDesc:
        parts = []
        while self.pos < len(self.spec) and self._peek() in '0123456789,':
            if self._peek() == ',':
                self._advance()
                continue
            num, self.pos = self._read_int()
            parts.append(num)

        while len(parts) < 4:
            parts.append(0)

        return LayerDesc(
            type="input",
            batch=parts[0], height=parts[1],
            width=parts[2], depth=parts[3],
        )

    def _parse_layer(self) -> LayerDesc:
        ch = self._peek()

        if ch == 'C':
            return self._parse_conv()
        elif ch == 'M':
            return self._parse_maxpool()
        elif ch == 'L':
            return self._parse_lstm()
        elif ch == 'F':
            return self._parse_fc()
        elif ch == 'O':
            return self._parse_output()
        elif ch == 'S':
            return self._parse_reconfig()
        elif ch == '[':
            return self._parse_series()
        elif ch == '(':
            return self._parse_parallel()
        else:
            raise ValueError(f"Unexpected char '{ch}' at pos {self.pos} in spec")

    def _parse_conv(self) -> LayerDesc:
        self._advance()  # skip 'C'
        act_ch = self._advance()
        activation = {"s": "sigmoid", "t": "tanh", "r": "relu",
                      "l": "linear", "m": "softmax"}.get(act_ch, "tanh")
        y, x = self._parse_pair()
        if self._peek() == ',':
            self._advance()
        depth = self._parse_number()
        return LayerDesc(
            type="conv", kernel_y=y, kernel_x=x,
            activation=activation, num_outputs=depth,
        )

    def _parse_maxpool(self) -> LayerDesc:
        self._advance()  # skip 'M'
        self._advance()  # skip 'p'
        y, x = self._parse_pair()
        return LayerDesc(type="maxpool", y_scale=y, x_scale=x)

    def _parse_reconfig(self) -> LayerDesc:
        self._advance()  # skip 'S'
        y, x = self._parse_pair()
        return LayerDesc(type="reconfig", y_scale=y, x_scale=x)

    def _parse_lstm(self) -> LayerDesc:
        self._advance()  # skip 'L'
        dir_ch = self._advance()
        direction = {"f": "forward", "r": "reverse", "b": "bidirectional"}.get(
            dir_ch, "forward")
        dim = self._advance()
        summary = False
        if self._peek() == 's':
            self._advance()
            summary = True
        num_states = self._parse_number()
        return LayerDesc(
            type="lstm", direction=direction, dim=dim,
            summary=summary, num_states=num_states,
        )

    def _parse_fc(self) -> LayerDesc:
        self._advance()  # skip 'F'
        act_ch = self._advance()
        activation = {"s": "sigmoid", "t": "tanh", "r": "relu",
                      "l": "linear", "m": "softmax"}.get(act_ch, "linear")
        num_outputs = self._parse_number()
        return LayerDesc(type="fc", activation=activation, num_outputs=num_outputs)

    def _parse_output(self) -> LayerDesc:
        self._advance()  # skip 'O'
        type_ch = self._advance()
        output_type = {"2": "heatmap", "1": "sequence", "0": "category"}.get(
            type_ch, "sequence")
        loss_ch = self._advance()
        loss_type = {"l": "logistic", "s": "softmax", "c": "ctc"}.get(
            loss_ch, "softmax")
        num_outputs = self._parse_number()
        return LayerDesc(
            type="output", output_type=output_type,
            loss_type=loss_type, num_outputs=num_outputs,
        )

    def _parse_pair(self) -> tuple[int, int]:
        a = self._parse_number()
        self._advance()  # skip ','
        b = self._parse_number()
        return a, b

    def _parse_number(self) -> int:
        start = self.pos
        while self.pos < len(self.spec) and self.spec[self.pos].isdigit():
            self.pos += 1
        if self.pos == start:
            return 0
        return int(self.spec[start:self.pos])

    def _read_int(self) -> tuple[int, int]:
        start = self.pos
        while self.pos < len(self.spec) and self.spec[self.pos].isdigit():
            self.pos += 1
        return int(self.spec[start:self.pos]), self.pos
