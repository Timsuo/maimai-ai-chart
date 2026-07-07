"""Future event-token tokenizer for ChartIR-like structures."""

from __future__ import annotations

from typing import Any


class ChartEventTokenizer:
    """Skeleton for a future v2.6 event-token decoder pipeline."""

    def __init__(self, tokens: list[str] | None = None) -> None:
        base_tokens = tokens or ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.token_to_id = {token: index for index, token in enumerate(base_tokens)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode_chart_ir(self, chart_ir: dict[str, Any]) -> list[int]:
        raise NotImplementedError("Event-token ChartIR encoding is reserved for v2.6.")

    def decode_tokens(self, tokens: list[int]) -> dict[str, Any]:
        raise NotImplementedError("Event-token ChartIR decoding is reserved for v2.6.")
