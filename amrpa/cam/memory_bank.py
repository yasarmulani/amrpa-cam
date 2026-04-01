"""
CAM: Memory Bank
-----------------
Sliding window storage of CompressedMemory objects.

Design:
    - One CAMMemoryBank per AMRPA layer
    - Fixed max length W: memory cost bounded regardless of sequence/step count
    - FIFO eviction: oldest memory dropped when window full
    - detach() on store: no gradient through stored memories (training stability)
    - Encoder and decoder banks always isolated via CAMMemoryBankSet

Memory cost analysis (worst case):
    W=8, proj_rank=16, d_k=64, batch=32, n_layers=4:
    Per memory: (16×64×2 + seq) × 32 × 4 bytes ≈ 267 KB
    Total: 8 × 4 × 267 KB ≈ 8.5 MB

    Compare naive (seq=384, T=384 steps):
    384 × 4 × 384 × 384 × 32 × 4 bytes ≈ 29 GB
    CAM: ~99.97% reduction
"""

from collections import deque
from typing import List, Optional
from .cam_compression import CompressedMemory


class CAMMemoryBank:
    """
    Sliding window deque of CompressedMemory for one AMRPA layer.

    Not an nn.Module — pure Python container.
    Stateful: reset() must be called between sequences.
    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self._buffer: deque = deque(maxlen=window_size)

    def store(self, compressed: CompressedMemory) -> None:
        """Store compressed memory. Detaches from graph. FIFO eviction."""
        self._buffer.append(compressed.detach())

    def get_all(self) -> List[CompressedMemory]:
        """All stored memories, oldest first."""
        return list(self._buffer)

    def get_last_k(self, k: int) -> List[CompressedMemory]:
        """Most recent k memories, oldest first."""
        buf = list(self._buffer)
        return buf[-k:] if len(buf) >= k else buf

    def reset(self) -> None:
        """Clear all stored memories. Call between sequences."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    def memory_bytes(self) -> int:
        return sum(m.memory_bytes() for m in self._buffer)


class CAMMemoryBankSet:
    """
    Full set of memory banks for one model — one bank per AMRPA layer.

    Handles encoder/decoder isolation for encoder-decoder models:
        encoder banks: bidirectional attention history
        decoder banks: causal attention history
        These never share data.

    Usage pattern (must follow):
        # At start of every new sequence:
        bank_set.reset()

        # During forward pass at layer i:
        memories = bank_set.get(layer_idx=i, side='decoder')
        bank_set.store(layer_idx=i, compressed=cm, side='decoder')
    """

    def __init__(
        self,
        n_amrpa_layers: int,
        window_size: int,
        arch: str = 'encoder'   # 'encoder' | 'decoder' | 'encoder_decoder'
    ):
        self.n_amrpa_layers = n_amrpa_layers
        self.window_size = window_size
        self.arch = arch

        # Primary banks (encoder or decoder depending on arch)
        self._banks: List[CAMMemoryBank] = [
            CAMMemoryBank(window_size) for _ in range(n_amrpa_layers)
        ]

        # Secondary banks for encoder side of encoder_decoder
        self._encoder_banks: Optional[List[CAMMemoryBank]] = None
        if arch == 'encoder_decoder':
            self._encoder_banks = [
                CAMMemoryBank(window_size) for _ in range(n_amrpa_layers)
            ]

    def store(
        self,
        layer_idx: int,
        compressed: CompressedMemory,
        side: str = 'main'   # 'main' | 'encoder' | 'decoder'
    ) -> None:
        self._resolve_bank(layer_idx, side).store(compressed)

    def get(
        self,
        layer_idx: int,
        side: str = 'main'
    ) -> List[CompressedMemory]:
        return self._resolve_bank(layer_idx, side).get_all()

    def get_last_k(
        self,
        layer_idx: int,
        k: int,
        side: str = 'main'
    ) -> List[CompressedMemory]:
        return self._resolve_bank(layer_idx, side).get_last_k(k)

    def reset(self, side: str = 'all') -> None:
        """
        Reset memory banks between sequences.
        ALWAYS call this before processing a new sequence.
        """
        if side in ('all', 'main', 'decoder'):
            for b in self._banks:
                b.reset()
        if self._encoder_banks and side in ('all', 'encoder'):
            for b in self._encoder_banks:
                b.reset()

    def _resolve_bank(self, layer_idx: int, side: str) -> CAMMemoryBank:
        if side == 'encoder' and self._encoder_banks is not None:
            return self._encoder_banks[layer_idx]
        return self._banks[layer_idx]

    def summary(self) -> dict:
        return {
            'arch': self.arch,
            'n_layers': self.n_amrpa_layers,
            'window_size': self.window_size,
            'stored_per_layer': [len(b) for b in self._banks],
            'total_memory_mb': self.total_memory_bytes() / (1024 * 1024)
        }

    def total_memory_bytes(self) -> int:
        total = sum(b.memory_bytes() for b in self._banks)
        if self._encoder_banks:
            total += sum(b.memory_bytes() for b in self._encoder_banks)
        return total
