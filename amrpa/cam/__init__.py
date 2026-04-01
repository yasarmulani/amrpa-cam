"""
CAM: Compressed Attention Memory
----------------------------------
Internal CAM module — used by AMRPA adapters.
"""

from .cam_config import CAMConfig
from .cam_module import CAMModule
from .memory_bank import CAMMemoryBank, CAMMemoryBankSet
from .cam_compression import CAMCompressor, CompressedMemory
from .importance import LearnedImportance, HeuristicImportance, build_importance
from .injection import CAMInjector

__all__ = [
    'CAMConfig',
    'CAMModule',
    'CAMMemoryBank',
    'CAMMemoryBankSet',
    'CAMCompressor',
    'CompressedMemory',
    'LearnedImportance',
    'HeuristicImportance',
    'build_importance',
    'CAMInjector',
]
