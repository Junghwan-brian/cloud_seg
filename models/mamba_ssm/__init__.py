__version__ = "1.1.1"

import warnings

try:
    from .ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, SELECTIVE_SCAN_CUDA_AVAILABLE
    from .modules.mamba_simple import Mamba
except ImportError as e:
    warnings.warn(
        f"[mamba_ssm] Failed to import Mamba modules: {e}\n"
        "Mamba will not be available. Install required dependencies:\n"
        "  pip install einops\n"
        "For CUDA acceleration, install mamba-ssm: pip install mamba-ssm"
    )
    selective_scan_fn, mamba_inner_fn = None, None
    Mamba = None
    SELECTIVE_SCAN_CUDA_AVAILABLE = False
