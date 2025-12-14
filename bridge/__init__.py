"""
Bridge Package
==============

WebSocket bridge between Nuclear Spinner firmware and Rosetta-Helix.

Usage:
    from bridge import SpinnerBridge

    bridge = SpinnerBridge()
    await bridge.run()

Signature: bridge|v1.0.0|helix
"""

from .spinner_bridge import (
    SpinnerBridge,
    SpinnerState,
    BridgeConfig,
)

__version__ = "1.0.0"

__all__ = [
    'SpinnerBridge',
    'SpinnerState',
    'BridgeConfig',
]
