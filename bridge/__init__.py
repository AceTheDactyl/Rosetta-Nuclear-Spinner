"""
Bridge Package
==============

WebSocket bridge between Nuclear Spinner firmware and Rosetta-Helix.

Components:
    SpinnerBridge: WebSocket server + serial relay for hardware/simulation
    UnifiedStateBridge: Client for receiving unified physics state
    TrainingStateBridge: Extended bridge with training integration

Usage:
    from bridge import SpinnerBridge, UnifiedStateBridge

    # Server side (hardware/simulation)
    bridge = SpinnerBridge()
    await bridge.run()

    # Client side (receive state)
    client = UnifiedStateBridge()
    await client.connect()
    state = client.get_state()

Signature: bridge|v1.0.0|helix
"""

from .spinner_bridge import (
    SpinnerBridge,
    SpinnerState,
    BridgeConfig,
    SpinnerSimulator,
)

from .unified_state_bridge import (
    UnifiedStateBridge,
    TrainingStateBridge,
    UnifiedState,
    TriadState,
    KuramotoState,
    GHMPState,
)

__version__ = "1.0.0"

__all__ = [
    # Spinner Bridge (server)
    'SpinnerBridge',
    'SpinnerState',
    'BridgeConfig',
    'SpinnerSimulator',

    # Unified State Bridge (client)
    'UnifiedStateBridge',
    'TrainingStateBridge',
    'UnifiedState',
    'TriadState',
    'KuramotoState',
    'GHMPState',
]
