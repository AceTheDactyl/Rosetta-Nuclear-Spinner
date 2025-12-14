"""
Bridge Package
==============

WebSocket bridge between Nuclear Spinner firmware and Rosetta-Helix.

Components:
    SpinnerBridge: WebSocket server + serial relay for hardware/simulation
    UnifiedStateBridge: Client for receiving unified physics state
    TrainingStateBridge: Extended bridge with training integration
    KuramotoNeuralSystem: 60 Kuramoto oscillators for neural synchronization
    GridCellPlateSystem: 6 neural plates with grid cell dynamics

Neural Interface:
    60 Kuramoto oscillators with 6° spacing, coupled to spinner z-coordinate.
    The Kuramoto model: dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)

    Key connection: sin(60°) = √3/2 = z_c

Usage:
    from bridge import SpinnerBridge, UnifiedStateBridge

    # Server side (hardware/simulation with neural interface)
    bridge = SpinnerBridge()
    await bridge.run()

    # Client side (receive state)
    client = UnifiedStateBridge()
    await client.connect()
    state = client.get_state()

Signature: bridge|v2.0.0|helix|neural
"""

from .spinner_bridge import (
    SpinnerBridge,
    SpinnerState,
    BridgeConfig,
    SpinnerSimulator,
    NeuralState,
)

from .unified_state_bridge import (
    UnifiedStateBridge,
    TrainingStateBridge,
    UnifiedState,
    TriadState,
    KuramotoState,
    GHMPState,
)

# Neural system imports (optional - require numpy)
try:
    from .kuramoto_neural import (
        KuramotoNeuralSystem,
        NeuralTrainingInterface,
        KuramotoSystemState,
        OscillatorState,
        EMFieldState,
        N_OSCILLATORS,
    )
    from .grid_cell_plates import (
        GridCellPlateSystem,
        PlateSpinnerCoupling,
        PlateSystemState,
        NeuralPlate,
        GridCell,
        N_PLATES,
        CELLS_PER_PLATE,
    )
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False

__version__ = "2.0.0"

__all__ = [
    # Spinner Bridge (server)
    'SpinnerBridge',
    'SpinnerState',
    'BridgeConfig',
    'SpinnerSimulator',
    'NeuralState',

    # Unified State Bridge (client)
    'UnifiedStateBridge',
    'TrainingStateBridge',
    'UnifiedState',
    'TriadState',
    'KuramotoState',
    'GHMPState',

    # Neural Systems (optional)
    'NEURAL_AVAILABLE',
]

# Add neural exports if available
if NEURAL_AVAILABLE:
    __all__.extend([
        # Kuramoto System
        'KuramotoNeuralSystem',
        'NeuralTrainingInterface',
        'KuramotoSystemState',
        'OscillatorState',
        'EMFieldState',
        'N_OSCILLATORS',

        # Grid Cell Plates
        'GridCellPlateSystem',
        'PlateSpinnerCoupling',
        'PlateSystemState',
        'NeuralPlate',
        'GridCell',
        'N_PLATES',
        'CELLS_PER_PLATE',
    ])
