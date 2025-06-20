from simulator_base import MatrixAcceleratorSimulator, SimulatorConfig
from amx_simulator import AMXSimulator
from riscv_extension_simulator import RISCVExtensionSimulator
# from sme_simulator import SMESimulator


SUPPORTED_ARCHITECTURES = {
    "AMX": AMXSimulator,
    "RISC-V Ext": RISCVExtensionSimulator,
    # "SME": SMESimulator,
}

def create_simulator(architecture_name: str, config: SimulatorConfig) -> MatrixAcceleratorSimulator:
    """
    Factory function to create a simulator instance based on architecture name.
    """
    simulator_class = SUPPORTED_ARCHITECTURES.get(architecture_name)
    if not simulator_class:
        raise ValueError(f"Unsupported architecture: {architecture_name}. Supported are: {list(SUPPORTED_ARCHITECTURES.keys())}")
    return simulator_class(config)

def get_supported_architectures() -> list[str]:
    """
    Returns a list of supported architecture names.
    """
    return list(SUPPORTED_ARCHITECTURES.keys())