import sys
from PyQt5.QtWidgets import QApplication, QMessageBox

from simulator_base import SimulatorConfig
from gui import MainWindow
from simulator_factory import create_simulator, get_supported_architectures

# Global constants for default initial configuration
DEFAULT_N_DIM, DEFAULT_M_DIM, DEFAULT_P_DIM = 8, 8, 8
DEFAULT_TILE_TMM0_M_ROWS = 8  # M-dim for tmm0
DEFAULT_TILE_TMM0_K_COLS = 8  # K-dim for tmm0
DEFAULT_TILE_TMM1_K_ROWS = 8  # K-dim for tmm1 (must match TMM0_K_COLS)
DEFAULT_TILE_TMM1_N_COLS = 8  # N-dim for tmm1
DEFAULT_DTYPE = "int8"
DEFAULT_ARCH = "AMX" 

def main():
    initial_config = SimulatorConfig(
        N=DEFAULT_N_DIM, M=DEFAULT_M_DIM, P=DEFAULT_P_DIM,
        tile_tmm0_m_rows=DEFAULT_TILE_TMM0_M_ROWS,
        tile_tmm0_k_cols=DEFAULT_TILE_TMM0_K_COLS,
        tile_tmm1_k_rows=DEFAULT_TILE_TMM1_K_ROWS,
        tile_tmm1_n_cols=DEFAULT_TILE_TMM1_N_COLS,
        dtype_str=DEFAULT_DTYPE
    )

    initial_simulator = None
    try:
        initial_simulator = create_simulator(DEFAULT_ARCH, initial_config)
        initial_simulator.initialize_simulation()

        init_warnings = initial_simulator.get_initialization_warnings()
        if init_warnings:
            if not QApplication.instance(): 
                _app = QApplication(sys.argv)
            QMessageBox.warning(None, "Initial Configuration Warnings", "\n".join(init_warnings))

    except ValueError as e:
        print(f"Error initializing simulator with default config: {e}")
        if not QApplication.instance():
            _app = QApplication(sys.argv)
        QMessageBox.critical(None, "Initialization Error", f"Failed to initialize simulator: {e}")
        sys.exit(1)
    except Exception as e: 
        print(f"Unexpected error during initial simulator setup: {e}")
        if not QApplication.instance():
            _app = QApplication(sys.argv)
        QMessageBox.critical(None, "Fatal Error", f"An unexpected error occurred: {e}")
        sys.exit(1)

    app = QApplication.instance() or QApplication(sys.argv) 
    main_window = MainWindow(initial_simulator)
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()