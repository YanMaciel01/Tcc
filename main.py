import sys
import argparse
from PyQt5.QtWidgets import QApplication

from simulator_base import SimulatorConfig
from amx_simulator import AMXSimulator 
from gui import MainWindow

# Global constants (can be part of config or args)
DEFAULT_N_DIM, DEFAULT_M_DIM, DEFAULT_P_DIM = 16, 16, 16 
DEFAULT_TILE_H, DEFAULT_TILE_W = 8, 8
DEFAULT_DTYPE = "int8"

def main():
    parser = argparse.ArgumentParser(description="Matrix Accelerator Simulator")
    parser.add_argument("--tile_h", type=int, default=DEFAULT_TILE_H, help="Default height of logical Reg0/Reg2_Acc tiles.")
    parser.add_argument("--tile_w", type=int, default=DEFAULT_TILE_W, help="Default width of logical Reg0/Reg2_Acc tiles, common K for Reg1.")
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE, choices=["int8", "bf16"], help="Default data type.")
    parser.add_argument("--N", type=int, default=DEFAULT_N_DIM, help="Default N dimension of matrix A (NxM).")
    parser.add_argument("--M", type=int, default=DEFAULT_M_DIM, help="Default M dimension of matrix A, B (NxM, MxP).")
    parser.add_argument("--P", type=int, default=DEFAULT_P_DIM, help="Default P dimension of matrix B (MxP).")
    
    args = parser.parse_args()

    # Create initial simulator configuration from defaults/args
    initial_config = SimulatorConfig(
        N=args.N, M=args.M, P=args.P,
        tile_h=args.tile_h, 
        tile_w=args.tile_w, 
        dtype_str=args.dtype
    )
    
    # Instantiate the initial simulator
    # For now, directly AMX. Could be a factory if multiple architectures.
    initial_simulator = AMXSimulator(initial_config)
    
    try:
        initial_simulator.initialize_simulation()
    except ValueError as e:
        print(f"Error initializing simulator with default/CLI config: {e}")
        # Optionally, show a Qt message box here or just exit
        # For GUI, it's better to let GUI handle errors if possible
        # but initial setup errors might need CLI feedback.
        QApplication.instance() # Ensure QApplication exists for QMessageBox
        if not QApplication.instance(): # If no instance, create one temporarily
            _app = QApplication(sys.argv)
            QMessageBox.critical(None, "Initialization Error", f"Failed to initialize simulator: {e}")
            sys.exit(1)
        else:
             QMessageBox.critical(None, "Initialization Error", f"Failed to initialize simulator: {e}")
             sys.exit(1)


    app = QApplication(sys.argv)
    # Pass the fully configured and initialized simulator to the MainWindow
    main_window = MainWindow(initial_simulator)
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()