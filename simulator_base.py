from abc import ABC, abstractmethod
import numpy as np

class SimulatorConfig:
    def __init__(self, N=16, M=16, P=16, tile_h=8, tile_w=8, dtype_str="int8"):
        self.N, self.M, self.P = N, M, P
        self.tile_h, self.tile_w = tile_h, tile_w
        self.dtype_str = dtype_str
        
        self.input_dtype = None
        self.acc_dtype = None
        self.tdp_instruction_name = ""
        self._set_derived_dtypes()

    def _set_derived_dtypes(self):
        if self.dtype_str == "int8":
            self.input_dtype, self.acc_dtype = np.int8, np.int32
            self.tdp_instruction_name = "TDPBSSD" # Tile Dot Product Byte Signed-Signed Dword
        elif self.dtype_str == "bf16":
            # NumPy doesn't have a native bf16, use float32 for simulation
            # but acknowledge it's representing bf16 inputs and fp32 accumulation
            self.input_dtype, self.acc_dtype = np.float32, np.float32
            self.tdp_instruction_name = "TDPBF16PS" # Tile Dot Product BF16 Pair to Single
        else:
            raise ValueError(f"Unsupported dtype_str: {self.dtype_str}")

class SimulationStep:
    def __init__(self, op_type, description, mode, **params):
        self.op_type = op_type
        self.description = description
        self.mode = mode # "macro", "detailed", "setup", "macro_marker_for_detailed_start"
        self.params = params # e.g., ti, tj, tk, li, lj, lk_inner

class SimulatorState:
    def __init__(self):
        # Matrix data (NumPy arrays)
        self.A_data, self.B_data, self.C_data = None, None, None
        self.reg0_data, self.reg1_data, self.reg2_acc_data = None, None, None

        # Color maps for matrices (NumPy arrays of integers for indexing)
        self.A_colors, self.B_colors, self.C_colors = None, None, None
        self.reg0_colors, self.reg1_colors, self.reg2_acc_colors = None, None, None
        
        # Sidebar info
        self.current_op_desc = "N/A"
        self.current_step_info_text = "Step: 0 / 0"
        self.current_mode_text = "Mode: Initial"
        self.metrics = {
            "mac_ops": 0,
            "elements_loaded": 0,
            "elements_stored": 0,
            "ci": 0.0
        }
        self.window_title_info = "Simulator"


class MatrixAcceleratorSimulator(ABC):
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.steps = []
        self.current_step_index = -1
        self.total_steps = 0

        # Core data matrices (names match sketch)
        self.A_orig = None  # Source Matrix A
        self.B_orig = None  # Source Matrix B
        self.C_result = None # Result Matrix C (accumulator in global memory)
        
        self.reg0 = None # Corresponds to tmm_A in AMX
        self.reg1 = None # Corresponds to tmm_B in AMX
        self.reg2_acc = None # Corresponds to tmm_C_acc in AMX

        # Metrics
        self.mac_ops = 0
        self.elements_loaded = 0
        self.elements_stored = 0

    @abstractmethod
    def initialize_simulation(self):
        """Initialize data, generate steps, and reset runtime state."""
        pass

    @abstractmethod
    def _generate_steps(self):
        """Generate the sequence of simulation steps."""
        pass

    @abstractmethod
    def _process_current_step_data_logic(self):
        """
        Apply the data changes for the self.steps[self.current_step_index].
        Updates self.reg0, self.reg1, self.reg2_acc, self.C_result, and metrics.
        """
        pass
    
    @abstractmethod
    def _get_color_maps_for_current_state(self):
        """
        Generate color map arrays for all 6 matrices based on the current step.
        Returns a tuple of 6 color map arrays.
        """
        pass

    def _clear_runtime_state(self):
        """Resets matrices that change during simulation and metrics."""
        if self.C_result is not None: self.C_result.fill(0)
        if self.reg0 is not None: self.reg0.fill(0)
        if self.reg1 is not None: self.reg1.fill(0)
        if self.reg2_acc is not None: self.reg2_acc.fill(0)
        self.mac_ops = 0
        self.elements_loaded = 0
        self.elements_stored = 0

    def step_forward(self) -> bool:
        """Advance to the next step. Returns True if successful."""
        if self.current_step_index < self.total_steps: # Can step into "Done" state (index == total_steps)
            self.current_step_index += 1
            if self.current_step_index < self.total_steps: # If it's a valid data processing step
                self._process_current_step_data_logic()
            return True
        return False # Already at or beyond "Done" state

    def step_backward(self) -> bool:
        """Go to the previous step by re-simulating. Returns True if successful."""
        if self.current_step_index > -1:
            target_idx = self.current_step_index - 1
            
            # Reset runtime state and re-simulate from the beginning
            self._clear_runtime_state() 
            # A_orig and B_orig are preserved
            
            # Important: current_step_index must be updated *before* calling _process_current_step_data_logic
            # as _process_current_step_data_logic might use self.current_step_index to fetch step_info
            original_current_step_index_for_replay = self.current_step_index
            
            for i in range(target_idx + 1): # Replay from step 0 up to target_idx
                self.current_step_index = i
                self._process_current_step_data_logic()
            
            self.current_step_index = target_idx # Final correct index
            return True
        return False # Already at the beginning

    def get_current_gui_state(self) -> SimulatorState:
        """Prepare and return the SimulatorState object for the GUI."""
        state = SimulatorState()

        # Assign data matrices
        state.A_data = self.A_orig.copy() if self.A_orig is not None else None
        state.B_data = self.B_orig.copy() if self.B_orig is not None else None
        state.C_data = self.C_result.copy() if self.C_result is not None else None
        state.reg0_data = self.reg0.copy() if self.reg0 is not None else None
        state.reg1_data = self.reg1.copy() if self.reg1 is not None else None
        state.reg2_acc_data = self.reg2_acc.copy() if self.reg2_acc is not None else None
        
        # Generate and assign color maps
        (state.A_colors, state.B_colors, state.C_colors,
         state.reg0_colors, state.reg1_colors, state.reg2_acc_colors) = self._get_color_maps_for_current_state()

        # Populate sidebar info
        if -1 < self.current_step_index < self.total_steps:
            current_step_obj = self.steps[self.current_step_index]
            state.current_op_desc = current_step_obj.description
            state.current_mode_text = f"Mode: {'Detailed Scalar' if current_step_obj.mode == 'detailed' else ('Macro' if current_step_obj.mode == 'macro' else 'Setup')}"
            
            title_suffix = ""
            op_params = current_step_obj.params
            ti, tj, tk = op_params.get("ti",-1), op_params.get("tj",-1), op_params.get("tk",-1)

            if current_step_obj.mode == "detailed":
                li, lj, lk_inner = op_params["li"], op_params["lj"], op_params["lk_inner"]
                title_suffix = f" (Detail: Reg2[{li},{lj}] += Reg0[{li},{lk_inner}]*Reg1[{lk_inner},{lj}])"
            elif ti != -1 and tk != -1 and tj != -1 : title_suffix = f" (Tile A({ti},{tk}), B({tk},{tj}) -> C({ti},{tj}))"
            elif ti != -1 and tk != -1 : title_suffix = f" (Tile A({ti},{tk}))"
            elif tk != -1 and tj != -1 : title_suffix = f" (Tile B({tk},{tj}))"
            elif ti != -1 and tj != -1 : title_suffix = f" (Tile C({ti},{tj}))"
            state.window_title_info = f"{self.get_architecture_name()} - Op: {current_step_obj.op_type}{title_suffix}"

        elif self.current_step_index == -1:
            state.current_op_desc = "Press Next to Start"
            state.current_mode_text = "Mode: Initial"
            state.window_title_info = f"{self.get_architecture_name()} (Initial)"
        elif self.current_step_index == self.total_steps:
            state.current_op_desc = "COMPUTATION COMPLETE"
            state.current_mode_text = "Mode: Done"
            state.window_title_info = f"{self.get_architecture_name()} (Complete)"

        state.current_step_info_text = f"Step: {max(0, self.current_step_index + 1)} / {self.total_steps}"
        if self.current_step_index == self.total_steps:
             state.current_step_info_text = f"Step: {self.total_steps} / {self.total_steps} (Done)"


        # Metrics
        state.metrics["mac_ops"] = self.mac_ops
        state.metrics["elements_loaded"] = self.elements_loaded
        state.metrics["elements_stored"] = self.elements_stored
        total_io = self.elements_loaded + self.elements_stored
        state.metrics["ci"] = self.mac_ops / total_io if total_io > 0 else 0.0
        
        return state

    def get_initial_gui_config(self):
        return {
            "N_DIM": self.config.N, "M_DIM": self.config.M, "P_DIM": self.config.P,
            "tile_h": self.config.tile_h, "tile_w": self.config.tile_w,
            "input_dtype_str": self.config.dtype_str,
            "architecture_name": self.get_architecture_name()
        }

    @abstractmethod
    def get_architecture_name(self) -> str:
        pass