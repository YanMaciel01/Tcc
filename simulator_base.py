from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Union, List

class SimulatorConfig:
    def __init__(self, N: int = 8, M: int = 8, P: int = 8, 
                 tile_tmm0_m_rows: int = 8, tile_tmm0_k_cols: int = 8, 
                 tile_tmm1_k_rows: int = 8, tile_tmm1_n_cols: int = 8,
                 dtype_str: str = "int8"):
        self.N, self.M, self.P = N, M, P
        self.tile_tmm0_m_rows = tile_tmm0_m_rows # Rows for tmm0 (M-dim of tile compute C[M,N] = A[M,K] * B[K,N])
        self.tile_tmm0_k_cols = tile_tmm0_k_cols # Cols for tmm0 (K-dim of tile compute)
        self.tile_tmm1_k_rows = tile_tmm1_k_rows # Rows for tmm1 (K-dim of tile compute)
        self.tile_tmm1_n_cols = tile_tmm1_n_cols # Cols for tmm1 (N-dim of tile compute)
        self.dtype_str = dtype_str
        
        self.input_dtype = None
        self.acc_dtype = None
        self._set_derived_dtypes()

    def _set_derived_dtypes(self):
        if self.dtype_str == "int8":
            self.input_dtype, self.acc_dtype = np.int8, np.int32
        elif self.dtype_str == "bf16":
            self.input_dtype, self.acc_dtype = np.float16, np.float32 
        elif self.dtype_str == "fp32":
            self.input_dtype, self.acc_dtype = np.float32, np.float32 
        else:
            raise ValueError(f"Unsupported dtype_str: {self.dtype_str}")
        
class SimulationStep:
    def __init__(self, op_type: str, description: str, mode: str, **params: Any) -> None:
        self.op_type = op_type
        self.description = description
        self.mode = mode 
        self.params = params 

class SimulatorState:
    def __init__(self):
        self.A_data, self.B_data, self.C_data = None, None, None
        self.register_data: Dict[str, Union[np.ndarray, None]] = { 
            "reg0": None, "reg1": None, "reg2_acc": None 
        }
        self.A_colors, self.B_colors, self.C_colors = None, None, None
        self.register_colors: Dict[str, Union[np.ndarray, None]] = {
            "reg0": None, "reg1": None, "reg2_acc": None
        }
        self.current_op_desc = "N/A"
        self.current_step_info_text = "Step: 0 / 0"
        self.current_mode_text = "Mode: Initial"
        self.metrics: Dict[str, Union[int, float]] = {
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
        self.A_orig = None 
        self.B_orig = None 
        self.C_result = None 
        self.registers: Dict[str, Union[np.ndarray, None]] = {
            "reg0": None, "reg1": None, "reg2_acc": None
        }
        self.mac_ops = 0
        self.elements_loaded = 0
        self.elements_stored = 0

    @abstractmethod
    def initialize_simulation(self):
        pass

    @abstractmethod
    def _generate_steps(self):
        pass

    @abstractmethod
    def _process_current_step_data_logic(self):
        pass
    
    @abstractmethod
    def _get_color_maps_for_current_state(self) -> tuple[Any, ...]: 
        pass

    @abstractmethod
    def get_architecture_name(self) -> str:
        pass

    @abstractmethod
    def get_register_display_info(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_tdp_instruction_name(self) -> str:
        pass

    def get_initialization_warnings(self) -> List[str]:
        return []

    def _clear_runtime_state(self):
        if self.C_result is not None: self.C_result.fill(0)
        for reg_id in self.registers:
            if self.registers[reg_id] is not None:
                self.registers[reg_id].fill(0)
        self.mac_ops = 0
        self.elements_loaded = 0
        self.elements_stored = 0

    def step_forward(self) -> bool:
        if self.current_step_index < self.total_steps: 
            self.current_step_index += 1
            if self.current_step_index < self.total_steps: 
                self._process_current_step_data_logic()
            return True
        return False 

    def step_backward(self) -> bool:
        if self.current_step_index > -1:
            target_idx = self.current_step_index - 1
            self._clear_runtime_state() 
            for i in range(target_idx + 1): 
                self.current_step_index = i
                self._process_current_step_data_logic()
            self.current_step_index = target_idx 
            return True
        return False 

    def get_current_gui_state(self) -> SimulatorState:
        state = SimulatorState()
        state.A_data = self.A_orig.copy() if self.A_orig is not None else None
        state.B_data = self.B_orig.copy() if self.B_orig is not None else None
        state.C_data = self.C_result.copy() if self.C_result is not None else None
        for reg_id in self.registers:
            if self.registers[reg_id] is not None:
                state.register_data[reg_id] = self.registers[reg_id].copy()
            else:
                state.register_data[reg_id] = None
        color_A, color_B, color_C, dict_color_registers = self._get_color_maps_for_current_state()
        state.A_colors, state.B_colors, state.C_colors = color_A, color_B, color_C
        state.register_colors = dict_color_registers
        if -1 < self.current_step_index < self.total_steps:
            current_step_obj = self.steps[self.current_step_index]
            state.current_op_desc = current_step_obj.description
            state.current_mode_text = f"Mode: {'Detailed Scalar' if current_step_obj.mode == 'detailed' else ('Macro' if current_step_obj.mode == 'macro' else 'Setup')}"
            title_suffix = ""
            op_params = current_step_obj.params
            ti, tj, tk = op_params.get("ti",-1), op_params.get("tj",-1), op_params.get("tk",-1)
            if current_step_obj.mode == "detailed":
                li, lj, lk_inner = op_params["li"], op_params["lj"], op_params["lk_inner"]
                reg_disp_info = self.get_register_display_info()
                title_reg0 = next((info['title'] for info in reg_disp_info if info['id'] == 'reg0'), 'Reg0')
                title_reg1 = next((info['title'] for info in reg_disp_info if info['id'] == 'reg1'), 'Reg1')
                title_reg2_acc = next((info['title'] for info in reg_disp_info if info['id'] == 'reg2_acc'), 'RegAcc')
                title_suffix = f" (Detail: {title_reg2_acc}[{li},{lj}] += {title_reg0}[{li},{lk_inner}]*{title_reg1}[{lk_inner},{lj}])"
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
        state.metrics["mac_ops"] = self.mac_ops
        state.metrics["elements_loaded"] = self.elements_loaded
        state.metrics["elements_stored"] = self.elements_stored
        total_io = self.elements_loaded + self.elements_stored
        state.metrics["ci"] = self.mac_ops / total_io if total_io > 0 else 0.0
        return state

    def get_initial_gui_config(self): 
        return {
            "N_DIM": self.config.N, "M_DIM": self.config.M, "P_DIM": self.config.P,
            "tile_tmm0_m_rows": self.config.tile_tmm0_m_rows,
            "tile_tmm0_k_cols": self.config.tile_tmm0_k_cols,
            "tile_tmm1_k_rows": self.config.tile_tmm1_k_rows,
            "tile_tmm1_n_cols": self.config.tile_tmm1_n_cols,
            "input_dtype_str": self.config.dtype_str,
            "architecture_name": self.get_architecture_name()
        }
