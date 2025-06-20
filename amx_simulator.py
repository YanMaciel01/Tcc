import numpy as np
from simulator_base import MatrixAcceleratorSimulator, SimulatorConfig, SimulationStep
from typing import Dict, Any, List 

class AMXSimulator(MatrixAcceleratorSimulator):
    def __init__(self, config: SimulatorConfig):
        super().__init__(config)
        self.init_warnings: List[str] = [] 
        self.cfg_k_dim = 0 # Shared K dimension after validation

    def get_architecture_name(self) -> str:
        return "AMX"

    def get_tdp_instruction_name(self) -> str:
        if self.config.dtype_str == "int8":
            return "TDPBSSD"
        elif self.config.dtype_str == "bf16":
            return "TDPBF16PS"
        return "UNKNOWN_TDP_INSTR"

    def get_register_display_info(self) -> List[Dict[str, Any]]: 
        # tmm2 (accumulator) dimensions depend on tmm0's M and tmm1's N.
        # tmm0 K-cols and tmm1 K-rows must match for TDP.
        return [
            {'id': 'reg0',     'title': 'tmm0', 'rows_attr': 'tile_tmm0_m_rows', 'cols_attr': 'tile_tmm0_k_cols', 'is_accumulator': False},
            {'id': 'reg1',     'title': 'tmm1', 'rows_attr': 'tile_tmm1_k_rows', 'cols_attr': 'tile_tmm1_n_cols', 'is_accumulator': False},
            {'id': 'reg2_acc', 'title': 'tmm2', 'rows_attr': 'tile_tmm0_m_rows', 'cols_attr': 'tile_tmm1_n_cols', 'is_accumulator': True},
        ]

    def get_initialization_warnings(self) -> List[str]:
        return self.init_warnings

    def initialize_simulation(self):
        self.init_warnings.clear() 

        # Critical check for K-dimension compatibility
        if self.config.tile_tmm0_k_cols != self.config.tile_tmm1_k_rows:
            raise ValueError(
                f"K-dimension mismatch: tmm0 K-cols ({self.config.tile_tmm0_k_cols}) "
                f"must equal tmm1 K-rows ({self.config.tile_tmm1_k_rows}) for TDP operation."
            )
        self.cfg_k_dim = self.config.tile_tmm0_k_cols # Store validated shared K-dimension

        # AMX Physical Tile Dimension Warnings
        if self.config.tile_tmm0_m_rows > 16:
            self.init_warnings.append(
                f"AMX Warning: tmm0 Rows (M) ({self.config.tile_tmm0_m_rows}) > 16. "
                f"AMX physical tiles are max 16 rows. Simulation proceeds with logical size."
            )
        if self.config.tile_tmm1_k_rows > 16: # K-dim for tmm1 also defines its row count
             self.init_warnings.append(
                f"AMX Warning: tmm1 Rows (K) ({self.config.tile_tmm1_k_rows}) > 16. "
                f"AMX physical tiles are max 16 rows. Simulation proceeds with logical size."
            )
        
        bytes_per_element = np.dtype(self.config.input_dtype).itemsize
        
        tmm0_width_bytes = self.config.tile_tmm0_k_cols * bytes_per_element
        if tmm0_width_bytes > 64:
            self.init_warnings.append(
                f"AMX Warning: tmm0 Cols (K) ({self.config.tile_tmm0_k_cols}) for dtype {self.config.dtype_str} "
                f"({bytes_per_element} bytes/elem) results in {tmm0_width_bytes} bytes per row for tmm0. "
                f"AMX physical tiles are max 64 bytes wide. Simulation proceeds."
            )

        tmm1_width_bytes = self.config.tile_tmm1_n_cols * bytes_per_element
        if tmm1_width_bytes > 64:
            self.init_warnings.append(
                f"AMX Warning: tmm1 Cols (N) ({self.config.tile_tmm1_n_cols}) for dtype {self.config.dtype_str} "
                f"({bytes_per_element} bytes/elem) results in {tmm1_width_bytes} bytes per row for tmm1. "
                f"AMX physical tiles are max 64 bytes wide. Simulation proceeds."
            )
        
        # Critical Divisibility Checks for Matrix Dimensions by Tile Dimensions
        error_messages = []
        if self.config.N % self.config.tile_tmm0_m_rows != 0:
            error_messages.append(f"  Global N ({self.config.N}) by tmm0 Rows (M) ({self.config.tile_tmm0_m_rows})")
        if self.config.M % self.cfg_k_dim != 0: # Use validated shared K-dim
            error_messages.append(f"  Global M ({self.config.M}) by Tile K-dim ({self.cfg_k_dim})")
        if self.config.P % self.config.tile_tmm1_n_cols != 0:
            error_messages.append(f"  Global P ({self.config.P}) by tmm1 Cols (N) ({self.config.tile_tmm1_n_cols})")
        
        if error_messages:
            raise ValueError("Matrix dimensions must be divisible by corresponding tile dimensions:\n" + "\n".join(error_messages))

        self.A_orig = np.random.randint(-10, 10, size=(self.config.N, self.config.M)).astype(self.config.input_dtype)
        self.B_orig = np.random.randint(-10, 10, size=(self.config.M, self.config.P)).astype(self.config.input_dtype)        
        self.C_result = np.zeros((self.config.N, self.config.P), dtype=self.config.acc_dtype)
        
        self.registers['reg0'] = np.zeros((self.config.tile_tmm0_m_rows, self.cfg_k_dim), dtype=self.config.input_dtype)
        self.registers['reg1'] = np.zeros((self.cfg_k_dim, self.config.tile_tmm1_n_cols), dtype=self.config.input_dtype) 
        self.registers['reg2_acc'] = np.zeros((self.config.tile_tmm0_m_rows, self.config.tile_tmm1_n_cols), dtype=self.config.acc_dtype)

        self._generate_steps()
        self._clear_runtime_state() 
        self.current_step_index = -1
        
    def _generate_steps(self):
        self.steps = []
        
        num_tiles_global_N = self.config.N // self.config.tile_tmm0_m_rows
        num_tiles_global_P = self.config.P // self.config.tile_tmm1_n_cols
        num_tiles_global_M_K = self.config.M // self.cfg_k_dim # K-dim of tiles from global M

        reg_info_list = self.get_register_display_info()
        reg_titles = {info['id']: info['title'] for info in reg_info_list}
        title_reg0 = reg_titles.get('reg0', 'RegA')
        title_reg1 = reg_titles.get('reg1', 'RegB')
        title_reg2_acc = reg_titles.get('reg2_acc', 'RegAcc')
        tdp_instr = self.get_tdp_instruction_name()

        for ti_N_idx in range(num_tiles_global_N): 
            for tj_P_idx in range(num_tiles_global_P): 
                self.steps.append(SimulationStep(
                    op_type="TILEZERO", 
                    description=f"TILEZERO {title_reg2_acc}", 
                    mode="macro", ti=ti_N_idx, tj=tj_P_idx, tk=-1 ))
                
                for tk_K_idx in range(num_tiles_global_M_K): 
                    self.steps.append(SimulationStep(
                        op_type="LOAD_A", 
                        description=f"TILELOADD {title_reg0}, [A({ti_N_idx},{tk_K_idx})]", 
                        mode="macro", ti=ti_N_idx, tj=-1, tk=tk_K_idx))
                    self.steps.append(SimulationStep(
                        op_type="LOAD_B", 
                        description=f"TILELOADD {title_reg1}, [B({tk_K_idx},{tj_P_idx})]", 
                        mode="macro", ti=-1, tj=tj_P_idx, tk=tk_K_idx ))
                    
                    is_first_tile_prod = (ti_N_idx == 0 and tj_P_idx == 0 and tk_K_idx == 0)
                    can_do_detailed = num_tiles_global_N > 0 and num_tiles_global_P > 0 and num_tiles_global_M_K > 0
                    
                    if is_first_tile_prod and can_do_detailed: 
                        self.steps.append(SimulationStep(
                            op_type="COMPUTE_START_DETAILED",
                            description=f"{tdp_instr} {title_reg2_acc}, {title_reg0}, {title_reg1} (detailed view)", 
                            mode="macro_marker_for_detailed_start", ti=ti_N_idx, tj=tj_P_idx, tk=tk_K_idx ))
                        
                        for li_m in range(self.config.tile_tmm0_m_rows): # M-dim of tile (rows of tmm0/tmm2)
                            for lj_n in range(self.config.tile_tmm1_n_cols): # N-dim of tile (cols of tmm1/tmm2)
                                for lk_k_inner in range(self.cfg_k_dim): # K-dim of tile (cols tmm0/rows tmm1)
                                    self.steps.append(SimulationStep(
                                        op_type="SCALAR_MAC",
                                        description=f"{title_reg2_acc}[{li_m},{lj_n}] += {title_reg0}[{li_m},{lk_k_inner}]*{title_reg1}[{lk_k_inner},{lj_n}]", 
                                        mode="detailed", 
                                        ti=ti_N_idx, tj=tj_P_idx, tk=tk_K_idx,
                                        li=li_m, lj=lj_n, lk_inner=lk_k_inner
                                    ))
                    else: 
                        self.steps.append(SimulationStep(
                            op_type="COMPUTE_MACRO",
                            description=f"{tdp_instr} {title_reg2_acc}, {title_reg0}, {title_reg1}", 
                            mode="macro", ti=ti_N_idx, tj=tj_P_idx, tk=tk_K_idx ))
                
                self.steps.append(SimulationStep(
                    op_type="STORE_C",
                    description=f"TILESTORED [C({ti_N_idx},{tj_P_idx})], {title_reg2_acc}", 
                    mode="macro", ti=ti_N_idx, tj=tj_P_idx, tk=-1 ))
        self.total_steps = len(self.steps)

    def _process_current_step_data_logic(self):
        if not (0 <= self.current_step_index < self.total_steps):
            return 
        step = self.steps[self.current_step_index]
        op_params = step.params
        ti_N_idx, tj_P_idx, tk_K_idx = op_params.get("ti",-1), op_params.get("tj",-1), op_params.get("tk",-1)
        
        reg0 = self.registers['reg0']
        reg1 = self.registers['reg1']
        reg2_acc = self.registers['reg2_acc']

        if step.op_type == "TILEZERO":
            reg2_acc.fill(0)
        elif step.op_type == "LOAD_A":
            A_r_slice = slice(ti_N_idx * self.config.tile_tmm0_m_rows, (ti_N_idx + 1) * self.config.tile_tmm0_m_rows)
            A_c_slice = slice(tk_K_idx * self.cfg_k_dim, (tk_K_idx + 1) * self.cfg_k_dim)
            reg0[:] = self.A_orig[A_r_slice, A_c_slice]
            self.elements_loaded += reg0.size
        elif step.op_type == "LOAD_B":
            B_r_slice = slice(tk_K_idx * self.cfg_k_dim, (tk_K_idx + 1) * self.cfg_k_dim)
            B_c_slice = slice(tj_P_idx * self.config.tile_tmm1_n_cols, (tj_P_idx + 1) * self.config.tile_tmm1_n_cols)
            reg1[:] = self.B_orig[B_r_slice, B_c_slice]
            self.elements_loaded += reg1.size
        elif step.op_type == "SCALAR_MAC":
            li_m, lj_n, lk_k_inner = op_params["li"], op_params["lj"], op_params["lk_inner"]
            val_A = reg0[li_m, lk_k_inner]
            val_B = reg1[lk_k_inner, lj_n]
            reg2_acc[li_m, lj_n] += val_A.astype(self.config.acc_dtype) * val_B.astype(self.config.acc_dtype)
            self.mac_ops += 1
        elif step.op_type == "COMPUTE_MACRO":
            prod = np.dot(reg0.astype(self.config.acc_dtype), 
                          reg1.astype(self.config.acc_dtype))
            reg2_acc += prod
            self.mac_ops += self.config.tile_tmm0_m_rows * self.config.tile_tmm1_n_cols * self.cfg_k_dim
        elif step.op_type == "STORE_C":
            C_r_slice = slice(ti_N_idx * self.config.tile_tmm0_m_rows, (ti_N_idx + 1) * self.config.tile_tmm0_m_rows)
            C_c_slice = slice(tj_P_idx * self.config.tile_tmm1_n_cols, (tj_P_idx + 1) * self.config.tile_tmm1_n_cols)
            self.C_result[C_r_slice, C_c_slice] = reg2_acc
            self.elements_stored += reg2_acc.size
        
    def _get_color_maps_for_current_state(self):
        color_A = np.zeros_like(self.A_orig, dtype=int)
        color_B = np.zeros_like(self.B_orig, dtype=int)
        color_C = np.zeros_like(self.C_result, dtype=int)
        color_registers: Dict[str, np.ndarray] = {}
        for reg_id, reg_data in self.registers.items():
            if reg_data is not None:
                color_registers[reg_id] = np.zeros_like(reg_data, dtype=int)
            else:
                color_registers[reg_id] = np.array([[]], dtype=int) 

        if not (0 <= self.current_step_index < self.total_steps):
            if self.current_step_index == self.total_steps: 
                color_C.fill(2) 
            return color_A, color_B, color_C, color_registers

        step = self.steps[self.current_step_index]
        op_params = step.params
        ti_N_idx, tj_P_idx, tk_K_idx = op_params.get("ti",-1), op_params.get("tj",-1), op_params.get("tk",-1)
        
        color_reg0 = color_registers['reg0']
        color_reg1 = color_registers['reg1']
        color_reg2_acc = color_registers['reg2_acc']

        if ti_N_idx != -1 and tj_P_idx != -1 and step.op_type not in ["LOAD_A", "LOAD_B", "SCALAR_MAC", "COMPUTE_START_DETAILED"]:
            C_r_slice = slice(ti_N_idx * self.config.tile_tmm0_m_rows, (ti_N_idx + 1) * self.config.tile_tmm0_m_rows)
            C_c_slice = slice(tj_P_idx * self.config.tile_tmm1_n_cols, (tj_P_idx + 1) * self.config.tile_tmm1_n_cols)
            color_C[C_r_slice, C_c_slice] = 2 if step.op_type == "STORE_C" else 4 

        if step.op_type == "TILEZERO":
            if color_reg2_acc is not None: color_reg2_acc.fill(1) 
        elif step.op_type == "LOAD_A":
            A_r_slice = slice(ti_N_idx * self.config.tile_tmm0_m_rows, (ti_N_idx + 1) * self.config.tile_tmm0_m_rows)
            A_c_slice = slice(tk_K_idx * self.cfg_k_dim, (tk_K_idx + 1) * self.cfg_k_dim)
            color_A[A_r_slice, A_c_slice] = 1 
            if color_reg0 is not None: color_reg0.fill(1) 
        elif step.op_type == "LOAD_B":
            B_r_slice = slice(tk_K_idx * self.cfg_k_dim, (tk_K_idx + 1) * self.cfg_k_dim)
            B_c_slice = slice(tj_P_idx * self.config.tile_tmm1_n_cols, (tj_P_idx + 1) * self.config.tile_tmm1_n_cols)
            color_B[B_r_slice, B_c_slice] = 1 
            if color_reg1 is not None: color_reg1.fill(1) 
        elif step.op_type == "COMPUTE_START_DETAILED":
            A_r_slice_det = slice(ti_N_idx * self.config.tile_tmm0_m_rows, (ti_N_idx + 1) * self.config.tile_tmm0_m_rows)
            A_c_slice_det = slice(tk_K_idx * self.cfg_k_dim, (tk_K_idx + 1) * self.cfg_k_dim)
            color_A[A_r_slice_det, A_c_slice_det] = 4 
            
            B_r_slice_det = slice(tk_K_idx * self.cfg_k_dim, (tk_K_idx + 1) * self.cfg_k_dim)
            B_c_slice_det = slice(tj_P_idx * self.config.tile_tmm1_n_cols, (tj_P_idx + 1) * self.config.tile_tmm1_n_cols)
            color_B[B_r_slice_det, B_c_slice_det] = 4

            C_r_slice_det = slice(ti_N_idx * self.config.tile_tmm0_m_rows, (ti_N_idx + 1) * self.config.tile_tmm0_m_rows)
            C_c_slice_det = slice(tj_P_idx * self.config.tile_tmm1_n_cols, (tj_P_idx + 1) * self.config.tile_tmm1_n_cols)
            color_C[C_r_slice_det, C_c_slice_det] = 4

            if color_reg0 is not None: color_reg0.fill(2)  
            if color_reg1 is not None: color_reg1.fill(2)  
            if color_reg2_acc is not None: color_reg2_acc.fill(1) 
        elif step.op_type == "SCALAR_MAC":
            li_m, lj_n, lk_k_inner = op_params["li"], op_params["lj"], op_params["lk_inner"]
            
            global_A_row = ti_N_idx * self.config.tile_tmm0_m_rows + li_m
            global_A_col = tk_K_idx * self.cfg_k_dim + lk_k_inner
            color_A[global_A_row, global_A_col] = 6 
            
            global_B_row = tk_K_idx * self.cfg_k_dim + lk_k_inner
            global_B_col = tj_P_idx * self.config.tile_tmm1_n_cols + lj_n
            color_B[global_B_row, global_B_col] = 6 

            C_r_slice_det = slice(ti_N_idx * self.config.tile_tmm0_m_rows, (ti_N_idx + 1) * self.config.tile_tmm0_m_rows)
            C_c_slice_det = slice(tj_P_idx * self.config.tile_tmm1_n_cols, (tj_P_idx + 1) * self.config.tile_tmm1_n_cols)
            color_C[C_r_slice_det, C_c_slice_det] = 4 

            if color_reg0 is not None: color_reg0.fill(2); color_reg0[li_m, lk_k_inner] = 6 
            if color_reg1 is not None: color_reg1.fill(2); color_reg1[lk_k_inner, lj_n] = 6 
            if color_reg2_acc is not None: color_reg2_acc.fill(1); color_reg2_acc[li_m, lj_n] = 3 
        elif step.op_type == "COMPUTE_MACRO":
            if color_reg0 is not None: color_reg0.fill(2) 
            if color_reg1 is not None: color_reg1.fill(2) 
            if color_reg2_acc is not None: color_reg2_acc.fill(3) 
        elif step.op_type == "STORE_C":
            if color_reg2_acc is not None: color_reg2_acc.fill(5) 
            
        return color_A, color_B, color_C, color_registers