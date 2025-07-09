import numpy as np
from simulator_base import MatrixAcceleratorSimulator, SimulatorConfig, SimulationStep, SimulatorState
from typing import Dict, Any, List

class RISCVExtensionSimulator(MatrixAcceleratorSimulator):
    def __init__(self, config: SimulatorConfig):
        super().__init__(config)
        self.init_warnings: List[str] = []
        self.L: int = 0
        self.lambda_val: int = 0

    def get_architecture_name(self) -> str:
        return "RISC-V Ext"

    def get_tdp_instruction_name(self) -> str:
        return "mopf"

    def get_register_display_info(self) -> List[Dict[str, Any]]:
        acc_cols_display = 1
        lambda_display = 1 # Default if lambda_val is 0
        if self.lambda_val > 0:
            acc_cols_display = self.lambda_val * self.lambda_val
            lambda_display = self.lambda_val
        
        acc_cols_display = max(1, acc_cols_display) # ensure at least 1x1
        lambda_display = max(1, lambda_display)   # ensure at least 1x1

        return [
            # u0 displayed as a 1D row vector of lambda elements
            {'id': 'reg_u0',   'title': 'u0 (A-Col as Row)', 'rows': 1, 'cols': lambda_display, 'is_accumulator': False},
            {'id': 'reg_u1',   'title': 'u1 (B-Input Row)',  'rows': 1, 'cols': lambda_display, 'is_accumulator': False},
            {'id': 'reg_acc',  'title': 'Accumulator (Flattened)', 'rows': 1, 'cols': acc_cols_display, 'is_accumulator': True},
        ]

    def get_initialization_warnings(self) -> List[str]:
        return self.init_warnings

    def initialize_simulation(self):
        self.init_warnings.clear()
        self.L = self.config.tile_tmm0_m_rows
        if self.L <= 0:
            raise ValueError(f"L (elements per vector register) must be positive. Got {self.L}.")
        self.lambda_val = int(np.floor(np.sqrt(self.L)))
        if self.lambda_val == 0:
            raise ValueError(f"L={self.L} results in lambda=0. Min L is 1.")

        error_messages = []
        if self.config.N % self.lambda_val != 0: error_messages.append(f"  Global N ({self.config.N}) by lambda ({self.lambda_val})")
        if self.config.M % self.lambda_val != 0: error_messages.append(f"  Global M ({self.config.M}) by lambda ({self.lambda_val})")
        if self.config.P % self.lambda_val != 0: error_messages.append(f"  Global P ({self.config.P}) by lambda ({self.lambda_val})")
        if error_messages: raise ValueError("Matrix dimensions must be divisible by lambda for RISC-V Ext:\n" + "\n".join(error_messages))

        self.A_orig = np.random.randint(-10, 10, size=(self.config.N, self.config.M)).astype(self.config.input_dtype)
        self.B_orig = np.random.randint(-10, 10, size=(self.config.M, self.config.P)).astype(self.config.input_dtype)
        self.C_result = np.zeros((self.config.N, self.config.P), dtype=self.config.acc_dtype)
        
        # Internal representation of u0 is a column vector for dot product with row vector u1
        self.registers['reg_u0'] = np.zeros((self.lambda_val, 1), dtype=self.config.input_dtype)
        self.registers['reg_u1'] = np.zeros((1, self.lambda_val), dtype=self.config.input_dtype)
        self.registers['reg_acc'] = np.zeros((self.lambda_val, self.lambda_val), dtype=self.config.acc_dtype) # Internal 2D

        self._generate_steps()
        self._clear_runtime_state()
        self.current_step_index = -1
        
    def _generate_steps(self): 
        self.steps = []
        if self.lambda_val == 0: self.total_steps = 0; return
        num_tiles_N = self.config.N // self.lambda_val; num_tiles_P = self.config.P // self.lambda_val
        num_tiles_M_K = self.config.M // self.lambda_val
        reg_info_list = self.get_register_display_info() # Call once
        reg_titles = {info['id']: info['title'] for info in reg_info_list}
        title_u0, title_u1, title_acc = reg_titles.get('reg_u0','u0'), reg_titles.get('reg_u1','u1'), reg_titles.get('reg_acc','Acc')
        mopf_instr = self.get_tdp_instruction_name()
        for ti_N in range(num_tiles_N):
            for tj_P in range(num_tiles_P):
                self.steps.append(SimulationStep("ACC_ZERO", f"ZERO {title_acc} (for C tile ({ti_N},{tj_P}))", "macro", ti=ti_N, tj=tj_P, tk=-1))
                for tk_M in range(num_tiles_M_K):
                    for k_inner_idx in range(self.lambda_val):
                        self.steps.append(SimulationStep("LOAD_A_COL", f"LOAD {title_u0} from A({ti_N},{tk_M}) col {k_inner_idx}", "macro", ti=ti_N, tj=-1, tk=tk_M, k_inner=k_inner_idx))
                        self.steps.append(SimulationStep("LOAD_B_ROW", f"LOAD {title_u1} from B({tk_M},{tj_P}) row {k_inner_idx}", "macro", ti=-1, tj=tj_P, tk=tk_M, k_inner=k_inner_idx))
                        self.steps.append(SimulationStep("COMPUTE_MOPF", f"{mopf_instr} {title_acc}, {title_u0}, {title_u1}", "macro", ti=ti_N, tj=tj_P, tk=tk_M, k_inner=k_inner_idx))
                self.steps.append(SimulationStep("STORE_C_TILE", f"STORE {title_acc} to C tile ({ti_N},{tj_P})", "macro", ti=ti_N, tj=tj_P, tk=-1))
        self.total_steps = len(self.steps)

    def _process_current_step_data_logic(self):
        if not (0 <= self.current_step_index < self.total_steps) or self.lambda_val == 0: return
        step = self.steps[self.current_step_index]
        op_params = step.params
        ti_N, tj_P = op_params.get("ti",-1), op_params.get("tj",-1); tk_M, k_inner = op_params.get("tk",-1), op_params.get("k_inner",-1)
        reg_u0, reg_u1, reg_acc = self.registers['reg_u0'], self.registers['reg_u1'], self.registers['reg_acc']
        if step.op_type == "ACC_ZERO": reg_acc.fill(0)
        elif step.op_type == "LOAD_A_COL":
            rsA, reA, cA = ti_N*self.lambda_val, (ti_N+1)*self.lambda_val, tk_M*self.lambda_val + k_inner
            reg_u0[:] = self.A_orig[rsA:reA, cA].reshape(self.lambda_val, 1) # Internal u0 is column
            self.elements_loaded += self.lambda_val
        elif step.op_type == "LOAD_B_ROW":
            rB, csB, ceB = tk_M*self.lambda_val + k_inner, tj_P*self.lambda_val, (tj_P+1)*self.lambda_val
            reg_u1[:] = self.B_orig[rB, csB:ceB].reshape(1, self.lambda_val)
            self.elements_loaded += self.lambda_val
        elif step.op_type == "COMPUTE_MOPF":
            outer_product = np.dot(reg_u0.astype(self.config.acc_dtype), reg_u1.astype(self.config.acc_dtype))
            reg_acc += outer_product
            self.mac_ops += self.lambda_val * self.lambda_val
        elif step.op_type == "STORE_C_TILE":
            rsC,reC,csC,ceC = ti_N*self.lambda_val,(ti_N+1)*self.lambda_val, tj_P*self.lambda_val,(tj_P+1)*self.lambda_val
            self.C_result[rsC:reC, csC:ceC] = reg_acc
            self.elements_stored += reg_acc.size

    def _get_color_maps_for_current_state(self):
        color_A = np.zeros_like(self.A_orig, dtype=int) if self.A_orig is not None else np.array([[]],dtype=int)
        color_B = np.zeros_like(self.B_orig, dtype=int) if self.B_orig is not None else np.array([[]],dtype=int)
        color_C = np.zeros_like(self.C_result, dtype=int) if self.C_result is not None else np.array([[]],dtype=int)
        
        color_registers: Dict[str, np.ndarray] = {}
        lambda_d = max(1, self.lambda_val)
        acc_cols_d = max(1, self.lambda_val * self.lambda_val if self.lambda_val > 0 else 1)

        # Color map for u0 (flattened 1 x lambda for display)
        color_registers['reg_u0'] = np.zeros((1, lambda_d), dtype=int)
        # Color map for u1 (already 1 x lambda for display)
        color_registers['reg_u1'] = np.zeros((1, lambda_d), dtype=int)
        # Color map for accumulator (flattened 1 x lambda*lambda for display)
        color_registers['reg_acc'] = np.zeros((1, acc_cols_d), dtype=int)

        if not (0 <= self.current_step_index < self.total_steps) or self.lambda_val == 0:
            if self.current_step_index == self.total_steps and self.C_result is not None: color_C.fill(2)
            return color_A, color_B, color_C, color_registers

        step = self.steps[self.current_step_index]
        op_params = step.params
        ti_N, tj_P = op_params.get("ti",-1), op_params.get("tj",-1); tk_M, k_inner = op_params.get("tk",-1), op_params.get("k_inner",-1)

        if ti_N != -1 and tj_P != -1:
            c_rs,c_re,c_cs,c_ce = ti_N*self.lambda_val,(ti_N+1)*self.lambda_val, tj_P*self.lambda_val,(tj_P+1)*self.lambda_val
            if step.op_type == "STORE_C_TILE": color_C[c_rs:c_re, c_cs:c_ce] = 2
            elif step.op_type not in ["LOAD_A_COL", "LOAD_B_ROW"]: color_C[c_rs:c_re, c_cs:c_ce] = 4
        
        if step.op_type == "ACC_ZERO": color_registers['reg_acc'].fill(1)
        elif step.op_type == "LOAD_A_COL":
            rsA,reA,cA = ti_N*self.lambda_val,(ti_N+1)*self.lambda_val,tk_M*self.lambda_val+k_inner
            color_A[rsA:reA, cA] = 1
            color_registers['reg_u0'].fill(1) # Fill the 1D color map for u0
        elif step.op_type == "LOAD_B_ROW":
            rB,csB,ceB = tk_M*self.lambda_val+k_inner,tj_P*self.lambda_val,(tj_P+1)*self.lambda_val
            color_B[rB, csB:ceB] = 1
            color_registers['reg_u1'].fill(1)
        elif step.op_type == "COMPUTE_MOPF":
            color_registers['reg_u0'].fill(2); color_registers['reg_u1'].fill(2); color_registers['reg_acc'].fill(3)
            rsA,reA,cA = ti_N*self.lambda_val,(ti_N+1)*self.lambda_val,tk_M*self.lambda_val+k_inner
            color_A[rsA:reA, cA] = 6 
            rB,csB,ceB = tk_M*self.lambda_val+k_inner,tj_P*self.lambda_val,(tj_P+1)*self.lambda_val
            color_B[rB, csB:ceB] = 6
        elif step.op_type == "STORE_C_TILE": color_registers['reg_acc'].fill(5)
            
        return color_A, color_B, color_C, color_registers

    def get_current_gui_state(self) -> SimulatorState:
        state = super().get_current_gui_state()
        if self.get_architecture_name() == "RISC-V Ext" and self.lambda_val > 0:
            # Reshape u0 for GUI display (internal is lambda_val x 1)
            internal_u0_data = self.registers['reg_u0']
            if internal_u0_data is not None:
                state.register_data['reg_u0'] = internal_u0_data.reshape(1, self.lambda_val)
            
            # Reshape accumulator for GUI display (internal is lambda_val x lambda_val)
            internal_acc_data = self.registers['reg_acc']
            if internal_acc_data is not None:
                state.register_data['reg_acc'] = internal_acc_data.reshape(1, self.lambda_val * self.lambda_val)
            # u1 is already 1xlambda internally and for display, so no change needed unless internal form changes.
            # Color maps are already generated in the target display shape by _get_color_maps_for_current_state
        return state
