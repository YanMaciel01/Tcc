import numpy as np
from simulator_base import MatrixAcceleratorSimulator, SimulatorConfig, SimulationStep, SimulatorState
from typing import Dict, Any, List

class RISCVExtensionSimulator2(MatrixAcceleratorSimulator):
    def __init__(self, config: SimulatorConfig):
        super().__init__(config)
        self.init_warnings: List[str] = []
        self.L: int = 0  # total elements in register (must be square)

    def get_register_display_info(self) -> List[Dict[str, Any]]:
        return [
            {'id': 'reg_u0',   'title': 'u0 (A-Col as Row)', 'rows': 1, 'cols': self.L, 'is_accumulator': False},
            {'id': 'reg_u1',   'title': 'u1 (B-Input Row)',  'rows': 1, 'cols': self.L, 'is_accumulator': False},
            {'id': 'reg_acc',  'title': 'Accumulator', 'rows': self.L, 'cols': self.L, 'is_accumulator': True},
        ]

    def initialize_simulation(self):
        self.init_warnings.clear()
        self.L = self.config.tile_tmm0_m_rows
        
        if self.L <= 0:
            raise ValueError(f"L (elements per vector register) must be positive. Got {self.L}.")

        # Verifica se o produto de N e P é divisível por (L x L)
        if (self.config.N * self.config.P) % (self.L * self.L) != 0:
            raise ValueError(f"The product of N ({self.config.N}) and P ({self.config.P}) must be divisible by L x L ({self.L * self.L}).")

        self.A_orig = np.random.randint(-10, 10, size=(self.config.N, self.config.M)).astype(self.config.input_dtype)
        self.B_orig = np.random.randint(-10, 10, size=(self.config.M, self.config.P)).astype(self.config.input_dtype)
        self.C_result = np.zeros((self.config.N, self.config.P), dtype=self.config.acc_dtype)

        self.registers['reg_u0'] = np.zeros((self.L, 1), dtype=self.config.input_dtype)
        self.registers['reg_u1'] = np.zeros((1, self.L), dtype=self.config.input_dtype)
        self.registers['reg_acc'] = np.zeros((self.L, self.L), dtype=self.config.acc_dtype)

        self._generate_steps()
        self._clear_runtime_state()
        self.current_step_index = -1


    def _generate_steps(self):
        num_tiles_N = self.config.N // self.L
        num_tiles_P = self.config.P // self.L
        num_tiles_M_K = self.config.M // self.L
        self.steps = []

        reg_info_list = self.get_register_display_info()
        reg_titles = {info['id']: info['title'] for info in reg_info_list}
        title_u0 = reg_titles['reg_u0']
        title_u1 = reg_titles['reg_u1']
        title_acc = reg_titles['reg_acc']

        for ti_N in range(num_tiles_N):
            for tj_P in range(num_tiles_P):
                self.steps.append(SimulationStep(
                    "ACC_ZERO",
                    f"ZERO Accumulator (for C tile ({ti_N},{tj_P}))",
                    "macro",
                    ti=ti_N, tj=tj_P, tk=-1
                ))
                for tk_M in range(num_tiles_M_K):
                    for k_inner_idx in range(self.L):
                        self.steps.append(SimulationStep(
                            "LOAD_A_COL",
                            f"LOAD u0 A({ti_N},{tk_M}) col {k_inner_idx}",
                            "macro",
                            ti=ti_N, tj=-1, tk=tk_M, k_inner=k_inner_idx
                        ))
                        self.steps.append(SimulationStep(
                            "LOAD_B_ROW",
                            f"LOAD u1 B({tk_M},{tj_P}) row {k_inner_idx}",
                            "macro",
                            ti=-1, tj=tj_P, tk=tk_M, k_inner=k_inner_idx
                        ))
                        self.steps.append(SimulationStep(
                            "COMPUTE_MOPF",
                            f"mopf {title_acc}, {title_u0} , {title_u1}",
                            "macro",
                            ti=ti_N, tj=tj_P, tk=tk_M, k_inner=k_inner_idx
                        ))
                self.steps.append(SimulationStep(
                    "STORE_C_TILE",
                    f"C({ti_N},{tj_P}) ← Reg_ACC",
                    "macro",
                    ti=ti_N, tj=tj_P, tk=-1
                ))
        self.total_steps = len(self.steps)


    def _process_current_step_data_logic(self):
        if not (0 <= self.current_step_index < self.total_steps): return
        step = self.steps[self.current_step_index]
        ti, tj, tk = step.params.get("ti",-1), step.params.get("tj",-1), step.params.get("tk",-1)
        k_inner = step.params.get("k_inner", -1)
        reg_u0 = self.registers['reg_u0']
        reg_u1 = self.registers['reg_u1']
        reg_acc = self.registers['reg_acc']

        if step.op_type == "ACC_ZERO":
            reg_acc.fill(0)
        elif step.op_type == "LOAD_A_COL":
            reg_u0[:] = self.A_orig[ti * self.L: (ti + 1) * self.L, tk * self.L + k_inner].reshape(self.L, 1)
            self.elements_loaded += self.L
        elif step.op_type == "LOAD_B_ROW":
            reg_u1[:] = self.B_orig[tk * self.L + k_inner, tj * self.L: (tj + 1) * self.L].reshape(1, self.L)
            self.elements_loaded += self.L
        elif step.op_type == "COMPUTE_MOPF":
            reg_acc += np.dot(reg_u0.astype(self.config.acc_dtype), reg_u1.astype(self.config.acc_dtype))
            self.mac_ops += self.L
        elif step.op_type == "STORE_C_TILE":
            self.C_result[ti*self.L:(ti+1)*self.L, tj*self.L:(tj+1)*self.L] = reg_acc
            self.elements_stored += self.L * self.L
    
    def _get_color_maps_for_current_state(self):
        color_A = np.zeros_like(self.A_orig, dtype=int) if self.A_orig is not None else np.array([[]], dtype=int)
        color_B = np.zeros_like(self.B_orig, dtype=int) if self.B_orig is not None else np.array([[]], dtype=int)
        color_C = np.zeros_like(self.C_result, dtype=int) if self.C_result is not None else np.array([[]], dtype=int)

        color_registers: Dict[str, np.ndarray] = {}
        color_registers['reg_u0'] = np.zeros((1, self.L), dtype=int)
        color_registers['reg_u1'] = np.zeros((1, self.L), dtype=int)
        color_registers['reg_acc'] = np.zeros((self.L, self.L), dtype=int)

        if not (0 <= self.current_step_index < self.total_steps):
            if self.current_step_index == self.total_steps and self.C_result is not None:
                color_C.fill(2)
            return color_A, color_B, color_C, color_registers

        step = self.steps[self.current_step_index]
        op_params = step.params
        ti_N, tj_P = op_params.get("ti", -1), op_params.get("tj", -1)
        tk_M, k_inner = op_params.get("tk", -1), op_params.get("k_inner", -1)

        if ti_N != -1 and tj_P != -1:
            c_rs, c_re = ti_N * self.L, (ti_N + 1) * self.L
            c_cs, c_ce = tj_P * self.L, (tj_P + 1) * self.L
            if step.op_type == "STORE_C_TILE":
                color_C[c_rs:c_re, c_cs:c_ce] = 2
            elif step.op_type not in ["LOAD_A_COL", "LOAD_B_ROW"]:
                color_C[c_rs:c_re, c_cs:c_ce] = 4

        if step.op_type == "ACC_ZERO":
            color_registers['reg_acc'].fill(1)
        elif step.op_type == "LOAD_A_COL":
            rsA, reA = ti_N * self.L, (ti_N + 1) * self.L
            cA = tk_M * self.L + k_inner
            color_A[rsA:reA, cA] = 1
            color_registers['reg_u0'].fill(1)
        elif step.op_type == "LOAD_B_ROW":
            rB = tk_M * self.L + k_inner
            csB, ceB = tj_P * self.L, (tj_P + 1) * self.L
            color_B[rB, csB:ceB] = 1
            color_registers['reg_u1'].fill(1)
        elif step.op_type == "COMPUTE_MOPF":
            color_registers['reg_u0'].fill(2)
            color_registers['reg_u1'].fill(2)
            color_registers['reg_acc'].fill(3)
            rsA, reA = ti_N * self.L, (ti_N + 1) * self.L
            cA = tk_M * self.L + k_inner
            color_A[rsA:reA, cA] = 6
            rB = tk_M * self.L + k_inner
            csB, ceB = tj_P * self.L, (tj_P + 1) * self.L
            color_B[rB, csB:ceB] = 6
        elif step.op_type == "STORE_C_TILE":
            color_registers['reg_acc'].fill(5)

        return color_A, color_B, color_C, color_registers

    def get_current_gui_state(self) -> SimulatorState:
        state = super().get_current_gui_state()
        if self.get_architecture_name() == "RISC-V Ext 2" and self.L > 0:

            internal_u0_data = self.registers['reg_u0']
            if internal_u0_data is not None:
                state.register_data['reg_u0'] = internal_u0_data.reshape(1, self.L)

            internal_acc_data = self.registers['reg_acc']
            if internal_acc_data is not None:
                state.register_data['reg_acc'] = internal_acc_data.reshape(self.L, self.L)

        return state
    
    def get_architecture_name(self) -> str:
        return "RISC-V Ext 2"

    def get_tdp_instruction_name(self) -> str:
        return "MOP.F"
