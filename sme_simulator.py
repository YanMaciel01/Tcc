import numpy as np
from simulator_base import MatrixAcceleratorSimulator, SimulatorConfig, SimulationStep, SimulatorState

class SMESimulator(MatrixAcceleratorSimulator):
    def __init__(self, config: SimulatorConfig):
        super().__init__(config)

    def get_architecture_name(self) -> str:
        return "SME"

    def initialize_simulation(self):
        self.config.tile_h = self.config.N
        self.config.tile_w = self.config.P

        self.A_orig = np.random.randint(-10, 10, size=(self.config.N, self.config.M)).astype(self.config.input_dtype)
        self.B_orig = np.random.randint(-10, 10, size=(self.config.M, self.config.P)).astype(self.config.input_dtype)
        self.C_result = np.zeros((self.config.N, self.config.P), dtype=self.config.acc_dtype)

        self.reg0 = np.zeros((self.config.N,), dtype=self.config.input_dtype)
        self.reg1 = np.zeros((self.config.P,), dtype=self.config.input_dtype)
        self.reg2_acc = np.zeros((self.config.N, self.config.P), dtype=self.config.acc_dtype)

        self._generate_steps()
        self._clear_runtime_state()
        self.current_step_index = -1

    def _generate_steps(self):
        self.steps = []

        # Zerar o acumulador (instrução equivalente: SMSTART ZA / MOV ZA, #0)
        self.steps.append(SimulationStep(
            op_type="TILEZERO", 
            description="SMSTART ZA; MOV ZA, #0 ",
            mode="macro",
            k=-1
        ))

        for k in range(self.config.M):
            # Carregar coluna k de A (LD1B)
            self.steps.append(SimulationStep(
                op_type="LOAD_A", 
                description=f"LD1B Z0.B, P0/Z, [A_col_{k}]",
                mode="macro",
                k=k
            ))

            # Carregar linha k de B (LD1B)
            self.steps.append(SimulationStep(
                op_type="LOAD_B", 
                description=f"LD1B Z1.B, P1/Z, [B_row_{k}]",
                mode="macro",
                k=k
            ))

            if k == 0:
                # Início da visualização detalhada do outer product
                self.steps.append(SimulationStep(
                    op_type="COMPUTE_START_DETAILED",
                    description="Início da multiplicação escalar detalhada (simulada)",
                    mode="macro_marker_for_detailed_start",
                    k=k
                ))
                for i in range(self.config.N):
                    for j in range(self.config.P):
                        self.steps.append(SimulationStep(
                            op_type="SCALAR_MAC",
                            description=f"ZA[{i},{j}] += A[{i},{k}] * B[{k},{j}]",
                            mode="detailed",
                            k=k, i=i, j=j
                        ))
            else:
                # Produto externo completo (SMOPA)
                self.steps.append(SimulationStep(
                    op_type="COMPUTE_MACRO",
                    description=f"SMOPA ZA.S, P2/M, P3/M, Z0.B, Z1.B",
                    mode="macro",
                    k=k
                ))

        # Armazenar resultado (ST1B)
        self.steps.append(SimulationStep(
            op_type="STORE_C",
            description="ST1B ZA, P4, [C]",
            mode="macro",
            k=-1
        ))

        self.total_steps = len(self.steps)

    def _process_current_step_data_logic(self):
        if not (0 <= self.current_step_index < self.total_steps):
            return

        step = self.steps[self.current_step_index]
        op_params = step.params
        k = op_params.get("k", -1)

        if step.op_type == "TILEZERO":
            self.reg2_acc.fill(0)
        elif step.op_type == "LOAD_A" and k != -1:
            self.reg0[:] = self.A_orig[:, k]
            self.elements_loaded += self.config.N
        elif step.op_type == "LOAD_B" and k != -1:
            self.reg1[:] = self.B_orig[k, :]
            self.elements_loaded += self.config.P
        elif step.op_type == "SCALAR_MAC":
            i = op_params["i"]
            j = op_params["j"]
            self.reg2_acc[i, j] += self.reg0[i] * self.reg1[j]
            self.mac_ops += 1
        elif step.op_type == "COMPUTE_MACRO" and k != -1:
            outer_product = np.outer(self.reg0, self.reg1)
            self.reg2_acc += outer_product
            self.mac_ops += self.config.N * self.config.P
        elif step.op_type == "STORE_C":
            self.C_result[:, :] = self.reg2_acc
            self.elements_stored += self.config.N * self.config.P

    def _get_color_maps_for_current_state(self):
        color_A = np.zeros_like(self.A_orig, dtype=int)
        color_B = np.zeros_like(self.B_orig, dtype=int)
        color_C = np.zeros_like(self.C_result, dtype=int)
        color_registers = {
            "reg0": np.zeros((self.config.N, 1), dtype=int),
            "reg1": np.zeros((1, self.config.P), dtype=int),
            "reg2_acc": np.zeros_like(self.reg2_acc, dtype=int)
        }

        if not (0 <= self.current_step_index < self.total_steps):
            if self.current_step_index == self.total_steps:
                color_C.fill(2)  # Green = done
            return color_A, color_B, color_C, color_registers

        step = self.steps[self.current_step_index]
        op_params = step.params
        k = op_params.get("k", -1)

        if step.op_type == "TILEZERO":
            color_registers["reg2_acc"].fill(1)  # Light blue

        elif step.op_type == "LOAD_A" and k != -1:
            color_A[:, k] = 1
            color_registers["reg0"].fill(1)

        elif step.op_type == "LOAD_B" and k != -1:
            color_B[k, :] = 1
            color_registers["reg1"].fill(1)

        elif step.op_type == "COMPUTE_START_DETAILED":
            color_registers["reg0"].fill(2)
            color_registers["reg1"].fill(2)
            color_registers["reg2_acc"].fill(1)

        elif step.op_type == "SCALAR_MAC":
            i, j = op_params["i"], op_params["j"]
            color_registers["reg0"].fill(2)
            color_registers["reg1"].fill(2)
            color_registers["reg2_acc"].fill(1)
            color_registers["reg0"][i, 0] = 6
            color_registers["reg1"][0, j] = 6
            color_registers["reg2_acc"][i, j] = 3
            color_A[i, k] = 6
            color_B[k, j] = 6
            color_C[i, j] = 3

        elif step.op_type == "COMPUTE_MACRO":
            color_registers["reg0"].fill(2)
            color_registers["reg1"].fill(2)
            color_registers["reg2_acc"].fill(3)
            color_A[:, k] = 2
            color_B[k, :] = 2
            color_C[:, :] = 4

        elif step.op_type == "STORE_C":
            color_registers["reg2_acc"].fill(5)
            color_C[:, :] = 5

        return color_A, color_B, color_C, color_registers

    def get_tdp_instruction_name(self) -> str:
        if self.config.dtype_str == "int8":
            return "TDPBSSD"
        elif self.config.dtype_str == "bf16":
            return "TDPBF16PS"
        return "UNKNOWN_TDP_INSTR"

    def get_register_display_info(self) -> list[dict]:
        return [
            {
                "id": "reg0",
                "title": "Z0.B",
                "rows": self.config.N,
                "cols": 1,
                "is_accumulator": False
            },
            {
                "id": "reg1",
                "title": "Z1.B",
                "rows": 1,
                "cols": self.config.P,
                "is_accumulator": False
            },
            {
                "id": "reg2_acc",
                "title": "ZA Accumulator",
                "rows": self.config.N,
                "cols": self.config.P,
                "is_accumulator": True
            }
        ]

    def _compute_computational_intensity(self):
        total_io = self.elements_loaded + self.elements_stored
        return self.mac_ops / total_io if total_io > 0 else 0.0

    def get_current_gui_state(self) -> SimulatorState:
        state = SimulatorState()
        state.window_title_info = f"{self.get_architecture_name()} Simulator - Step {self.current_step_index + 1}/{self.total_steps}"
        state.A_data = self.A_orig.copy()
        state.B_data = self.B_orig.copy()
        state.C_data = self.C_result.copy()
        state.register_data = {
            "reg0": self.reg0.reshape((self.config.N, 1)),
            "reg1": self.reg1.reshape((1, self.config.P)),
            "reg2_acc": self.reg2_acc.copy(),
        }

        color_A, color_B, color_C, reg_colors = self._get_color_maps_for_current_state()
        state.A_colors = color_A
        state.B_colors = color_B
        state.C_colors = color_C
        state.register_colors = reg_colors

        if 0 <= self.current_step_index < self.total_steps:
            current_step = self.steps[self.current_step_index]
            state.current_op_desc = current_step.description
            state.current_mode_text = f"Mode: {current_step.mode.capitalize()}"
            state.current_step_info_text = f"Step: {self.current_step_index + 1} / {self.total_steps}"
        else:
            state.current_op_desc = "Simulation complete."
            state.current_mode_text = "Mode: Finished"
            state.current_step_info_text = f"Step: {self.total_steps} / {self.total_steps}"

        state.metrics = {
            "mac_ops": self.mac_ops,
            "elements_loaded": self.elements_loaded,
            "elements_stored": self.elements_stored,
            "ci": self._compute_computational_intensity(),
        }

        return state
