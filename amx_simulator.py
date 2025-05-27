import numpy as np
from simulator_base import MatrixAcceleratorSimulator, SimulatorConfig, SimulationStep

class AMXSimulator(MatrixAcceleratorSimulator):
    def __init__(self, config: SimulatorConfig):
        super().__init__(config)
        # AMX specific register names (mapping to generic reg0, reg1, reg2_acc)
        # self.reg0 will be tmm0 (A-Tile)
        # self.reg1 will be tmm1 (B-Tile)
        # self.reg2_acc will be tmm2 (C-Accumulator Tile)

    def get_architecture_name(self) -> str:
        return "AMX"

    def initialize_simulation(self):
        # Validate dimensions (AMX specific checks)
        if self.config.tile_h > 16:
            print(f"Warning: AMX TILE_H ({self.config.tile_h}) > 16. AMX physical tiles are 16 rows.")
        if self.config.tile_w * np.dtype(self.config.input_dtype).itemsize > 64:
            print(f"Warning: AMX TILE_W ({self.config.tile_w}) * sizeof(dtype) > 64 bytes. AMX physical tiles are 64 bytes wide.")
        if self.config.N % self.config.tile_h != 0 or \
           self.config.M % self.config.tile_w != 0 or \
           self.config.P % self.config.tile_w != 0:
            raise ValueError("Matrix dimensions must be divisible by tile dimensions for AMX.")

        # Initialize data matrices
        self.A_orig = np.random.randint(-10, 10, size=(self.config.N, self.config.M)).astype(self.config.input_dtype)
        self.B_orig = np.random.randint(-10, 10, size=(self.config.M, self.config.P)).astype(self.config.input_dtype)
        self.C_result = np.zeros((self.config.N, self.config.P), dtype=self.config.acc_dtype)
        
        # Initialize AMX tile registers (reg0, reg1, reg2_acc)
        self.reg0 = np.zeros((self.config.tile_h, self.config.tile_w), dtype=self.config.input_dtype)
        self.reg1 = np.zeros((self.config.tile_w, self.config.tile_w), dtype=self.config.input_dtype) # AMX B-tile is KxK_bytes, so tile_w x tile_w elements
        self.reg2_acc = np.zeros((self.config.tile_h, self.config.tile_w), dtype=self.config.acc_dtype)

        self._generate_steps()
        self._clear_runtime_state() # Ensure metrics and changeable matrices are zeroed
        self.current_step_index = -1
        
    def _generate_steps(self):
        self.steps = []
        num_tiles_n = self.config.N // self.config.tile_h
        num_tiles_m = self.config.M // self.config.tile_w
        num_tiles_p = self.config.P // self.config.tile_w

        for ti_idx in range(num_tiles_n):
            for tj_idx in range(num_tiles_p):
                self.steps.append(SimulationStep(
                    op_type="TILEZERO", 
                    description="TILEZERO tmm2",
                    mode="macro",
                    ti=ti_idx, tj=tj_idx, tk=-1 # tk not relevant for TILEZERO C
                ))
                for tk_idx in range(num_tiles_m):
                    self.steps.append(SimulationStep(
                        op_type="LOAD_A", 
                        description=f"TILELOADD tmm0, [A({ti_idx},{tk_idx})]",
                        mode="macro",
                        ti=ti_idx, tj=-1, tk=tk_idx
                    ))
                    self.steps.append(SimulationStep(
                        op_type="LOAD_B", 
                        description=f"TILELOADD tmm1, [B({tk_idx},{tj_idx})]",
                        mode="macro",
                        ti=-1, tj=tj_idx, tk=tk_idx
                    ))
                    
                    is_first_tile_prod = (ti_idx == 0 and tj_idx == 0 and tk_idx == 0)
                    can_do_detailed = num_tiles_n > 0 and num_tiles_p > 0 and num_tiles_m > 0

                    if is_first_tile_prod and can_do_detailed: # Detailed for first product
                        self.steps.append(SimulationStep(
                            op_type="COMPUTE_START_DETAILED",
                            description=f"{self.config.tdp_instruction_name} tmm2, tmm0, tmm1 (detailed view)",
                            mode="macro_marker_for_detailed_start", # Special mode
                            ti=ti_idx, tj=tj_idx, tk=tk_idx
                        ))
                        for li in range(self.config.tile_h):
                            for lj in range(self.config.tile_w):
                                for lk_inner in range(self.config.tile_w): # Inner dimension of matmul
                                    self.steps.append(SimulationStep(
                                        op_type="SCALAR_MAC",
                                        description=f"tmm2[{li},{lj}] += tmm0[{li},{lk_inner}]*tmm1[{lk_inner},{lj}]",
                                        mode="detailed",
                                        ti=ti_idx, tj=tj_idx, tk=tk_idx,
                                        li=li, lj=lj, lk_inner=lk_inner
                                    ))
                    else: # Macro compute for other products
                        self.steps.append(SimulationStep(
                            op_type="COMPUTE_MACRO",
                            description=f"{self.config.tdp_instruction_name} tmm2, tmm0, tmm1",
                            mode="macro",
                            ti=ti_idx, tj=tj_idx, tk=tk_idx
                        ))
                self.steps.append(SimulationStep(
                    op_type="STORE_C",
                    description=f"TILESTORED [C({ti_idx},{tj_idx})], tmm2",
                    mode="macro",
                    ti=ti_idx, tj=tj_idx, tk=-1
                ))
        self.total_steps = len(self.steps)

    def _process_current_step_data_logic(self):
        if not (0 <= self.current_step_index < self.total_steps):
            return # Not a valid data processing step

        step = self.steps[self.current_step_index]
        op_params = step.params
        ti, tj, tk = op_params.get("ti",-1), op_params.get("tj",-1), op_params.get("tk",-1)

        if step.op_type == "TILEZERO":
            self.reg2_acc.fill(0)
        elif step.op_type == "LOAD_A":
            A_r_slice = slice(ti * self.config.tile_h, (ti + 1) * self.config.tile_h)
            A_c_slice = slice(tk * self.config.tile_w, (tk + 1) * self.config.tile_w)
            self.reg0[:] = self.A_orig[A_r_slice, A_c_slice]
            self.elements_loaded += self.reg0.size
        elif step.op_type == "LOAD_B":
            B_r_slice = slice(tk * self.config.tile_w, (tk + 1) * self.config.tile_w)
            B_c_slice = slice(tj * self.config.tile_w, (tj + 1) * self.config.tile_w)
            self.reg1[:] = self.B_orig[B_r_slice, B_c_slice]
            self.elements_loaded += self.reg1.size
        elif step.op_type == "SCALAR_MAC":
            li, lj, lk_inner = op_params["li"], op_params["lj"], op_params["lk_inner"]
            val_A = self.reg0[li, lk_inner]
            val_B = self.reg1[lk_inner, lj]
            # Ensure accumulation happens in the accumulator's dtype
            self.reg2_acc[li, lj] += val_A.astype(self.config.acc_dtype) * val_B.astype(self.config.acc_dtype)
            self.mac_ops += 1
        elif step.op_type == "COMPUTE_MACRO":
            # Perform tile-level matrix multiplication
            prod = np.dot(self.reg0.astype(self.config.acc_dtype), 
                          self.reg1.astype(self.config.acc_dtype))
            self.reg2_acc += prod
            self.mac_ops += self.config.tile_h * self.config.tile_w * self.config.tile_w # M*N*K for tiles
        elif step.op_type == "STORE_C":
            C_r_slice = slice(ti * self.config.tile_h, (ti + 1) * self.config.tile_h)
            C_c_slice = slice(tj * self.config.tile_w, (tj + 1) * self.config.tile_w)
            self.C_result[C_r_slice, C_c_slice] = self.reg2_acc
            self.elements_stored += self.reg2_acc.size
        # COMPUTE_START_DETAILED is a marker, no data op itself.
        
    def _get_color_maps_for_current_state(self):
        # Color indices: 0:white, 1:lightskyblue (load), 2:dodgerblue (compute src), 
        # 3:yellow (accum update), 4:lightcoral (C active tile), 5:coral (store), 6:palegreen (detailed scalar sources)
        
        color_A = np.zeros_like(self.A_orig, dtype=int)
        color_B = np.zeros_like(self.B_orig, dtype=int)
        color_C = np.zeros_like(self.C_result, dtype=int)
        color_reg0 = np.zeros_like(self.reg0, dtype=int)
        color_reg1 = np.zeros_like(self.reg1, dtype=int)
        color_reg2_acc = np.zeros_like(self.reg2_acc, dtype=int)

        if not (0 <= self.current_step_index < self.total_steps):
            if self.current_step_index == self.total_steps: # Done state
                color_C.fill(2) # Highlight all of C as complete
            return color_A, color_B, color_C, color_reg0, color_reg1, color_reg2_acc

        step = self.steps[self.current_step_index]
        op_params = step.params
        ti, tj, tk = op_params.get("ti",-1), op_params.get("tj",-1), op_params.get("tk",-1)

        # Highlight C_result tile that is being accumulated into or stored from
        if ti != -1 and tj != -1 and step.op_type not in ["LOAD_A", "LOAD_B"]: # If ti, tj are relevant for C
            C_r_slice = slice(ti * self.config.tile_h, (ti + 1) * self.config.tile_h)
            C_c_slice = slice(tj * self.config.tile_w, (tj + 1) * self.config.tile_w)
            color_C[C_r_slice, C_c_slice] = 2 if step.op_type == "STORE_C" else 4 # 2 for store, 4 for active accum region

        if step.op_type == "TILEZERO":
            color_reg2_acc.fill(1) # Light blue for "being actioned on" (zeroed)
        elif step.op_type == "LOAD_A":
            A_r_slice = slice(ti * self.config.tile_h, (ti + 1) * self.config.tile_h)
            A_c_slice = slice(tk * self.config.tile_w, (tk + 1) * self.config.tile_w)
            color_A[A_r_slice, A_c_slice] = 1 # Source from A
            color_reg0.fill(1) # Target in reg0
        elif step.op_type == "LOAD_B":
            B_r_slice = slice(tk * self.config.tile_w, (tk + 1) * self.config.tile_w)
            B_c_slice = slice(tj * self.config.tile_w, (tj + 1) * self.config.tile_w)
            color_B[B_r_slice, B_c_slice] = 1 # Source from B
            color_reg1.fill(1) # Target in reg1
        elif step.op_type == "COMPUTE_START_DETAILED":
            color_reg0.fill(2)  # Source for compute
            color_reg1.fill(2)  # Source for compute
            color_reg2_acc.fill(1) # Will be updated (initial highlight before detailed steps)
        elif step.op_type == "SCALAR_MAC":
            li, lj, lk_inner = op_params["li"], op_params["lj"], op_params["lk_inner"]
            color_reg0[li, lk_inner] = 6 # palegreen for specific A element
            color_reg1[lk_inner, lj] = 6 # palegreen for specific B element
            color_reg2_acc[li, lj] = 3 # yellow for accumulating C element
        elif step.op_type == "COMPUTE_MACRO":
            color_reg0.fill(2) # Source for compute
            color_reg1.fill(2) # Source for compute
            color_reg2_acc.fill(3) # Accumulating
        elif step.op_type == "STORE_C":
            color_reg2_acc.fill(5) # Source for store
            # C_result tile highlight already handled above
            
        return color_A, color_B, color_C, color_reg0, color_reg1, color_reg2_acc