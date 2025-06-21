from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView, QSizePolicy, QFrame, QLineEdit,
    QComboBox, QMessageBox, QSpinBox, QTabWidget
)
from PyQt5.QtGui import QColor, QFont, QPalette, QIntValidator
from PyQt5.QtCore import Qt, QSize
import numpy as np
from simulator_base import MatrixAcceleratorSimulator, SimulatorState, SimulatorConfig
from simulator_factory import create_simulator, get_supported_architectures

QT_CMAP_COLORS = [
    QColor(Qt.white), QColor("lightskyblue"), QColor("dodgerblue"),
    QColor("yellow"), QColor("lightcoral"), QColor("coral"), QColor("palegreen")
]

class MatrixDisplayWidget(QTableWidget): # No changes
    def __init__(self, rows, cols, title="Matrix", parent=None):
        super().__init__(rows, cols, parent)
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(False)
        self.setFocusPolicy(Qt.NoFocus)
        for r in range(rows):
            for c in range(cols):
                item = QTableWidgetItem("")
                item.setTextAlignment(Qt.AlignCenter)
                self.setItem(r, c, item)
        self.base_font_size = 9
        self._adjust_cell_sizes_and_font()

    def _adjust_cell_sizes_and_font(self):
        available_width = self.viewport().width()
        available_height = self.viewport().height()
        rows, cols = self.rowCount(), self.columnCount()
        if rows == 0 or cols == 0: return
        if rows * cols > 256 or self.width() < 150 or self.height() < 100 :
             self.base_font_size = 7
        elif rows * cols > 100:
             self.base_font_size = 8
        else:
             self.base_font_size = 10
        if rows <=8 and cols <=8: self.base_font_size +=1
        font = QFont()
        font.setPointSize(self.base_font_size)
        self.setFont(font)
        cell_w = max(20, (available_width - 5) // cols if cols > 0 else 20)
        cell_h = max(20, (available_height - 5) // rows if rows > 0 else 20)
        for c_idx in range(cols): self.setColumnWidth(c_idx, cell_w)
        for r_idx in range(rows): self.setRowHeight(r_idx, cell_h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._adjust_cell_sizes_and_font()

    def update_matrix(self, data_matrix: np.ndarray, color_matrix: np.ndarray, is_acc_matrix=False, dtype_str='int8'):
        if data_matrix is None or color_matrix is None or self.rowCount() != data_matrix.shape[0] or self.columnCount() != data_matrix.shape[1]:
            for r in range(self.rowCount()):
                for c in range(self.columnCount()):
                    item = self.item(r, c)
                    if item:
                        item.setText("")
                        item.setBackground(QT_CMAP_COLORS[0])
            return
        font = QFont()
        current_base_font_size = self.base_font_size
        fs_adj = current_base_font_size
        if data_matrix.size > 256: fs_adj = max(4, current_base_font_size - 3)
        elif data_matrix.size > 100: fs_adj = max(5, current_base_font_size - 2)
        is_tile_reg = (data_matrix.shape[0] <= 16 and data_matrix.shape[1] <= 16)
        if is_tile_reg and data_matrix.shape[0] * data_matrix.shape[1] > 64 and not is_acc_matrix:
             fs_adj = max(4, current_base_font_size - 4)
        font.setPointSize(fs_adj)
        for r in range(data_matrix.shape[0]):
            for c in range(data_matrix.shape[1]):
                item = self.item(r, c)
                if item is None:
                    item = QTableWidgetItem(); item.setTextAlignment(Qt.AlignCenter); self.setItem(r,c, item)
                val = data_matrix[r, c]
                is_bf16_non_acc = (dtype_str == 'bf16' and not is_acc_matrix)
                if is_bf16_non_acc: val_str = f"{int(val)}"
                elif isinstance(val, (float, np.floating)): val_str = f"{val:.0f}"
                else: val_str = f"{val}"
                item.setText(val_str); item.setFont(font)
                item.setBackground(QT_CMAP_COLORS[color_matrix[r, c] % len(QT_CMAP_COLORS)])

class MainWindow(QMainWindow):
    def __init__(self, initial_simulator: MatrixAcceleratorSimulator):
        super().__init__()
        self.simulator = initial_simulator
        self.initial_sim_config_dict = initial_simulator.get_initial_gui_config()
        self.setWindowTitle(f"Matrix Accelerator Simulator") 
        self.setGeometry(50, 50, 1600, 800)
        self.central_widget = QWidget(); self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.matrix_display_area_widget = QWidget()
        self.matrix_area_layout = QGridLayout(self.matrix_display_area_widget)
        self.main_layout.addWidget(self.matrix_display_area_widget, 4) 

        self.sidebar_frame = QFrame(); self.sidebar_frame.setFrameShape(QFrame.StyledPanel)
        self.sidebar_layout = QVBoxLayout(self.sidebar_frame) 
        self.sidebar_layout.setAlignment(Qt.AlignTop)
        self.main_layout.addWidget(self.sidebar_frame, 1) 
        self.sidebar_frame.setMinimumWidth(380); self.sidebar_frame.setMaximumWidth(500)

        self.register_widgets: dict[str, MatrixDisplayWidget] = {}
        self.arch_tabs_widgets: dict[str, dict] = {} 

        self._setup_sidebar_layout() 

        self._rebuild_matrix_displays_and_labels(self.simulator.config) 
        self._setup_sidebar_dynamic_elements() 
        self._connect_signals()
        self.setWindowTitle(f"{self.simulator.get_architecture_name()} Simulator") 
        self.update_gui()


    def _rebuild_matrix_displays_and_labels(self, current_config: SimulatorConfig): # No changes
        
        layout = self.matrix_area_layout
        while layout.count():
            item = layout.takeAt(0); widget = item.widget()
            if widget: widget.deleteLater()
            else:
                sub_layout = item.layout()
                if sub_layout:
                    while sub_layout.count():
                        sub_item = sub_layout.takeAt(0); sub_widget = sub_item.widget()
                        if sub_widget: sub_widget.deleteLater()
                    sub_layout.deleteLater()

        self.register_widgets.clear()
        self.matrix_A_widget = MatrixDisplayWidget(current_config.N, current_config.M, "Matrix A (Source)")
        self.matrix_B_widget = MatrixDisplayWidget(current_config.M, current_config.P, "Matrix B (Source)")
        self.matrix_C_widget = MatrixDisplayWidget(current_config.N, current_config.P, "Matrix C (Result)")
        self._add_matrix_widget_to_layout(self.matrix_A_widget, 0, 0)
        self._add_matrix_widget_to_layout(self.matrix_B_widget, 0, 1)
        self._add_matrix_widget_to_layout(self.matrix_C_widget, 0, 2)
        register_infos = self.simulator.get_register_display_info()
        
        for i, reg_info in enumerate(register_infos):
            reg_id = reg_info['id']; reg_title = reg_info['title']; rows, cols = 1, 1
            
            if 'rows_attr' in reg_info and 'cols_attr' in reg_info:
                rows = getattr(current_config, reg_info['rows_attr'], 1)
                cols = getattr(current_config, reg_info['cols_attr'], 1)
            
            elif 'rows' in reg_info and 'cols' in reg_info:
                rows = reg_info['rows']; cols = reg_info['cols']
            
            else: QMessageBox.warning(self, "Register Display Error", f"Register '{reg_id}' incomplete.")
            
            rows = max(1, rows); cols = max(1, cols)
            widget = MatrixDisplayWidget(rows, cols, reg_title)
            self.register_widgets[reg_id] = widget
            self._add_matrix_widget_to_layout(widget, 1, i)
        
        for i in range(2): self.matrix_area_layout.setRowStretch(i, 1)
        
        num_display_cols = max(3, len(register_infos))
        
        for i in range(num_display_cols): self.matrix_area_layout.setColumnStretch(i, 1)

    def _add_matrix_widget_to_layout(self, matrix_widget: MatrixDisplayWidget, r, c): # No changes
        container = QVBoxLayout(); container.addWidget(matrix_widget.title_label)
        container.addWidget(matrix_widget); self.matrix_area_layout.addLayout(container, r, c)

    def _create_param_input_pair_widget(self, label_text, default_value, label_width=120, input_width=50): # No changes
        container_widget = QWidget()
        
        layout = QHBoxLayout(container_widget)
        layout.setContentsMargins(0,0,0,0) 
        layout.setSpacing(5)
        
        label = QLabel(label_text); label.setFixedWidth(label_width)
        
        line_edit = QLineEdit(str(default_value))
        line_edit.setValidator(QIntValidator(1, 4096)); line_edit.setFixedWidth(input_width)
        layout.addWidget(label); layout.addWidget(line_edit); layout.addStretch()
        
        return container_widget, line_edit


    def _setup_sidebar_layout(self):
        self.sidebar_layout.addWidget(self._create_sidebar_header("Configuration Styles:"))

        self.tab_widget = QTabWidget()
        supported_architectures = get_supported_architectures()
        default_arch_name = self.simulator.get_architecture_name() 
        default_tab_idx = 0

        for idx, arch_name in enumerate(supported_architectures):
            print(f"--- Creating Tab for: {arch_name} ---") # DEBUG
            arch_tab_content_widget = QWidget() 
            arch_tab_layout = QGridLayout(arch_tab_content_widget)
            arch_tab_layout.setVerticalSpacing(8) 
            arch_tab_layout.setHorizontalSpacing(5)

            tab_inputs = {} 

            n_widget, n_input = self._create_param_input_pair_widget("Global N:", self.initial_sim_config_dict["N_DIM"], 100)
            m_widget, m_input = self._create_param_input_pair_widget("Global M:", self.initial_sim_config_dict["M_DIM"], 100)
            p_widget, p_input = self._create_param_input_pair_widget("Global P:", self.initial_sim_config_dict["P_DIM"], 100)
            tab_inputs['N'] = n_input; tab_inputs['M'] = m_input; tab_inputs['P'] = p_input
            arch_tab_layout.addWidget(n_widget, 0, 0); arch_tab_layout.addWidget(m_widget, 0, 1)
            arch_tab_layout.addWidget(p_widget, 1, 0)
            
            dtype_widget_container = QWidget()
            dtype_layout_h = QHBoxLayout(dtype_widget_container)
            dtype_layout_h.setContentsMargins(0,0,0,0); dtype_layout_h.setSpacing(5)
            
            dtype_label = QLabel("Data Type:"); dtype_label.setFixedWidth(100)
            dtype_combo = QComboBox(); dtype_combo.addItems(["int8", "bf16"])
            dtype_combo.setCurrentText(self.initial_sim_config_dict["input_dtype_str"])
            dtype_combo.setFixedWidth(70)
            tab_inputs['dtype'] = dtype_combo
            dtype_layout_h.addWidget(dtype_label); dtype_layout_h.addWidget(dtype_combo); dtype_layout_h.addStretch()
            arch_tab_layout.addWidget(dtype_widget_container, 1, 1)

            tile_label_w = 110 
            if arch_name == "AMX":
                print(f"  Adding AMX parameters to tab '{arch_name}'")
                t0m_w, t0m_i = self._create_param_input_pair_widget("tmm0 Rows (M):", self.initial_sim_config_dict["tile_tmm0_m_rows"], tile_label_w)
                t0k_w, t0k_i = self._create_param_input_pair_widget("tmm0 Cols (K el.):", self.initial_sim_config_dict["tile_tmm0_k_cols"], tile_label_w)
                tab_inputs['tmm0_m'] = t0m_i; tab_inputs['tmm0_k'] = t0k_i
                arch_tab_layout.addWidget(t0m_w, 2, 0); arch_tab_layout.addWidget(t0k_w, 2, 1)
                
                t1k_w, t1k_i = self._create_param_input_pair_widget("tmm1 Rows (K el.):", self.initial_sim_config_dict["tile_tmm1_k_rows"], tile_label_w)
                t1n_w, t1n_i = self._create_param_input_pair_widget("tmm1 Cols (N el.):", self.initial_sim_config_dict["tile_tmm1_n_cols"], tile_label_w)
                tab_inputs['tmm1_k'] = t1k_i; tab_inputs['tmm1_n'] = t1n_i
                arch_tab_layout.addWidget(t1k_w, 3, 0); arch_tab_layout.addWidget(t1n_w, 3, 1)

            elif arch_name == "SME":
                print(f"  Adding SME parameters to tab '{arch_name}'")
                dtype_combo.setEnabled(False)
                
            elif arch_name == "RISC-V Ext":
                print(f"  Adding RISC-V Ext parameters to tab '{arch_name}'") 
                l_widget, l_input = self._create_param_input_pair_widget("L (Elems/vector reg):", self.initial_sim_config_dict["tile_tmm0_m_rows"], tile_label_w + 35, 50) 
                tab_inputs['L'] = l_input
                arch_tab_layout.addWidget(l_widget, 2, 0, 1, 2) 
                            
            arch_tab_layout.setRowStretch(arch_tab_layout.rowCount(), 1) 
            
            self.tab_widget.addTab(arch_tab_content_widget, arch_name)
            self.arch_tabs_widgets[arch_name] = tab_inputs 
            if arch_name == default_arch_name:
                default_tab_idx = idx
            
            # Additional check after tab is added
            if arch_name == "RISC-V Ext":
                # Ensure the widget is explicitly shown after being added to the tab
                l_widget.show()
                arch_tab_content_widget.update() # try to force update of tab content
                arch_tab_content_widget.adjustSize()
               

        self.tab_widget.setCurrentIndex(default_tab_idx)
        self.sidebar_layout.addWidget(self.tab_widget) 

        self.apply_config_button = QPushButton("Apply Configuration & Restart")
        self.sidebar_layout.addWidget(self.apply_config_button)
        self.sidebar_layout.addSpacing(15) 


    def _setup_sidebar_dynamic_elements(self):
        self.mode_label = self._create_sidebar_header("Mode: Initial", bold=True, color="darkblue")
        self.sidebar_layout.addWidget(self.mode_label)
        self.sidebar_layout.addSpacing(10)
        self.asm_op_header_label = self._create_sidebar_header(f"{self.simulator.get_architecture_name()} Assembly / Operation:")
        self.sidebar_layout.addWidget(self.asm_op_header_label)
        
        self.asm_op_label = QLabel("...")
        self.asm_op_label.setWordWrap(True); self.asm_op_label.setStyleSheet("font-family: monospace;")
        self.sidebar_layout.addWidget(self.asm_op_label)
        self.sidebar_layout.addSpacing(10)
        self.sidebar_layout.addWidget(self._create_sidebar_header("Computational Intensity Metrics:"))
        
        self.metrics_mac_ops_label = QLabel("MAC Operations: 0")
        self.metrics_elements_loaded_label = QLabel("Elements Loaded: 0")
        self.metrics_elements_stored_label = QLabel("Elements Stored: 0")
       
        self.metrics_ci_label = QLabel("Computational Intensity (MACs/Elem IO): 0.00")
        self.sidebar_layout.addWidget(self.metrics_mac_ops_label)
        self.sidebar_layout.addWidget(self.metrics_elements_loaded_label)
        self.sidebar_layout.addWidget(self.metrics_elements_stored_label)
        self.sidebar_layout.addWidget(self.metrics_ci_label)
        self.sidebar_layout.addSpacing(10)
        
        self.step_info_label = QLabel("Step: 0 / 0")
        self.sidebar_layout.addWidget(self.step_info_label)
        self.sidebar_layout.addSpacing(20)
        controls_widget = QWidget()
        self.controls_layout = QHBoxLayout(controls_widget)
        self.controls_layout.setContentsMargins(0,0,0,0)
        
        self.prev_button = QPushButton("Previous (Left Arrow)")
        self.next_button = QPushButton("Next (Right Arrow)")
        self.controls_layout.addWidget(self.prev_button); self.controls_layout.addWidget(self.next_button)
        self.sidebar_layout.addWidget(controls_widget)
        self.sidebar_layout.addStretch(1) 

    def _create_sidebar_header(self, text, bold=True, color=None):
        label = QLabel(text); font = label.font(); font.setBold(bold)
        font.setPointSize(font.pointSize() + (1 if bold else 0)); label.setFont(font)
        if color: label.setStyleSheet(f"color: {color};")
        return label

    def _connect_signals(self):
        self.apply_config_button.clicked.connect(self.on_apply_config)
        if hasattr(self, 'next_button'): 
            self.next_button.clicked.connect(self.on_next_step)
            self.prev_button.clicked.connect(self.on_prev_step)


    def on_apply_config(self):
        try:
            current_tab_arch_name = self.tab_widget.tabText(self.tab_widget.currentIndex())
            inputs = self.arch_tabs_widgets[current_tab_arch_name]
            
            N = int(inputs['N'].text()); M = int(inputs['M'].text()); P = int(inputs['P'].text())
            
            dtype_str = inputs['dtype'].currentText()
            tile_tmm0_m_rows, tile_tmm0_k_cols, tile_tmm1_k_rows, tile_tmm1_n_cols = 1,1,1,1 
            
            if current_tab_arch_name == "AMX":
                tile_tmm0_m_rows = int(inputs['tmm0_m'].text()); tile_tmm0_k_cols = int(inputs['tmm0_k'].text())
                tile_tmm1_k_rows = int(inputs['tmm1_k'].text()); tile_tmm1_n_cols = int(inputs['tmm1_n'].text())
            
            elif current_tab_arch_name == "RISC-V Ext":
                tile_tmm0_m_rows = int(inputs['L'].text()) 
            
            if not (N > 0 and M > 0 and P > 0 and tile_tmm0_m_rows > 0 and \
                    tile_tmm0_k_cols > 0 and tile_tmm1_k_rows > 0 and tile_tmm1_n_cols > 0):
                raise ValueError("Dimensions and relevant tile/L sizes must be positive.")
            
            # Show warning if dimensions are not compatible with SME
            if current_tab_arch_name == "SME" and (N > 8 or M > 8 or P > 8):
                N = 8
                M = 8
                P = 8
                QMessageBox.warning(self, "Configuration Warning",
                                    "Matrix dimensions (N, M, P) should be 8 or lower for SME simulation\n"
                                    "For this simulator, the registers have an 128bits size and the ZA accumulator 512bits size.\n"
                                    "Setting N, M, P to 8 for compatibility with SME architecture.")
            
            new_config = SimulatorConfig(
                N=N, 
                M=M, 
                P=P,
                tile_tmm0_m_rows=tile_tmm0_m_rows, 
                tile_tmm0_k_cols=tile_tmm0_k_cols,
                tile_tmm1_k_rows=tile_tmm1_k_rows, 
                tile_tmm1_n_cols=tile_tmm1_n_cols,
                dtype_str=dtype_str
            )

            self.simulator = create_simulator(current_tab_arch_name, new_config)
            self.simulator.initialize_simulation()
            init_warnings = self.simulator.get_initialization_warnings()
            
            if init_warnings: QMessageBox.warning(self, f"{current_tab_arch_name} Config Warnings", "\n".join(init_warnings))
            self._rebuild_matrix_displays_and_labels(new_config)
            self.setWindowTitle(f"{self.simulator.get_architecture_name()} Simulator")
            
            if hasattr(self, 'asm_op_header_label'): self.asm_op_header_label.setText(f"{self.simulator.get_architecture_name()} Assembly / Operation:")
                
            self.update_gui()

        except ValueError as e: QMessageBox.critical(self, "Configuration Error", f"Invalid input or configuration: {str(e)}")
        except Exception as e: QMessageBox.critical(self, "Error", f"Failed to apply config: {str(e)}")

    def on_next_step(self):
        if self.simulator.step_forward(): self.update_gui()

    def on_prev_step(self):
        if self.simulator.step_backward(): self.update_gui()

    def update_gui(self):
        if not hasattr(self, 'matrix_A_widget'): return
        
        sim_state: SimulatorState = self.simulator.get_current_gui_state()
        self.setWindowTitle(sim_state.window_title_info) 
        
        current_dtype_str = self.simulator.config.dtype_str
        self.matrix_A_widget.update_matrix(sim_state.A_data, sim_state.A_colors, dtype_str=current_dtype_str)
        self.matrix_B_widget.update_matrix(sim_state.B_data, sim_state.B_colors, dtype_str=current_dtype_str)
        self.matrix_C_widget.update_matrix(sim_state.C_data, sim_state.C_colors, is_acc_matrix=True, dtype_str=current_dtype_str)
        register_infos = self.simulator.get_register_display_info()
        
        for reg_info in register_infos:
            reg_id = reg_info['id']; widget = self.register_widgets.get(reg_id)
            if widget:
                data = sim_state.register_data.get(reg_id); colors = sim_state.register_colors.get(reg_id)
                is_acc = reg_info.get('is_accumulator', False)
                widget.update_matrix(data, colors, is_acc_matrix=is_acc, dtype_str=current_dtype_str)
        
        if hasattr(self, 'mode_label'): 
            self.mode_label.setText(sim_state.current_mode_text)
            self.asm_op_label.setText(sim_state.current_op_desc)
            self.metrics_mac_ops_label.setText(f"MAC Operations: {sim_state.metrics['mac_ops']}")
            self.metrics_elements_loaded_label.setText(f"Elements Loaded: {sim_state.metrics['elements_loaded']}")
            self.metrics_elements_stored_label.setText(f"Elements Stored: {sim_state.metrics['elements_stored']}")
            self.metrics_ci_label.setText(f"Computational Intensity (MACs/Elem IO): {sim_state.metrics['ci']:.2f}")
            self.step_info_label.setText(sim_state.current_step_info_text)
            self.prev_button.setEnabled(self.simulator.current_step_index > -1)
            self.next_button.setEnabled(self.simulator.current_step_index < self.simulator.total_steps)

    def keyPressEvent(self, event): # No changes
        if event.key() == Qt.Key_Right:
            if hasattr(self, 'next_button') and self.next_button.isEnabled(): self.on_next_step()
        elif event.key() == Qt.Key_Left:
            if hasattr(self, 'prev_button') and self.prev_button.isEnabled(): self.on_prev_step()
        else: super().keyPressEvent(event)