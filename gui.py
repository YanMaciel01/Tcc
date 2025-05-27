from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView, QSizePolicy, QFrame, QLineEdit,
    QComboBox, QMessageBox
)
from PyQt5.QtGui import QColor, QFont, QPalette, QIntValidator
from PyQt5.QtCore import Qt, QSize
import numpy as np
from simulator_base import MatrixAcceleratorSimulator, SimulatorState, SimulatorConfig
from amx_simulator import AMXSimulator 

# Standard color palette for the visualizer (same as before)
QT_CMAP_COLORS = [
    QColor(Qt.white), QColor("lightskyblue"), QColor("dodgerblue"),
    QColor("yellow"), QColor("lightcoral"), QColor("coral"), QColor("palegreen")
]

class MatrixDisplayWidget(QTableWidget): # No changes to this class from previous version
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

        cell_w = max(20, (available_width - 5) // cols if cols > 0 else 20) # -5 for scrollbar margin
        cell_h = max(20, (available_height - 5) // rows if rows > 0 else 20)

        for c_idx in range(cols): self.setColumnWidth(c_idx, cell_w)
        for r_idx in range(rows): self.setRowHeight(r_idx, cell_h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._adjust_cell_sizes_and_font()

    def update_matrix(self, data_matrix: np.ndarray, color_matrix: np.ndarray, is_acc_matrix=False, dtype_str='int8'):
        if data_matrix is None or color_matrix is None or self.rowCount() != data_matrix.shape[0] or self.columnCount() != data_matrix.shape[1]:
            # Clear the table if no data or dimensions mismatch (e.g. during reconfig)
            for r in range(self.rowCount()):
                for c in range(self.columnCount()):
                    item = self.item(r, c)
                    if item: # Check if item exists
                        item.setText("")
                        item.setBackground(QT_CMAP_COLORS[0])
            return

        font = QFont()
        current_base_font_size = self.base_font_size # Use the one calculated at init/resize
        
        fs_adj = current_base_font_size
        if data_matrix.size > 256: fs_adj = max(4, current_base_font_size - 3)
        elif data_matrix.size > 100: fs_adj = max(5, current_base_font_size - 2)
        
        # This logic needs context of which matrix it is.
        # Example: if it's a tile register and very dense
        is_tile_reg = (data_matrix.shape[0] <= 16 and data_matrix.shape[1] <= 16) # Heuristic
        if is_tile_reg and data_matrix.shape[0] * data_matrix.shape[1] > 64 and not is_acc_matrix:
             fs_adj = max(4, current_base_font_size - 4)

        font.setPointSize(fs_adj)

        for r in range(data_matrix.shape[0]):
            for c in range(data_matrix.shape[1]):
                item = self.item(r, c)
                if item is None:
                    item = QTableWidgetItem()
                    item.setTextAlignment(Qt.AlignCenter)
                    self.setItem(r,c, item)
                
                val = data_matrix[r, c]
                is_bf16_non_acc = (dtype_str == 'bf16' and not is_acc_matrix)
                if is_bf16_non_acc: val_str = f"{int(val)}"
                elif isinstance(val, (float, np.floating)): val_str = f"{val:.0f}"
                else: val_str = f"{val}"
                
                item.setText(val_str)
                item.setFont(font)
                item.setBackground(QT_CMAP_COLORS[color_matrix[r, c] % len(QT_CMAP_COLORS)])
        # self._adjust_cell_sizes_and_font() # Not strictly needed here if data content doesn't change cell size req


class MainWindow(QMainWindow):
    def __init__(self, initial_simulator: MatrixAcceleratorSimulator): # Takes initial simulator
        super().__init__()
        self.simulator = initial_simulator
        self.initial_config = initial_simulator.config # Store initial config for defaults

        self.setWindowTitle(f"{self.simulator.get_architecture_name()} Simulator")
        self.setGeometry(50, 50, 1600, 800) # Increased size slightly

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Matrix display area - this will be a QWidget that we can clear and repopulate
        self.matrix_display_area_widget = QWidget()
        self.matrix_area_layout = QGridLayout(self.matrix_display_area_widget) # Layout for the widget
        self.main_layout.addWidget(self.matrix_display_area_widget, 4)

        # Sidebar area (frame and layout)
        self.sidebar_frame = QFrame()
        self.sidebar_frame.setFrameShape(QFrame.StyledPanel)
        self.sidebar_layout = QVBoxLayout(self.sidebar_frame)
        self.sidebar_layout.setAlignment(Qt.AlignTop)
        self.main_layout.addWidget(self.sidebar_frame, 1)

        self._setup_parameter_inputs() # New method for parameter inputs
        self._rebuild_matrix_displays_and_labels(self.simulator.config) # Initial build
        self._setup_sidebar_dynamic_elements() # For mode, ASM, metrics, step, controls
        
        self._connect_signals()
        self.update_gui()

    def _rebuild_matrix_displays_and_labels(self, current_config: SimulatorConfig):
        # Clear existing matrix widgets from the layout if any
        while self.matrix_area_layout.count():
            child = self.matrix_area_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout(): # If it's a layout (our title + matrix container)
                # Recursively delete items in the sub-layout
                sub_layout = child.layout()
                while sub_layout.count():
                    sub_child = sub_layout.takeAt(0)
                    if sub_child.widget():
                        sub_child.widget().deleteLater()
                sub_layout.deleteLater()

        # Store current GUI configuration based on simulator's config
        self.current_gui_config = {
            "N_DIM": current_config.N, "M_DIM": current_config.M, "P_DIM": current_config.P,
            "tile_h": current_config.tile_h, "tile_w": current_config.tile_w,
            "input_dtype_str": current_config.dtype_str,
            "architecture_name": self.simulator.get_architecture_name()
        }
        arch_name = self.current_gui_config['architecture_name']

        # Create NEW MatrixDisplayWidgets with current_config dimensions
        self.matrix_A_widget = MatrixDisplayWidget(current_config.N, current_config.M, "Matrix A (Source)")
        self.matrix_B_widget = MatrixDisplayWidget(current_config.M, current_config.P, "Matrix B (Source)")
        self.matrix_C_widget = MatrixDisplayWidget(current_config.N, current_config.P, "Matrix C (Result)")
        
        self.reg0_widget = MatrixDisplayWidget(current_config.tile_h, current_config.tile_w, f"Reg0 ({arch_name} Tile)")
        self.reg1_widget = MatrixDisplayWidget(current_config.tile_w, current_config.tile_w, f"Reg1 ({arch_name} Tile)")
        self.reg2_acc_widget = MatrixDisplayWidget(current_config.tile_h, current_config.tile_w, f"Reg2 ({arch_name} Accumulator)")

        self._add_matrix_widget_to_layout(self.matrix_A_widget, 0, 0)
        self._add_matrix_widget_to_layout(self.matrix_B_widget, 0, 1)
        self._add_matrix_widget_to_layout(self.matrix_C_widget, 0, 2)
        self._add_matrix_widget_to_layout(self.reg0_widget, 1, 0)
        self._add_matrix_widget_to_layout(self.reg1_widget, 1, 1)
        self._add_matrix_widget_to_layout(self.reg2_acc_widget, 1, 2)

        for i in range(2): self.matrix_area_layout.setRowStretch(i, 1)
        for i in range(3): self.matrix_area_layout.setColumnStretch(i, 1)

    def _add_matrix_widget_to_layout(self, matrix_widget: MatrixDisplayWidget, r, c):
        container = QVBoxLayout() # Container for title + matrix
        container.addWidget(matrix_widget.title_label)
        container.addWidget(matrix_widget)
        self.matrix_area_layout.addLayout(container, r, c)
        
    def _create_param_input(self, label_text, default_value):
        layout = QHBoxLayout() # This is the layout we want to add to param_grid
        label = QLabel(label_text)
        line_edit = QLineEdit(str(default_value))
        line_edit.setValidator(QIntValidator(1, 1024))
        line_edit.setFixedWidth(50)
        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(line_edit)
        return layout, line_edit # Return the layout AND the line_edit

    def _setup_parameter_inputs(self):
        self.sidebar_layout.addWidget(self._create_sidebar_header("Input Parameters:"))
        
        param_grid = QGridLayout()
        
        # N, M, P
        # Store the returned layout to add it to the grid
        n_layout, self.N_input = self._create_param_input("N:", self.initial_config.N)
        m_layout, self.M_input = self._create_param_input("M:", self.initial_config.M)
        p_layout, self.P_input = self._create_param_input("P:", self.initial_config.P)
        param_grid.addLayout(n_layout, 0, 0) # Use the returned layout
        param_grid.addLayout(m_layout, 0, 1) # Use the returned layout
        param_grid.addLayout(p_layout, 0, 2) # Use the returned layout

        # Tile H, Tile W
        tile_h_layout, self.tile_h_input = self._create_param_input("Tile H:", self.initial_config.tile_h)
        tile_w_layout, self.tile_w_input = self._create_param_input("Tile W:", self.initial_config.tile_w)
        param_grid.addLayout(tile_h_layout, 1, 0) # Use the returned layout
        param_grid.addLayout(tile_w_layout, 1, 1) # Use the returned layout
        
        # Dtype
        dtype_layout = QHBoxLayout()
        dtype_label = QLabel("Data Type:")
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(["int8", "bf16"])
        self.dtype_combo.setCurrentText(self.initial_config.dtype_str)
        dtype_layout.addWidget(dtype_label)
        # dtype_layout.addStretch() # Optional: if you want combobox not to fill width
        dtype_layout.addWidget(self.dtype_combo)
        param_grid.addLayout(dtype_layout, 2, 0, 1, 2) # Span 2 columns

        self.sidebar_layout.addLayout(param_grid)

        self.apply_config_button = QPushButton("Apply Configuration & Restart")
        self.sidebar_layout.addWidget(self.apply_config_button)
        self.sidebar_layout.addSpacing(10)

    def _setup_sidebar_dynamic_elements(self):
        # Mode
        self.mode_label = self._create_sidebar_header("Mode: Initial", bold=True, color="darkblue")
        self.sidebar_layout.addWidget(self.mode_label)
        self.sidebar_layout.addSpacing(10)

        # ASM Operations
        arch_name = self.current_gui_config['architecture_name'] if hasattr(self, 'current_gui_config') else self.initial_config.N # Fallback
        self.sidebar_layout.addWidget(self._create_sidebar_header(f"{arch_name} Assembly / Operation:"))
        self.asm_op_label = QLabel("...")
        self.asm_op_label.setWordWrap(True)
        self.asm_op_label.setStyleSheet("font-family: monospace;")
        self.sidebar_layout.addWidget(self.asm_op_label)
        self.sidebar_layout.addSpacing(10)
        
        # Metrics
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

        # Step Info
        self.step_info_label = QLabel("Step: 0 / 0")
        self.sidebar_layout.addWidget(self.step_info_label)
        self.sidebar_layout.addSpacing(20)

        # Controls
        self.controls_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous (Left Arrow)")
        self.next_button = QPushButton("Next (Right Arrow)")
        self.reset_button = QPushButton("Reset Sim")
        self.controls_layout.addWidget(self.prev_button)
        self.controls_layout.addWidget(self.next_button)
        # self.controls_layout.addWidget(self.reset_button) # Reset will be handled by Apply Config for now
        self.sidebar_layout.addLayout(self.controls_layout)
        self.sidebar_layout.addStretch(1)

    def _create_sidebar_header(self, text, bold=True, color=None):
        # (Same as before)
        label = QLabel(text)
        font = label.font()
        font.setBold(bold)
        font.setPointSize(font.pointSize() + (1 if bold else 0))
        label.setFont(font)
        if color:
            label.setStyleSheet(f"color: {color};")
        return label

    def _connect_signals(self):
        self.next_button.clicked.connect(self.on_next_step)
        self.prev_button.clicked.connect(self.on_prev_step)
        self.apply_config_button.clicked.connect(self.on_apply_config)

    def on_apply_config(self):
        try:
            N = int(self.N_input.text())
            M = int(self.M_input.text())
            P = int(self.P_input.text())
            tile_h = int(self.tile_h_input.text())
            tile_w = int(self.tile_w_input.text())
            dtype_str = self.dtype_combo.currentText()

            if not (N > 0 and M > 0 and P > 0 and tile_h > 0 and tile_w > 0):
                raise ValueError("Dimensions and tile sizes must be positive.")
            if N % tile_h != 0 or M % tile_w != 0 or P % tile_w != 0: # Common divisibility
                 QMessageBox.warning(self, "Configuration Warning",
                                    "Matrix dimensions (N, M, P) should ideally be divisible by tile dimensions (Tile H, Tile W).\n"
                                    "AMX specifically requires M and P to be divisible by Tile W, and N by Tile H.\n"
                                    "Behavior might be unexpected if not perfectly divisible depending on simulator logic.")
                 # Allow to proceed with warning for now, AMXSimulator has its own checks.

            new_config = SimulatorConfig(N=N, M=M, P=P, tile_h=tile_h, tile_w=tile_w, dtype_str=dtype_str)
            
            # Re-instantiate simulator (assuming AMX for now, could be factory later)
            # TODO: Make this generic if multiple simulators are supported
            self.simulator = AMXSimulator(new_config)
            self.simulator.initialize_simulation() # This resets step_index, metrics etc.
            
            # Rebuild matrix displays based on the new config
            self._rebuild_matrix_displays_and_labels(new_config)
            
            self.setWindowTitle(f"{self.simulator.get_architecture_name()} Simulator") # Update window title if arch changes
            self.update_gui()

        except ValueError as e:
            QMessageBox.critical(self, "Configuration Error", f"Invalid input: {str(e)}")
        except Exception as e: # Catch other potential errors during reinitialization
            QMessageBox.critical(self, "Error", f"Failed to apply configuration: {str(e)}")

    def on_next_step(self):
        if self.simulator.step_forward():
            self.update_gui()

    def on_prev_step(self):
        if self.simulator.step_backward():
            self.update_gui()

    def update_gui(self):
        if not hasattr(self, 'matrix_A_widget'): # GUI not fully built yet
            return
            
        sim_state: SimulatorState = self.simulator.get_current_gui_state()

        self.setWindowTitle(sim_state.window_title_info)

        current_dtype_str = self.simulator.config.dtype_str

        self.matrix_A_widget.update_matrix(sim_state.A_data, sim_state.A_colors, dtype_str=current_dtype_str)
        self.matrix_B_widget.update_matrix(sim_state.B_data, sim_state.B_colors, dtype_str=current_dtype_str)
        self.matrix_C_widget.update_matrix(sim_state.C_data, sim_state.C_colors, is_acc_matrix=True, dtype_str=current_dtype_str)
        
        self.reg0_widget.update_matrix(sim_state.reg0_data, sim_state.reg0_colors, dtype_str=current_dtype_str)
        self.reg1_widget.update_matrix(sim_state.reg1_data, sim_state.reg1_colors, dtype_str=current_dtype_str)
        self.reg2_acc_widget.update_matrix(sim_state.reg2_acc_data, sim_state.reg2_acc_colors, is_acc_matrix=True, dtype_str=current_dtype_str)

        self.mode_label.setText(sim_state.current_mode_text)
        self.asm_op_label.setText(sim_state.current_op_desc)
        
        self.metrics_mac_ops_label.setText(f"MAC Operations: {sim_state.metrics['mac_ops']}")
        self.metrics_elements_loaded_label.setText(f"Elements Loaded: {sim_state.metrics['elements_loaded']}")
        self.metrics_elements_stored_label.setText(f"Elements Stored: {sim_state.metrics['elements_stored']}")
        self.metrics_ci_label.setText(f"Computational Intensity (MACs/Elem IO): {sim_state.metrics['ci']:.2f}")
        
        self.step_info_label.setText(sim_state.current_step_info_text)

        self.prev_button.setEnabled(self.simulator.current_step_index > -1)
        self.next_button.setEnabled(self.simulator.current_step_index < self.simulator.total_steps)
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            if self.next_button.isEnabled(): self.on_next_step()
        elif event.key() == Qt.Key_Left:
            if self.prev_button.isEnabled(): self.on_prev_step()
        else:
            super().keyPressEvent(event)