from __future__ import annotations

import time
from typing import Iterable

import numpy as np

from core import (
    BASIS_VECTORS,
    CUBE_FACES,
    CUBE_VERTICES,
    MatrixState,
    SceneState,
    UNIT_SQUARE_FACES,
    UNIT_SQUARE_VERTICES,
    analyze_matrix,
    build_scene_state,
    format_eigenvalues,
    format_linear_equations,
    format_scalar,
    get_preset_matrix,
    get_presets,
    interpolate_arrays,
    parse_linear_equations,
)

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtCore import Qt
except ImportError as exc:
    raise ImportError(
        "PyVista, pyvistaqt, and PyQt5 are required to launch the desktop visualizer. "
        "Install them with 'pip install -r requirements.txt'."
    ) from exc


REFERENCE_BASIS_COLOR = "#97a3b6"
REFERENCE_SHAPE_COLOR = "#c8d1dc"
REFERENCE_GRID_COLOR = "#58708b"
TRANSFORMED_GRID_DEFAULT = "#77b9ff"
BACKGROUND_BOTTOM = "#07111f"
BACKGROUND_TOP = "#21334b"
BASIS_COLORS = ("#ff8766", "#52ddae", "#6cb7ff")
GRID_BOUNDS_3D = (-2.5, 2.5, -2.5, 2.5, -2.5, 2.5)

TRANSFORMATION_STYLES = {
    "Identity": {"fill": "#6faaf7", "edge": "#dcecff", "grid": "#5f96d9"},
    "Diagonal scaling": {"fill": "#f5ad4a", "edge": "#ffe4b3", "grid": "#d89d43"},
    "Reflection": {"fill": "#f37d6a", "edge": "#ffd4ca", "grid": "#df6f61"},
    "Shear": {"fill": "#57d2b8", "edge": "#d2fff3", "grid": "#47bca4"},
    "Rotation": {"fill": "#69b7ff", "edge": "#def1ff", "grid": "#4d98db"},
    "Rotation about x": {"fill": "#69b7ff", "edge": "#def1ff", "grid": "#4d98db"},
    "Rotation about z": {"fill": "#69b7ff", "edge": "#def1ff", "grid": "#4d98db"},
    "Project onto X": {"fill": "#b07fff", "edge": "#e8d8ff", "grid": "#9a68e8"},
    "Project onto Y": {"fill": "#b07fff", "edge": "#e8d8ff", "grid": "#9a68e8"},
    "Project onto y=x": {"fill": "#b07fff", "edge": "#e8d8ff", "grid": "#9a68e8"},
    "Project onto XY": {"fill": "#b07fff", "edge": "#e8d8ff", "grid": "#9a68e8"},
    "Project onto XZ": {"fill": "#b07fff", "edge": "#e8d8ff", "grid": "#9a68e8"},
    "Project onto YZ": {"fill": "#b07fff", "edge": "#e8d8ff", "grid": "#9a68e8"},
    "Custom matrix": {"fill": "#ff9f5e", "edge": "#ffe0c4", "grid": "#dd8a51"},
}


# ---------------------------------------------------------------------------
# Blender-style camera interactor
# Middle-mouse  → orbit   Shift+Middle → pan   Scroll → zoom
# Left-mouse also orbits so touchpad users are not blocked.
# ---------------------------------------------------------------------------
try:
    import vtk as _vtk

    class _BlenderCameraInteractor(_vtk.vtkInteractorStyleTrackballCamera):
        def OnMiddleButtonDown(self) -> None:
            iren = self.GetInteractor()
            self.FindPokedRenderer(*iren.GetEventPosition())
            self.StartPan() if iren.GetShiftKey() else self.StartRotate()

        def OnMiddleButtonUp(self) -> None:
            state = self.GetState()
            if state == 2:    self.EndPan()      # VTKIS_PAN
            elif state == 1:  self.EndRotate()   # VTKIS_ROTATE

        def OnRightButtonDown(self) -> None:
            pass  # suppress right-drag; scroll zooms

        def OnRightButtonUp(self) -> None:
            pass

except Exception:
    _BlenderCameraInteractor = None  # type: ignore[assignment,misc]
# ---------------------------------------------------------------------------


class LinearTransformationWindow(QtWidgets.QMainWindow):
    animation_duration_ms = 950
    animation_interval_ms = 16

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TransformLab \u2014 Linear Transformation Visualizer")
        self.resize(1480, 920)

        self._current_dimension = 2
        self._current_preset_name = "Identity"
        self._presets_by_dimension = {2: get_presets(2), 3: get_presets(3)}

        self._scene_state: SceneState | None = None
        self._matrix_state: MatrixState | None = None
        self._animation_started_at = 0.0
        self._last_render_progress = 1.0
        self._default_views: dict[int, tuple[tuple, float]] = {}
        self._current_eigen_label_names: list[str] = []
        self._equations_dirty = False
        self._updating_equation_inputs = False
        self._updating_matrix_inputs = False

        self._composition_base: np.ndarray | None = None
        self._composition_base_name: str = ""
        self._handle_widgets: list[object] = []

        self._matrix_inputs: list[list[QtWidgets.QDoubleSpinBox]] = []
        self._equation_inputs: list[QtWidgets.QLineEdit] = []
        self._preset_buttons: dict[str, QtWidgets.QPushButton] = {}

        self._transformed_shape_mesh: pv.PolyData | None = None
        self._transformed_shape_actor: object | None = None
        self._transformed_grid_mesh: pv.PolyData | None = None
        self._transformed_grid_actor: object | None = None
        self._reference_grid_points: np.ndarray | None = None
        self._transformed_grid_points: np.ndarray | None = None
        self._original_grid_points: np.ndarray | None = None
        self._reference_basis_actors: list[object] = []
        self._transformed_basis_meshes: list[pv.PolyData] = []
        self._transformed_basis_actors: list[object] = []
        self._eigen_arrow_meshes: list[pv.PolyData] = []
        self._eigen_arrow_actors: list[object] = []

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick_animation)

        self._matrix_group: QtWidgets.QGroupBox
        self._equation_group: QtWidgets.QGroupBox
        self._size_combo: QtWidgets.QComboBox
        self._preset_combo: QtWidgets.QComboBox
        self._preset_buttons_layout: QtWidgets.QGridLayout
        self._show_eigen_checkbox: QtWidgets.QCheckBox
        self._mode_label: QtWidgets.QLabel
        self._transform_label: QtWidgets.QLabel
        self._determinant_label: QtWidgets.QLabel
        self._interpretation_label: QtWidgets.QLabel
        self._eigenvalue_label: QtWidgets.QLabel
        self._status_label: QtWidgets.QLabel
        self._trace_rank_label: QtWidgets.QLabel
        self._compose_label: QtWidgets.QLabel

        self._build_ui()
        self._rebuild_scene_for_dimension()
        self._refresh_dimension_ui()
        self._apply_preset("Identity", restore_camera=True)

    def _build_ui(self) -> None:
        self._apply_styles()

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        viewer_frame = QtWidgets.QFrame(central)
        viewer_frame.setObjectName("viewerFrame")
        viewer_layout = QtWidgets.QVBoxLayout(viewer_frame)
        viewer_layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(viewer_frame)
        viewer_widget = getattr(self.plotter, "interactor", self.plotter)
        viewer_layout.addWidget(viewer_widget)
        root_layout.addWidget(viewer_frame, stretch=4)

        panel = QtWidgets.QFrame(central)
        panel.setObjectName("controlPanel")
        panel.setMinimumWidth(380)
        panel.setMaximumWidth(450)
        panel_shell_layout = QtWidgets.QVBoxLayout(panel)
        panel_shell_layout.setContentsMargins(0, 0, 0, 0)
        panel_shell_layout.setSpacing(0)
        root_layout.addWidget(panel, stretch=0)

        panel_scroll = QtWidgets.QScrollArea(panel)
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        panel_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        panel_shell_layout.addWidget(panel_scroll)

        panel_content = QtWidgets.QWidget(panel_scroll)
        panel_scroll.setWidget(panel_content)
        panel_layout = QtWidgets.QVBoxLayout(panel_content)
        panel_layout.setContentsMargins(18, 18, 18, 18)
        panel_layout.setSpacing(14)

        header_card = QtWidgets.QFrame(panel_content)
        header_card.setObjectName("headerCard")
        _hdr = QtWidgets.QVBoxLayout(header_card)
        _hdr.setContentsMargins(14, 12, 14, 12)
        _hdr.setSpacing(3)
        _app_title = QtWidgets.QLabel("TransformLab", header_card)
        _app_title.setObjectName("appTitle")
        _app_sub = QtWidgets.QLabel("Linear Transformation Visualizer", header_card)
        _app_sub.setObjectName("appSubtitle")
        _course = QtWidgets.QLabel("Linear Algebra  \u00b7  Mini-Project", header_card)
        _course.setObjectName("courseTag")
        _hdr.addWidget(_app_title)
        _hdr.addWidget(_app_sub)
        _hdr.addWidget(_course)
        panel_layout.addWidget(header_card)

        size_group = QtWidgets.QGroupBox("Matrix Size", panel)
        size_layout = QtWidgets.QVBoxLayout(size_group)
        self._size_combo = QtWidgets.QComboBox(size_group)
        self._size_combo.addItem("2 x 2 (XY plane)", 2)
        self._size_combo.addItem("3 x 3 (3D extension)", 3)
        self._size_combo.currentIndexChanged.connect(self._change_dimension)
        size_layout.addWidget(self._size_combo)
        panel_layout.addWidget(size_group)

        preset_group = QtWidgets.QGroupBox("Choose a Transformation", panel)
        preset_layout = QtWidgets.QVBoxLayout(preset_group)
        preset_hint = QtWidgets.QLabel(
            "Select a preset to load its matrix and animate it instantly.",
            preset_group,
        )
        preset_hint.setObjectName("sectionHint")
        preset_hint.setWordWrap(True)
        preset_layout.addWidget(preset_hint)

        self._preset_combo = QtWidgets.QComboBox(preset_group)
        self._preset_combo.currentTextChanged.connect(self._load_selected_preset)
        preset_layout.addWidget(self._preset_combo)

        self._preset_buttons_layout = QtWidgets.QGridLayout()
        self._preset_buttons_layout.setHorizontalSpacing(8)
        self._preset_buttons_layout.setVerticalSpacing(8)
        preset_layout.addLayout(self._preset_buttons_layout)
        panel_layout.addWidget(preset_group)

        self._matrix_group = QtWidgets.QGroupBox("Custom Matrix", panel)
        matrix_layout = QtWidgets.QGridLayout(self._matrix_group)
        matrix_layout.setHorizontalSpacing(8)
        matrix_layout.setVerticalSpacing(8)

        for row_index in range(3):
            row_inputs: list[QtWidgets.QDoubleSpinBox] = []
            for column_index in range(3):
                input_box = QtWidgets.QDoubleSpinBox(self._matrix_group)
                input_box.setRange(-20.0, 20.0)
                input_box.setDecimals(3)
                input_box.setSingleStep(0.25)
                input_box.setKeyboardTracking(False)
                input_box.setAlignment(Qt.AlignCenter)
                input_box.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
                input_box.setMinimumHeight(34)
                input_box.valueChanged.connect(self._handle_matrix_input_changed)
                matrix_layout.addWidget(input_box, row_index, column_index)
                row_inputs.append(input_box)
            self._matrix_inputs.append(row_inputs)
        panel_layout.addWidget(self._matrix_group)

        manual_hint = QtWidgets.QLabel(
            "Editing entries here creates a custom transformation.",
            panel,
        )
        manual_hint.setObjectName("sectionHint")
        manual_hint.setWordWrap(True)
        panel_layout.addWidget(manual_hint)

        self._equation_group = QtWidgets.QGroupBox("Linear Equations", panel)
        equation_layout = QtWidgets.QVBoxLayout(self._equation_group)
        equation_hint = QtWidgets.QLabel(
            "Enter rules like  x\u2019 = 2x + y,  then click Animate.",
            self._equation_group,
        )
        equation_hint.setObjectName("sectionHint")
        equation_hint.setWordWrap(True)
        equation_layout.addWidget(equation_hint)

        for _index in range(3):
            equation_input = QtWidgets.QLineEdit(self._equation_group)
            equation_input.setMinimumHeight(34)
            equation_input.textEdited.connect(self._mark_equations_dirty)
            equation_layout.addWidget(equation_input)
            self._equation_inputs.append(equation_input)

        use_equations_button = QtWidgets.QPushButton("Use Equations", self._equation_group)
        use_equations_button.clicked.connect(self.apply_equations)
        equation_layout.addWidget(use_equations_button)
        panel_layout.addWidget(self._equation_group)

        animate_row = QtWidgets.QHBoxLayout()
        animate_row.setSpacing(8)
        animate_button = QtWidgets.QPushButton("Animate Transformation", panel)
        animate_button.setObjectName("animateButton")
        animate_button.setToolTip("Animate the transformation  [Space]")
        animate_button.clicked.connect(self.animate_matrix)
        replay_button = QtWidgets.QPushButton("\u21ba", panel)
        replay_button.setObjectName("replayButton")
        replay_button.setFixedWidth(46)
        replay_button.setToolTip("Replay animation")
        replay_button.clicked.connect(self.animate_matrix)
        animate_row.addWidget(animate_button, 1)
        animate_row.addWidget(replay_button)
        panel_layout.addLayout(animate_row)

        secondary_row = QtWidgets.QHBoxLayout()
        secondary_row.setSpacing(8)
        reset_button = QtWidgets.QPushButton("Reset All", panel)
        reset_button.clicked.connect(self.reset_matrix)
        home_button = QtWidgets.QPushButton("Home View", panel)
        home_button.clicked.connect(self.restore_view)
        secondary_row.addWidget(reset_button)
        secondary_row.addWidget(home_button)
        panel_layout.addLayout(secondary_row)

        self._show_eigen_checkbox = QtWidgets.QCheckBox("Show Eigenvectors", panel)
        self._show_eigen_checkbox.setChecked(False)
        self._show_eigen_checkbox.toggled.connect(self._refresh_scene)
        panel_layout.addWidget(self._show_eigen_checkbox)

        summary_group = QtWidgets.QGroupBox("Transformation Summary", panel)
        summary_layout = QtWidgets.QVBoxLayout(summary_group)
        self._mode_label = QtWidgets.QLabel(summary_group)
        self._mode_label.setObjectName("readout")
        self._transform_label = QtWidgets.QLabel(summary_group)
        self._transform_label.setObjectName("readout")
        self._determinant_label = QtWidgets.QLabel(summary_group)
        self._determinant_label.setObjectName("readout")
        self._trace_rank_label = QtWidgets.QLabel(summary_group)
        self._trace_rank_label.setObjectName("readout")
        self._interpretation_label = QtWidgets.QLabel(summary_group)
        self._interpretation_label.setObjectName("status")
        self._interpretation_label.setWordWrap(True)
        summary_layout.addWidget(self._mode_label)
        summary_layout.addWidget(self._transform_label)
        summary_layout.addWidget(self._determinant_label)
        summary_layout.addWidget(self._trace_rank_label)
        summary_layout.addWidget(self._interpretation_label)
        panel_layout.addWidget(summary_group)

        advanced_group = QtWidgets.QGroupBox("Eigenanalysis", panel)
        advanced_layout = QtWidgets.QVBoxLayout(advanced_group)
        advanced_hint = QtWidgets.QLabel(
            "Real eigenvalues scale their eigenvectors under the transformation. Enable the overlay to visualize them.",
            advanced_group,
        )
        advanced_hint.setObjectName("sectionHint")
        advanced_hint.setWordWrap(True)
        self._eigenvalue_label = QtWidgets.QLabel(advanced_group)
        self._eigenvalue_label.setObjectName("readout")
        self._eigenvalue_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._eigenvalue_label.setWordWrap(True)
        self._status_label = QtWidgets.QLabel(advanced_group)
        self._status_label.setObjectName("status")
        self._status_label.setWordWrap(True)
        advanced_layout.addWidget(advanced_hint)
        advanced_layout.addWidget(self._eigenvalue_label)
        advanced_layout.addWidget(self._status_label)
        panel_layout.addWidget(advanced_group)

        compose_group = QtWidgets.QGroupBox("Composition  (B \u2218 A)", panel)
        compose_layout = QtWidgets.QVBoxLayout(compose_group)
        compose_hint = QtWidgets.QLabel(
            "Lock A, change to B, then Animate to see B\u2218A. "
            "Swap the order to demonstrate non-commutativity.",
            compose_group,
        )
        compose_hint.setObjectName("sectionHint")
        compose_hint.setWordWrap(True)
        compose_layout.addWidget(compose_hint)
        compose_btn_row = QtWidgets.QHBoxLayout()
        compose_btn_row.setSpacing(8)
        lock_a_btn = QtWidgets.QPushButton("Lock as A", compose_group)
        lock_a_btn.clicked.connect(self._set_composition_base)
        clear_a_btn = QtWidgets.QPushButton("Clear A", compose_group)
        clear_a_btn.clicked.connect(self._clear_composition_base)
        compose_btn_row.addWidget(lock_a_btn)
        compose_btn_row.addWidget(clear_a_btn)
        compose_layout.addLayout(compose_btn_row)
        self._compose_label = QtWidgets.QLabel("A  =  not set", compose_group)
        self._compose_label.setObjectName("readout")
        compose_layout.addWidget(self._compose_label)
        panel_layout.addWidget(compose_group)

        _footer = QtWidgets.QLabel(
            "Linear Algebra  \u00b7  Mini-Project  \u00b7  TransformLab", panel_content
        )
        _footer.setObjectName("footer")
        _footer.setAlignment(Qt.AlignCenter)
        panel_layout.addStretch(1)
        panel_layout.addWidget(_footer)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background: #080f1c;
                color: #dce8f5;
                font-family: "Segoe UI", "Inter", sans-serif;
                font-size: 13px;
            }
            QFrame#controlPanel {
                background: #090e1a;
                border: 1px solid #182a40;
                border-radius: 14px;
            }
            QFrame#viewerFrame {
                border: 1px solid #182a40;
                border-radius: 14px;
            }
            QFrame#headerCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0d2244, stop:1 #091830);
                border-radius: 10px;
                border: 1px solid #1f3e6a;
            }
            QLabel#appTitle {
                font-size: 24px;
                font-weight: 800;
                color: #6ab4ff;
            }
            QLabel#appSubtitle {
                font-size: 12px;
                color: #3d6a9a;
                font-weight: 500;
            }
            QLabel#courseTag {
                font-size: 10px;
                color: #2a4a66;
                font-weight: 400;
            }
            QGroupBox {
                background: #080f1c;
                border: 1px solid #182a40;
                border-top: 2px solid #1e3f6e;
                border-radius: 10px;
                margin-top: 12px;
                padding: 14px 8px 8px 8px;
                font-weight: 600;
                font-size: 11px;
                color: #4a7aaa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                top: -1px;
                padding: 1px 5px;
                background: #080f1c;
            }
            QLabel#sectionHint {
                color: #344d65;
                font-size: 11px;
            }
            QLabel#readout {
                font-family: "Consolas", "Cascadia Code", monospace;
                font-size: 12px;
                color: #8ab0d0;
                background: #060c18;
                border-radius: 6px;
                padding: 5px 8px;
                border: 1px solid #101e32;
            }
            QLabel#status {
                color: #c8a040;
                font-size: 11px;
            }
            QLabel#footer {
                color: #1e3048;
                font-size: 10px;
                font-style: italic;
            }
            QPushButton#animateButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e5caa, stop:1 #143878);
                border: 1px solid #3a78c8;
                border-radius: 10px;
                padding: 11px 18px;
                font-size: 14px;
                font-weight: 700;
                color: #c4dcff;
                min-height: 44px;
            }
            QPushButton#animateButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2468bc, stop:1 #184488);
                border-color: #4e90dc;
            }
            QPushButton#animateButton:pressed {
                background: #102c60;
            }
            QPushButton {
                background: #0c1929;
                border: 1px solid #182a40;
                border-radius: 8px;
                padding: 7px 12px;
                font-weight: 600;
                min-height: 30px;
                color: #8ab0d0;
            }
            QPushButton:hover {
                background: #10203a;
                border-color: #223c58;
                color: #b0ccec;
            }
            QPushButton:pressed {
                background: #080f1c;
            }
            QPushButton[presetButton="true"] {
                text-align: left;
                padding: 7px 10px;
                font-size: 12px;
                background: #09131f;
                border: 1px solid #14202e;
                color: #6a90b0;
            }
            QPushButton[presetButton="true"]:hover {
                background: #0e1e30;
                border-color: #223c58;
                color: #9ab8d8;
            }
            QPushButton[presetButton="true"]:checked {
                background: #0e2640;
                border: 1px solid #2e6aa0;
                color: #60aaff;
                font-weight: 700;
            }
            QComboBox {
                background: #0c1929;
                border: 1px solid #182a40;
                border-radius: 8px;
                padding: 6px 8px;
                min-height: 30px;
                color: #8ab0d0;
            }
            QComboBox:hover {
                border-color: #223c58;
            }
            QComboBox::drop-down {
                border: none;
                width: 22px;
            }
            QComboBox QAbstractItemView {
                background: #0c1929;
                border: 1px solid #182a40;
                selection-background-color: #142030;
                outline: none;
            }
            QDoubleSpinBox {
                background: #0c1929;
                border: 1px solid #182a40;
                border-radius: 7px;
                padding: 5px 8px;
                min-height: 30px;
                color: #8ab0d0;
                font-family: "Consolas", monospace;
            }
            QDoubleSpinBox:focus {
                border-color: #2e60a0;
            }
            QLineEdit {
                background: #0c1929;
                border: 1px solid #182a40;
                border-radius: 7px;
                padding: 5px 8px;
                min-height: 30px;
                color: #8ab0d0;
                font-family: "Consolas", monospace;
            }
            QLineEdit:focus {
                border-color: #2e60a0;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #080f1c;
                width: 5px;
                border-radius: 2px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #182a40;
                border-radius: 2px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover {
                background: #223c58;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            QCheckBox {
                font-weight: 600;
                color: #6a90b0;
                spacing: 8px;
            }
            QCheckBox:hover {
                color: #9ab8d8;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
                border-radius: 4px;
                border: 1px solid #182a40;
                background: #0c1929;
            }
            QCheckBox::indicator:hover {
                border-color: #223c58;
            }
            QCheckBox::indicator:checked {
                background: #1e5caa;
                border-color: #3a78c8;
            }
            QPushButton#replayButton {
                font-size: 18px;
                padding: 0;
                background: #0c1929;
                border: 1px solid #182a40;
                border-radius: 8px;
                color: #4a8ac8;
                min-height: 44px;
            }
            QPushButton#replayButton:hover {
                background: #10203a;
                border-color: #3a78c8;
                color: #6ab4ff;
            }
            QPushButton#replayButton:pressed {
                background: #080f1c;
            }
            """
        )

    def _rebuild_scene_for_dimension(self) -> None:
        self._clear_eigen_labels()
        self.plotter.clear()
        self._reset_dynamic_handles()

        self.plotter.set_background(BACKGROUND_BOTTOM, top=BACKGROUND_TOP)
        self.plotter.enable_parallel_projection()
        try:
            self.plotter.enable_anti_aliasing()
        except TypeError:
            self.plotter.enable_anti_aliasing("ssaa")
        except AttributeError:
            pass

        self.plotter.add_axes(line_width=2, labels_off=False)
        self._add_lights()

        if self._current_dimension == 2:
            self._configure_2d_scene()
        else:
            self._configure_3d_scene()

        self._capture_default_view()
        if self._current_dimension == 3 and _BlenderCameraInteractor is not None:
            try:
                self.plotter.iren.SetInteractorStyle(_BlenderCameraInteractor())
            except Exception:
                pass
        self.plotter.render()

    def _reset_dynamic_handles(self) -> None:
        self._transformed_shape_mesh = None
        self._transformed_shape_actor = None
        self._transformed_grid_mesh = None
        self._transformed_grid_actor = None
        self._reference_grid_points = None
        self._transformed_grid_points = None
        self._original_grid_points = None
        self._reference_basis_actors.clear()
        self._transformed_basis_meshes.clear()
        self._transformed_basis_actors.clear()
        self._eigen_arrow_meshes.clear()
        self._eigen_arrow_actors.clear()

    def _add_lights(self) -> None:
        key_light = pv.Light(position=(4.5, 3.5, 5.0), focal_point=(0.0, 0.0, 0.0), intensity=0.9)
        fill_light = pv.Light(position=(-3.5, -2.5, 1.8), focal_point=(0.0, 0.0, 0.0), intensity=0.45)
        self.plotter.add_light(key_light)
        self.plotter.add_light(fill_light)

    def _configure_2d_scene(self) -> None:
        plane = pv.Plane(center=(0.0, 0.0, -0.02), direction=(0.0, 0.0, 1.0), i_size=5.2, j_size=5.2)
        self.plotter.add_mesh(
            plane,
            color="#0f2238",
            opacity=0.58,
            smooth_shading=True,
            reset_camera=False,
        )

        grid_points, grid_lines = self._build_plane_lattice(extent=2.0, step=0.5)
        self._original_grid_points = grid_points.copy()   # never mutated — used as transform source
        self._reference_grid_points = grid_points.copy()  # animation start (overwritten per-animate)
        self._transformed_grid_points = grid_points.copy()

        self.plotter.add_mesh(
            pv.PolyData(grid_points, lines=grid_lines),
            color=REFERENCE_GRID_COLOR,
            line_width=1,
            opacity=0.38,
            render_lines_as_tubes=True,
            reset_camera=False,
        )

        self._transformed_grid_mesh = pv.PolyData(grid_points.copy(), lines=grid_lines)
        self._transformed_grid_actor = self.plotter.add_mesh(
            self._transformed_grid_mesh,
            color=TRANSFORMED_GRID_DEFAULT,
            line_width=2,
            opacity=0.9,
            render_lines_as_tubes=True,
            reset_camera=False,
        )

        self.plotter.add_mesh(
            self._build_shape_mesh(UNIT_SQUARE_VERTICES.copy(), 2),
            name="reference_shape",
            style="wireframe",
            color=REFERENCE_SHAPE_COLOR,
            line_width=2.2,
            opacity=0.95,
            reset_camera=False,
        )

        self._transformed_shape_mesh = self._build_shape_mesh(UNIT_SQUARE_VERTICES.copy(), 2)
        self._transformed_shape_actor = self.plotter.add_mesh(
            self._transformed_shape_mesh,
            color=TRANSFORMATION_STYLES["Identity"]["fill"],
            opacity=0.52,
            specular=0.22,
            smooth_shading=True,
            show_edges=True,
            edge_color=TRANSFORMATION_STYLES["Identity"]["edge"],
            line_width=2.0,
            reset_camera=False,
        )

        self._add_reference_basis(visible_count=2)
        self._add_transformed_basis()
        self._add_eigen_placeholders()

        self.plotter.view_xy()
        self.plotter.camera.zoom(1.5)

    def _configure_3d_scene(self) -> None:
        self.plotter.show_grid(
            bounds=GRID_BOUNDS_3D,
            color=REFERENCE_GRID_COLOR,
            xtitle="X",
            ytitle="Y",
            ztitle="Z",
        )

        self.plotter.add_mesh(
            self._build_shape_mesh(CUBE_VERTICES.copy(), 3),
            name="reference_shape",
            style="wireframe",
            color=REFERENCE_SHAPE_COLOR,
            line_width=1.7,
            opacity=0.72,
            reset_camera=False,
        )

        self._transformed_shape_mesh = self._build_shape_mesh(CUBE_VERTICES.copy(), 3)
        self._transformed_shape_actor = self.plotter.add_mesh(
            self._transformed_shape_mesh,
            color=TRANSFORMATION_STYLES["Identity"]["fill"],
            opacity=0.42,
            specular=0.28,
            smooth_shading=True,
            show_edges=True,
            edge_color=TRANSFORMATION_STYLES["Identity"]["edge"],
            line_width=1.7,
            reset_camera=False,
        )

        self._add_reference_basis(visible_count=3)
        self._add_transformed_basis()
        self._add_eigen_placeholders()

        self.plotter.view_isometric()
        self.plotter.camera.zoom(1.1)

    def _add_reference_basis(self, visible_count: int) -> None:
        for index, vector in enumerate(BASIS_VECTORS):
            actor = self.plotter.add_mesh(
                self._build_arrow_mesh(vector, radius=0.024),
                name=f"reference_basis_{index}",
                color=REFERENCE_BASIS_COLOR,
                opacity=0.35,
                smooth_shading=True,
                reset_camera=False,
            )
            actor.SetVisibility(index < visible_count)
            self._reference_basis_actors.append(actor)

    def _add_transformed_basis(self) -> None:
        for index, vector in enumerate(BASIS_VECTORS):
            mesh = self._build_arrow_mesh(vector, radius=0.032)
            actor = self.plotter.add_mesh(
                mesh,
                name=f"transformed_basis_{index}",
                color=BASIS_COLORS[index],
                smooth_shading=True,
                reset_camera=False,
            )
            self._transformed_basis_meshes.append(mesh)
            self._transformed_basis_actors.append(actor)

    def _add_eigen_placeholders(self) -> None:
        for index in range(3):
            mesh = pv.Sphere(radius=0.02, center=(0.0, 0.0, 0.0))
            actor = self.plotter.add_mesh(
                mesh,
                name=f"eigen_arrow_{index}",
                color="#ffffff",
                smooth_shading=True,
                reset_camera=False,
            )
            actor.SetVisibility(False)
            self._eigen_arrow_meshes.append(mesh)
            self._eigen_arrow_actors.append(actor)

    def _capture_default_view(self) -> None:
        camera_position = self.plotter.camera_position
        default_camera = tuple(tuple(point) for point in camera_position)
        parallel_scale = float(self.plotter.camera.parallel_scale)
        self._default_views[self._current_dimension] = (default_camera, parallel_scale)

    def _restore_default_view(self) -> None:
        if self._current_dimension not in self._default_views:
            return
        camera_position, parallel_scale = self._default_views[self._current_dimension]
        self.plotter.camera_position = camera_position
        self.plotter.camera.parallel_scale = parallel_scale
        try:
            self.plotter.reset_camera_clipping_range()
        except AttributeError:
            pass
        self.plotter.render()

    def _current_presets(self) -> dict[str, np.ndarray]:
        return self._presets_by_dimension[self._current_dimension]

    def _change_dimension(self, _index: int | None = None) -> None:
        selected_dimension = int(self._size_combo.currentData())
        if selected_dimension == self._current_dimension:
            return

        self._timer.stop()
        self._current_dimension = selected_dimension
        self._current_preset_name = "Identity"
        self._clear_composition_base()
        self._rebuild_scene_for_dimension()
        self._refresh_dimension_ui()
        self._apply_preset("Identity", restore_camera=True)

    def _refresh_dimension_ui(self) -> None:
        if self._size_combo.currentData() != self._current_dimension:
            blocker = QtCore.QSignalBlocker(self._size_combo)
            self._size_combo.setCurrentIndex(0 if self._current_dimension == 2 else 1)
            del blocker

        self._matrix_group.setTitle(
            "Custom 2 x 2 Matrix" if self._current_dimension == 2 else "Custom 3 x 3 Matrix"
        )
        self._equation_group.setTitle(
            "2 x 2 Linear Equations" if self._current_dimension == 2 else "3 x 3 Linear Equations"
        )
        self._sync_matrix_input_visibility()
        self._sync_equation_input_visibility()
        self._rebuild_preset_controls()

    def _sync_matrix_input_visibility(self) -> None:
        for row_index, row_inputs in enumerate(self._matrix_inputs):
            for column_index, input_box in enumerate(row_inputs):
                visible = row_index < self._current_dimension and column_index < self._current_dimension
                input_box.setVisible(visible)
                input_box.setEnabled(visible)
                if not visible:
                    blocker = QtCore.QSignalBlocker(input_box)
                    input_box.setValue(1.0 if row_index == column_index else 0.0)
                    del blocker

    def _sync_equation_input_visibility(self) -> None:
        placeholders = {
            2: ("x' = 2x + y", "y' = -x + 3y"),
            3: ("x' = 2x + y", "y' = x - z", "z' = z"),
        }[self._current_dimension]
        for index, equation_input in enumerate(self._equation_inputs):
            visible = index < self._current_dimension
            equation_input.setVisible(visible)
            equation_input.setEnabled(visible)
            equation_input.setPlaceholderText(placeholders[index] if visible else "")
            if not visible:
                blocker = QtCore.QSignalBlocker(equation_input)
                equation_input.clear()
                del blocker

    def _mark_equations_dirty(self, _text: str) -> None:
        if not self._updating_equation_inputs:
            self._equations_dirty = True

    def _handle_matrix_input_changed(self, _value: float) -> None:
        if self._updating_matrix_inputs:
            return
        self._set_equation_inputs(self._matrix_from_inputs())

    def _visible_equation_texts(self) -> list[str]:
        return [self._equation_inputs[index].text() for index in range(self._current_dimension)]

    def _set_equation_inputs(self, matrix: np.ndarray) -> None:
        equations = format_linear_equations(matrix)
        self._updating_equation_inputs = True
        for index, equation_input in enumerate(self._equation_inputs):
            blocker = QtCore.QSignalBlocker(equation_input)
            if index < len(equations):
                equation_input.setText(equations[index])
            else:
                equation_input.clear()
            del blocker
        self._updating_equation_inputs = False
        self._equations_dirty = False

    def _rebuild_preset_controls(self) -> None:
        presets = self._current_presets()

        combo_blocker = QtCore.QSignalBlocker(self._preset_combo)
        self._preset_combo.clear()
        self._preset_combo.addItems(presets.keys())
        self._preset_combo.setCurrentText(self._current_preset_name if self._current_preset_name in presets else "Identity")
        del combo_blocker

        while self._preset_buttons_layout.count():
            item = self._preset_buttons_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._preset_buttons.clear()
        for index, preset_name in enumerate(presets.keys()):
            button = QtWidgets.QPushButton(preset_name)
            button.setCheckable(True)
            button.setProperty("presetButton", True)
            button.setMinimumHeight(34)
            button.clicked.connect(lambda _checked=False, name=preset_name: self._apply_preset(name))
            self._preset_buttons_layout.addWidget(button, index // 2, index % 2)
            self._preset_buttons[preset_name] = button

        self._preset_buttons_layout.setColumnStretch(0, 1)
        self._preset_buttons_layout.setColumnStretch(1, 1)
        self._sync_preset_button_state()

    def _sync_preset_button_state(self) -> None:
        for preset_name, button in self._preset_buttons.items():
            button.setChecked(preset_name == self._current_preset_name)
            button.style().unpolish(button)
            button.style().polish(button)

    def _matrix_from_inputs(self) -> np.ndarray:
        size = self._current_dimension
        values = [
            [self._matrix_inputs[row_index][column_index].value() for column_index in range(size)]
            for row_index in range(size)
        ]
        return np.array(values, dtype=float)

    def _set_matrix_inputs(self, matrix: np.ndarray) -> None:
        full_matrix = np.eye(3, dtype=float)
        full_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
        self._updating_matrix_inputs = True
        for row_index, row in enumerate(full_matrix):
            for column_index, value in enumerate(row):
                blocker = QtCore.QSignalBlocker(self._matrix_inputs[row_index][column_index])
                self._matrix_inputs[row_index][column_index].setValue(float(value))
                del blocker
        self._updating_matrix_inputs = False
        self._set_equation_inputs(matrix)

    def _load_selected_preset(self, preset_name: str) -> None:
        if preset_name:
            self._apply_preset(preset_name)

    def _resolve_matrix_for_action(self) -> np.ndarray | None:
        equation_texts = self._visible_equation_texts()
        if self._equations_dirty and any(text.strip() for text in equation_texts):
            matrix, errors = parse_linear_equations(equation_texts, dimension=self._current_dimension)
            if errors:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Equation Input Error",
                    "\n".join(errors),
                )
                return None
            self._set_matrix_inputs(matrix)
        else:
            matrix = self._matrix_from_inputs()

        if (
            self._composition_base is not None
            and self._composition_base.shape == matrix.shape
        ):
            matrix = matrix @ self._composition_base  # B∘A: apply A first, then B

        return matrix

    def apply_equations(self) -> None:
        matrix, errors = parse_linear_equations(
            self._visible_equation_texts(),
            dimension=self._current_dimension,
        )
        if errors:
            QtWidgets.QMessageBox.warning(
                self,
                "Equation Input Error",
                "\n".join(errors),
            )
            return
        self._set_matrix_inputs(matrix)
        self.apply_matrix()

    def _apply_preset(
        self,
        preset_name: str,
        *,
        restore_camera: bool = False,
    ) -> None:
        if preset_name not in self._current_presets():
            return

        self._current_preset_name = preset_name
        self._set_matrix_inputs(get_preset_matrix(preset_name, self._current_dimension))

        blocker = QtCore.QSignalBlocker(self._preset_combo)
        self._preset_combo.setCurrentText(preset_name)
        del blocker

        self._sync_preset_button_state()
        if restore_camera:
            self.apply_matrix(restore_camera=True)   # initial load — no animation needed
        # User-initiated preset clicks only load values into the inputs.
        # The user clicks Animate (or Space) when ready — preserving any edits they make.

    def reset_matrix(self) -> None:
        self._timer.stop()
        self._show_eigen_checkbox.setChecked(False)
        self._clear_composition_base()
        self._apply_preset("Identity", restore_camera=True)

    def restore_view(self) -> None:
        self._restore_default_view()

    # ------------------------------------------------------------------
    # Composition helpers
    # ------------------------------------------------------------------

    def _set_composition_base(self) -> None:
        self._composition_base = self._matrix_from_inputs().copy()
        name = self._current_preset_name or "Custom"
        self._composition_base_name = name
        self._compose_label.setText(f"A  =  {name}")

    def _clear_composition_base(self) -> None:
        self._composition_base = None
        self._composition_base_name = ""
        if hasattr(self, "_compose_label"):
            self._compose_label.setText("A  =  not set")

    # ------------------------------------------------------------------
    # Interactive drag handles — drag a basis-vector tip to reshape live
    # ------------------------------------------------------------------

    def _clear_handles(self) -> None:
        """Remove all sphere-widget handles from the scene."""
        try:
            self.plotter.clear_sphere_widgets()
        except AttributeError:
            for w in self._handle_widgets:
                try:
                    w.Off()
                except Exception:
                    pass
        self._handle_widgets.clear()

    def _setup_interactive_handles(self, matrix: np.ndarray) -> None:
        """Place draggable sphere handles at each transformed basis-vector tip."""
        self._clear_handles()
        dim = self._current_dimension
        for col_idx in range(dim):
            col = matrix[:, col_idx]
            center = [float(col[i]) if i < dim else 0.0 for i in range(3)]

            def _make_cb(_ci: int = col_idx) -> object:
                def _cb(pos: np.ndarray) -> None:
                    self._on_handle_dragged(_ci, pos)
                return _cb

            try:
                widget = self.plotter.add_sphere_widget(
                    callback=_make_cb(),
                    center=center,
                    radius=0.1,
                    color=BASIS_COLORS[col_idx],
                    style="surface",
                )
                self._handle_widgets.append(widget)
            except Exception:
                pass

    def _on_handle_dragged(self, col_idx: int, pos: np.ndarray) -> None:
        """Called by a sphere widget when the user drags a basis-vector handle."""
        if self._updating_matrix_inputs or self._timer.isActive():
            return  # ignore during animation or programmatic updates

        dim = self._current_dimension
        matrix = self._matrix_from_inputs()
        for row_idx in range(dim):
            matrix[row_idx, col_idx] = float(np.clip(pos[row_idx], -20.0, 20.0))

        self._updating_matrix_inputs = True
        self._set_matrix_inputs(matrix)
        self._updating_matrix_inputs = False

        self._matrix_state = analyze_matrix(matrix)
        self._scene_state = build_scene_state(self._matrix_state)
        self._update_dynamic_grid_targets()
        self._update_readouts(self._matrix_state)
        self._sync_preset_button_state()
        self._render_scene(progress=1.0)

    # ------------------------------------------------------------------
    # Keyboard shortcut
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self.animate_matrix()
        else:
            super().keyPressEvent(event)

    def apply_matrix(self, restore_camera: bool = False) -> None:
        self._timer.stop()
        matrix = self._resolve_matrix_for_action()
        if matrix is None:
            return
        self._matrix_state = analyze_matrix(matrix)
        self._scene_state = build_scene_state(self._matrix_state)
        self._sync_preset_button_state()
        self._update_dynamic_grid_targets()
        self._update_readouts(self._matrix_state)
        self._render_scene(progress=1.0)
        self._setup_interactive_handles(self._matrix_state.analysis_matrix)
        if restore_camera:
            self._restore_default_view()

    def animate_matrix(self) -> None:
        matrix = self._resolve_matrix_for_action()
        if matrix is None:
            return

        # ── Snapshot the current visual state so the animation starts from it ──
        if self._transformed_shape_mesh is not None and self._scene_state is not None:
            current_shape = self._transformed_shape_mesh.points.copy()
            current_basis = interpolate_arrays(
                self._scene_state.original_basis,
                self._scene_state.transformed_basis,
                self._last_render_progress,
            )
        else:
            current_shape = None
            current_basis = None

        # Snapshot current grid so the grid also animates from where it is now
        if self._current_dimension == 2 and self._transformed_grid_mesh is not None:
            self._reference_grid_points = self._transformed_grid_mesh.points.copy()

        self._matrix_state = analyze_matrix(matrix)
        new_scene = build_scene_state(self._matrix_state)

        # Build SceneState with current positions as origin, new positions as target
        if current_shape is not None and current_shape.shape == new_scene.transformed_vertices.shape:
            self._scene_state = SceneState(
                dimension=new_scene.dimension,
                original_vertices=current_shape,
                transformed_vertices=new_scene.transformed_vertices,
                original_basis=current_basis if current_basis is not None else new_scene.original_basis,
                transformed_basis=new_scene.transformed_basis,
                eigen_items=new_scene.eigen_items,
                animation_progress=0.0,
            )
        else:
            self._scene_state = new_scene

        self._sync_preset_button_state()
        self._update_dynamic_grid_targets()
        self._update_readouts(self._matrix_state)
        self._animation_started_at = time.perf_counter()
        self._clear_eigen_labels()
        self._clear_handles()
        self._render_scene(progress=0.0)
        self._timer.start(self.animation_interval_ms)

    def _tick_animation(self) -> None:
        elapsed_ms = (time.perf_counter() - self._animation_started_at) * 1000.0
        raw_progress = min(1.0, elapsed_ms / self.animation_duration_ms)
        eased_progress = raw_progress * raw_progress * (3.0 - 2.0 * raw_progress)
        self._render_scene(progress=eased_progress)
        if raw_progress >= 1.0:
            self._timer.stop()
            if self._matrix_state is not None:
                self._setup_interactive_handles(self._matrix_state.analysis_matrix)

    def _update_dynamic_grid_targets(self) -> None:
        source = self._original_grid_points if self._original_grid_points is not None else self._reference_grid_points
        if self._matrix_state is None or source is None:
            self._transformed_grid_points = None
            return
        self._transformed_grid_points = source @ self._matrix_state.matrix.T

    def _update_readouts(self, matrix_state: MatrixState) -> None:
        dim = matrix_state.dimension
        det = matrix_state.determinant
        plane = "XY plane" if dim == 2 else "3D space"
        self._mode_label.setText(f"{dim}\u00d7{dim} matrix  \u2014  {plane}")
        self._transform_label.setText(f"Type:  {matrix_state.transformation_label}")
        det_val = format_scalar(det)
        if abs(det) < 0.001:
            det_color = "#e07070"
        elif abs(det) < 0.5:
            det_color = "#d4a840"
        else:
            det_color = "#60b888"
        self._determinant_label.setText(
            f"det(A) = <span style='color:{det_color};'><b>{det_val}</b></span>"
        )
        trace = float(np.trace(matrix_state.analysis_matrix))
        rank = int(np.linalg.matrix_rank(matrix_state.analysis_matrix))
        self._trace_rank_label.setText(f"tr(A) = {format_scalar(trace)}   rank = {rank}")
        self._interpretation_label.setText(matrix_state.geometric_description)
        subscripts = "\u2081\u2082\u2083"
        eigen_lines = ["Eigenvalues"] + [
            f"  \u03bb{subscripts[i]} = {format_scalar(v)}"
            for i, v in enumerate(matrix_state.eigenvalues)
        ]
        self._eigenvalue_label.setText("\n".join(eigen_lines))
        self._refresh_advanced_status()

    def _refresh_advanced_status(self) -> None:
        if self._matrix_state is None:
            return

        lines: list[str] = []
        if self._show_eigen_checkbox.isChecked():
            lines.extend(self._matrix_state.warnings)
        else:
            real_count = len(self._matrix_state.eigen_items)
            lines.append("Overlay off \u2014 enable the checkbox to visualize eigenvectors.")
            if real_count == 0:
                lines.append("No real eigenvectors for this matrix.")
            elif real_count == 1:
                lines.append("1 real eigenvector available.")
            else:
                lines.append(f"{real_count} real eigenvectors available.")

        self._status_label.setText("\n".join(lines))

    def _render_scene(self, progress: float) -> None:
        if self._scene_state is None or self._transformed_shape_mesh is None or self._matrix_state is None:
            return

        clipped = float(np.clip(progress, 0.0, 1.0))
        self._last_render_progress = clipped
        self._apply_transformation_style(self._matrix_state.transformation_label)

        shape_points = interpolate_arrays(
            self._scene_state.original_vertices,
            self._scene_state.transformed_vertices,
            clipped,
        )
        self._transformed_shape_mesh.points = shape_points
        self._transformed_shape_mesh.Modified()

        if (
            self._current_dimension == 2
            and self._transformed_grid_mesh is not None
            and self._reference_grid_points is not None
            and self._transformed_grid_points is not None
        ):
            grid_points = interpolate_arrays(self._reference_grid_points, self._transformed_grid_points, clipped)
            self._transformed_grid_mesh.points = grid_points
            self._transformed_grid_mesh.Modified()

        basis_vectors = interpolate_arrays(
            self._scene_state.original_basis,
            self._scene_state.transformed_basis,
            clipped,
        )
        visible_count = 2 if self._current_dimension == 2 else 3

        for index, actor in enumerate(self._reference_basis_actors):
            actor.SetVisibility(index < visible_count)

        for index, (mesh, actor) in enumerate(zip(self._transformed_basis_meshes, self._transformed_basis_actors)):
            visible = index < visible_count
            actor.SetVisibility(visible)
            if visible:
                mesh.shallow_copy(self._build_arrow_mesh(basis_vectors[index], radius=0.032))

        if self._show_eigen_checkbox.isChecked():
            for index, actor in enumerate(self._eigen_arrow_actors):
                if index < len(self._scene_state.eigen_items):
                    eigen_item = self._scene_state.eigen_items[index]
                    actor.SetVisibility(True)
                    actor.GetProperty().SetColor(*eigen_item.color)
                    start_vector = eigen_item.eigenvector
                    target_vector = eigen_item.eigenvector * eigen_item.eigenvalue.real
                    animated_vector = interpolate_arrays(start_vector, target_vector, clipped)
                    self._eigen_arrow_meshes[index].shallow_copy(
                        self._build_arrow_mesh(animated_vector, radius=0.024)
                    )
                else:
                    actor.SetVisibility(False)
            if clipped >= 0.999:
                self._show_eigen_labels()
            else:
                self._clear_eigen_labels()
        else:
            for actor in self._eigen_arrow_actors:
                actor.SetVisibility(False)
            self._clear_eigen_labels()

        self.plotter.render()

    def _apply_transformation_style(self, transformation_label: str) -> None:
        style = TRANSFORMATION_STYLES.get(transformation_label, TRANSFORMATION_STYLES["Custom matrix"])
        if self._transformed_shape_actor is not None:
            shape_property = self._transformed_shape_actor.GetProperty()
            shape_property.SetColor(*self._hex_to_rgb(style["fill"]))
            try:
                shape_property.SetEdgeColor(*self._hex_to_rgb(style["edge"]))
            except AttributeError:
                pass
        if self._transformed_grid_actor is not None:
            self._transformed_grid_actor.GetProperty().SetColor(*self._hex_to_rgb(style["grid"]))

    def _show_eigen_labels(self) -> None:
        if self._scene_state is None:
            return

        self._clear_eigen_labels()
        for index, eigen_item in enumerate(self._scene_state.eigen_items):
            label_name = f"eigen_label_{index}"
            label_point = eigen_item.eigenvector * eigen_item.eigenvalue.real
            self.plotter.add_point_labels(
                np.array([label_point + self._label_offset(label_point)]),
                [eigen_item.label],
                name=label_name,
                text_color=self._rgb_to_hex(eigen_item.color),
                show_points=False,
                font_size=12,
                always_visible=True,
                reset_camera=False,
            )
            self._current_eigen_label_names.append(label_name)

    def _clear_eigen_labels(self) -> None:
        for label_name in self._current_eigen_label_names:
            try:
                self.plotter.remove_actor(label_name, reset_camera=False, render=False)
            except TypeError:
                try:
                    self.plotter.remove_actor(label_name)
                except Exception:
                    pass
            except Exception:
                pass
        self._current_eigen_label_names.clear()

    def _refresh_scene(self, _checked: bool | None = None) -> None:
        self._refresh_advanced_status()
        self._render_scene(progress=self._last_render_progress)

    def _build_shape_mesh(self, vertices: np.ndarray, dimension: int) -> pv.PolyData:
        faces = UNIT_SQUARE_FACES if dimension == 2 else CUBE_FACES
        return pv.PolyData(vertices, faces).triangulate()

    def _build_arrow_mesh(self, vector: Iterable[float], radius: float) -> pv.PolyData:
        direction = np.asarray(tuple(vector), dtype=float)
        length = float(np.linalg.norm(direction))
        if length <= 1e-8:
            return pv.Sphere(radius=0.04, center=(0.0, 0.0, 0.0))
        return pv.Arrow(
            start=(0.0, 0.0, 0.0),
            direction=tuple(direction / length),
            scale=length,
            shaft_radius=radius,
            tip_length=0.25,
            tip_radius=radius * 2.2,
        )

    def _build_plane_lattice(self, extent: float, step: float) -> tuple[np.ndarray, np.ndarray]:
        coordinates = np.arange(-extent, extent + 0.001, step)
        points: list[tuple[float, float, float]] = []
        lines: list[int] = []
        point_index = 0

        for x in coordinates:
            points.append((x, -extent, 0.015))
            points.append((x, extent, 0.015))
            lines.extend((2, point_index, point_index + 1))
            point_index += 2

        for y in coordinates:
            points.append((-extent, y, 0.015))
            points.append((extent, y, 0.015))
            lines.extend((2, point_index, point_index + 1))
            point_index += 2

        return np.array(points, dtype=float), np.array(lines, dtype=np.int64)

    def _label_offset(self, vector: np.ndarray) -> np.ndarray:
        length = float(np.linalg.norm(vector))
        if length <= 1e-8:
            return np.array([0.12, 0.12, 0.12], dtype=float)
        return 0.12 * (vector / length)

    def _hex_to_rgb(self, color: str) -> tuple[float, float, float]:
        color = color.lstrip("#")
        return tuple(int(color[index : index + 2], 16) / 255.0 for index in (0, 2, 4))

    def _rgb_to_hex(self, color: tuple[float, float, float]) -> str:
        channels = [max(0, min(255, round(component * 255))) for component in color]
        return "#{:02x}{:02x}{:02x}".format(*channels)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._timer.stop()
        self.plotter.close()
        super().closeEvent(event)


def launch_app() -> int:
    application = QtWidgets.QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QtWidgets.QApplication([])
        application.setStyle("Fusion")

    window = LinearTransformationWindow()
    window.show()

    if owns_application:
        return application.exec_()
    return 0


__all__ = ["LinearTransformationWindow", "launch_app"]
