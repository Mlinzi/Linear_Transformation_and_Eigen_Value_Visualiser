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
    "Custom matrix": {"fill": "#ff9f5e", "edge": "#ffe0c4", "grid": "#dd8a51"},
}


class LinearTransformationWindow(QtWidgets.QMainWindow):
    animation_duration_ms = 950
    animation_interval_ms = 16

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TransformLab: Animated Linear Transformation Visualizer")
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

        self._matrix_inputs: list[list[QtWidgets.QDoubleSpinBox]] = []
        self._equation_inputs: list[QtWidgets.QLineEdit] = []
        self._preset_buttons: dict[str, QtWidgets.QPushButton] = {}

        self._transformed_shape_mesh: pv.PolyData | None = None
        self._transformed_shape_actor: object | None = None
        self._transformed_grid_mesh: pv.PolyData | None = None
        self._transformed_grid_actor: object | None = None
        self._reference_grid_points: np.ndarray | None = None
        self._transformed_grid_points: np.ndarray | None = None
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

        title = QtWidgets.QLabel("Transformation Controls", panel)
        title.setObjectName("panelTitle")
        panel_layout.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "Choose a matrix type, animate it, and watch how the visual output changes.",
            panel,
        )
        subtitle.setObjectName("sectionHint")
        subtitle.setWordWrap(True)
        panel_layout.addWidget(subtitle)

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
            "Start with a preset. Use the custom matrix below only when you want a manual transform.",
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
            "Editing these values creates a custom transformation even if a preset is still selected above.",
            panel,
        )
        manual_hint.setObjectName("sectionHint")
        manual_hint.setWordWrap(True)
        panel_layout.addWidget(manual_hint)

        self._equation_group = QtWidgets.QGroupBox("Linear Equations", panel)
        equation_layout = QtWidgets.QVBoxLayout(self._equation_group)
        equation_hint = QtWidgets.QLabel(
            "Type rules like x' = 2x + y. Press Apply or Animate after editing equations.",
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

        button_grid = QtWidgets.QGridLayout()
        animate_button = QtWidgets.QPushButton("Animate", panel)
        animate_button.setProperty("primaryButton", True)
        animate_button.clicked.connect(self.animate_matrix)
        apply_button = QtWidgets.QPushButton("Apply", panel)
        apply_button.clicked.connect(self.apply_matrix)
        reset_button = QtWidgets.QPushButton("Reset All", panel)
        reset_button.clicked.connect(self.reset_matrix)
        home_button = QtWidgets.QPushButton("Home View", panel)
        home_button.clicked.connect(self.restore_view)

        button_grid.addWidget(animate_button, 0, 0)
        button_grid.addWidget(apply_button, 0, 1)
        button_grid.addWidget(reset_button, 1, 0)
        button_grid.addWidget(home_button, 1, 1)
        panel_layout.addLayout(button_grid)

        self._show_eigen_checkbox = QtWidgets.QCheckBox("Show Eigenvector Overlay", panel)
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
        self._interpretation_label = QtWidgets.QLabel(summary_group)
        self._interpretation_label.setObjectName("status")
        self._interpretation_label.setWordWrap(True)
        summary_layout.addWidget(self._mode_label)
        summary_layout.addWidget(self._transform_label)
        summary_layout.addWidget(self._determinant_label)
        summary_layout.addWidget(self._interpretation_label)
        panel_layout.addWidget(summary_group)

        advanced_group = QtWidgets.QGroupBox("Advanced Insight", panel)
        advanced_layout = QtWidgets.QVBoxLayout(advanced_group)
        advanced_hint = QtWidgets.QLabel(
            "Eigenvalues and eigenvectors are optional analysis layers, not the main interaction.",
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

        notes_group = QtWidgets.QGroupBox("Suggested Demo Flow", panel)
        notes_layout = QtWidgets.QVBoxLayout(notes_group)
        notes = QtWidgets.QLabel(
            "Start in 2 x 2 mode with Identity, then show Diagonal scaling, Reflection, Shear, and Rotation.\n"
            "Use 3 x 3 only as the advanced extension after the plane-based story is clear.\n"
            "Turn on the eigenvector overlay only for a simple case or when the audience asks.",
            notes_group,
        )
        notes.setWordWrap(True)
        notes_layout.addWidget(notes)
        panel_layout.addWidget(notes_group)

        panel_layout.addStretch(1)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background: #0b1523;
                color: #e7edf6;
                font-family: "Segoe UI";
                font-size: 13px;
            }
            QFrame#controlPanel, QFrame#viewerFrame {
                border: 1px solid #203349;
                border-radius: 16px;
            }
            QGroupBox {
                border: 1px solid #29435e;
                border-radius: 12px;
                margin-top: 10px;
                padding-top: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                left: 12px;
                padding: 0 4px 0 4px;
            }
            QLabel#panelTitle {
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#sectionHint {
                color: #97abc4;
                font-size: 12px;
            }
            QLabel#readout {
                font-family: "Consolas";
                line-height: 1.45;
            }
            QLabel#status {
                color: #ffd69b;
                line-height: 1.4;
            }
            QPushButton {
                background: #1d3552;
                border: 1px solid #315173;
                border-radius: 10px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #28486c;
            }
            QPushButton:pressed {
                background: #17314b;
            }
            QPushButton[primaryButton="true"] {
                background: #235287;
                border: 1px solid #6fb4ff;
            }
            QPushButton[primaryButton="true"]:hover {
                background: #2b639f;
            }
            QPushButton[presetButton="true"] {
                text-align: left;
                padding: 8px 10px;
            }
            QPushButton[presetButton="true"]:checked {
                background: #315c88;
                border: 1px solid #7ab7ff;
            }
            QComboBox, QDoubleSpinBox, QLineEdit {
                background: #0f1b2d;
                border: 1px solid #315173;
                border-radius: 8px;
                padding: 6px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QCheckBox {
                font-weight: 600;
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
        self.plotter.render()

    def _reset_dynamic_handles(self) -> None:
        self._transformed_shape_mesh = None
        self._transformed_shape_actor = None
        self._transformed_grid_mesh = None
        self._transformed_grid_actor = None
        self._reference_grid_points = None
        self._transformed_grid_points = None
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
        self._reference_grid_points = grid_points.copy()
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
            return matrix
        return self._matrix_from_inputs()

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
        self.apply_matrix(restore_camera=restore_camera)

    def reset_matrix(self) -> None:
        self._timer.stop()
        self._show_eigen_checkbox.setChecked(False)
        self._apply_preset("Identity", restore_camera=True)

    def restore_view(self) -> None:
        self._restore_default_view()

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
        if restore_camera:
            self._restore_default_view()

    def animate_matrix(self) -> None:
        matrix = self._resolve_matrix_for_action()
        if matrix is None:
            return
        self._matrix_state = analyze_matrix(matrix)
        self._scene_state = build_scene_state(self._matrix_state)
        self._sync_preset_button_state()
        self._update_dynamic_grid_targets()
        self._update_readouts(self._matrix_state)
        self._animation_started_at = time.perf_counter()
        self._clear_eigen_labels()
        self._render_scene(progress=0.0)
        self._timer.start(self.animation_interval_ms)

    def _tick_animation(self) -> None:
        elapsed_ms = (time.perf_counter() - self._animation_started_at) * 1000.0
        raw_progress = min(1.0, elapsed_ms / self.animation_duration_ms)
        eased_progress = raw_progress * raw_progress * (3.0 - 2.0 * raw_progress)
        self._render_scene(progress=eased_progress)
        if raw_progress >= 1.0:
            self._timer.stop()

    def _update_dynamic_grid_targets(self) -> None:
        if self._matrix_state is None or self._reference_grid_points is None:
            self._transformed_grid_points = None
            return
        self._transformed_grid_points = self._reference_grid_points @ self._matrix_state.matrix.T

    def _update_readouts(self, matrix_state: MatrixState) -> None:
        mode_text = f"Matrix size: {matrix_state.dimension} x {matrix_state.dimension}"
        mode_text += " on the XY plane" if matrix_state.dimension == 2 else " in full 3D"
        self._mode_label.setText(mode_text)
        self._transform_label.setText(f"Active transformation: {matrix_state.transformation_label}")
        self._determinant_label.setText(f"Determinant: {format_scalar(matrix_state.determinant)}")
        self._interpretation_label.setText(matrix_state.geometric_description)
        self._eigenvalue_label.setText(f"Eigenvalues\n{format_eigenvalues(matrix_state.eigenvalues)}")
        self._refresh_advanced_status()

    def _refresh_advanced_status(self) -> None:
        if self._matrix_state is None:
            return

        lines: list[str] = []
        if self._show_eigen_checkbox.isChecked():
            lines.extend(self._matrix_state.warnings)
        else:
            real_count = len(self._matrix_state.eigen_items)
            lines.append("Overlay hidden. Enable the checkbox to draw real eigenvectors.")
            if real_count == 0:
                lines.append("No real eigenvectors are available for this matrix.")
            elif real_count == 1:
                lines.append("1 real eigenvector is available.")
            else:
                lines.append(f"{real_count} real eigenvectors are available.")

        self._status_label.setText("Overlay status\n" + "\n".join(lines))

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
