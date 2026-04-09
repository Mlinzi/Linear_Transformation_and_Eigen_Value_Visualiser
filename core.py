from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

FLOAT_TOLERANCE = 1e-7
VARIABLE_NAMES = {
    2: ("x", "y"),
    3: ("x", "y", "z"),
}

UNIT_SQUARE_VERTICES = np.array(
    [
        (-0.5, -0.5, 0.0),
        (0.5, -0.5, 0.0),
        (0.5, 0.5, 0.0),
        (-0.5, 0.5, 0.0),
    ],
    dtype=float,
)

UNIT_SQUARE_FACES = np.array([4, 0, 1, 2, 3], dtype=int)

CUBE_VERTICES = np.array(
    [
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
    ],
    dtype=float,
)

CUBE_FACES = np.array(
    [
        4,
        0,
        1,
        2,
        3,
        4,
        4,
        5,
        6,
        7,
        4,
        0,
        1,
        5,
        4,
        4,
        2,
        3,
        7,
        6,
        4,
        1,
        2,
        6,
        5,
        4,
        0,
        3,
        7,
        4,
    ],
    dtype=int,
)

BASIS_VECTORS = np.eye(3, dtype=float)
EIGEN_COLORS = (
    (0.98, 0.82, 0.22),
    (0.30, 0.92, 0.86),
    (0.98, 0.45, 0.36),
)


def _rotation_x(theta_radians: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta_radians), -np.sin(theta_radians)],
            [0.0, np.sin(theta_radians), np.cos(theta_radians)],
        ],
        dtype=float,
    )


def _rotation_z(theta_radians: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta_radians), -np.sin(theta_radians), 0.0],
            [np.sin(theta_radians), np.cos(theta_radians), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


_PRESETS_2D: tuple[tuple[str, np.ndarray], ...] = (
    ("Identity", np.eye(2, dtype=float)),
    ("Diagonal scaling", np.diag([1.8, 0.75]).astype(float)),
    ("Reflection", np.diag([1.0, -1.0]).astype(float)),
    (
        "Shear",
        np.array(
            [
                [1.0, 0.8],
                [0.0, 1.0],
            ],
            dtype=float,
        ),
    ),
    ("Rotation", _rotation_z(np.deg2rad(45.0))[:2, :2]),
)

_PRESETS_3D: tuple[tuple[str, np.ndarray], ...] = (
    ("Identity", np.eye(3, dtype=float)),
    ("Diagonal scaling", np.diag([1.8, 0.75, 1.35]).astype(float)),
    ("Reflection", np.diag([1.0, -1.0, 1.0]).astype(float)),
    (
        "Shear",
        np.array(
            [
                [1.0, 0.8, 0.0],
                [0.0, 1.0, 0.45],
                [0.2, 0.0, 1.0],
            ],
            dtype=float,
        ),
    ),
    ("Rotation about x", _rotation_x(np.deg2rad(35.0))),
    ("Rotation about z", _rotation_z(np.deg2rad(45.0))),
)


@dataclass(frozen=True)
class EigenDisplayItem:
    eigenvalue: complex
    eigenvector: np.ndarray
    is_real: bool
    label: str
    color: tuple[float, float, float]


@dataclass(frozen=True)
class MatrixState:
    matrix: np.ndarray
    analysis_matrix: np.ndarray
    dimension: int
    determinant: float
    eigenvalues: tuple[complex, ...]
    eigen_items: tuple[EigenDisplayItem, ...]
    transformation_label: str
    geometric_description: str
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SceneState:
    dimension: int
    original_vertices: np.ndarray
    transformed_vertices: np.ndarray
    original_basis: np.ndarray
    transformed_basis: np.ndarray
    eigen_items: tuple[EigenDisplayItem, ...]
    animation_progress: float = 1.0


def preset_names(dimension: int = 3) -> tuple[str, ...]:
    return tuple(name for name, _ in _presets_for_dimension(dimension))


def get_presets(dimension: int = 3) -> dict[str, np.ndarray]:
    return {name: matrix.copy() for name, matrix in _presets_for_dimension(dimension)}


def get_preset_matrix(name: str, dimension: int = 3) -> np.ndarray:
    presets = get_presets(dimension)
    if name not in presets:
        raise KeyError(f"Unknown preset: {name}")
    return presets[name]


def identify_preset_name(
    matrix: Sequence[Sequence[float]],
    dimension: int | None = None,
    tolerance: float = FLOAT_TOLERANCE,
) -> str | None:
    candidate = validate_matrix(matrix)
    preset_dimension = dimension or candidate.shape[0]
    for preset_name, preset_matrix in _presets_for_dimension(preset_dimension):
        if np.allclose(candidate, preset_matrix, atol=tolerance, rtol=0.0):
            return preset_name
    return None


def describe_transformation(
    matrix: Sequence[Sequence[float]],
    transformation_label: str | None = None,
    tolerance: float = FLOAT_TOLERANCE,
) -> str:
    candidate = validate_matrix(matrix)
    label = (transformation_label or "").strip().lower()
    determinant = float(np.linalg.det(candidate))
    identity = np.eye(candidate.shape[0], dtype=float)
    is_orthogonal = np.allclose(candidate.T @ candidate, identity, atol=1e-5, rtol=0.0)
    is_diagonal = np.allclose(candidate, np.diag(np.diag(candidate)), atol=tolerance, rtol=0.0)
    has_off_diagonal_mix = np.max(np.abs(candidate - np.diag(np.diag(candidate)))) > tolerance

    if "identity" in label or np.allclose(candidate, identity, atol=tolerance, rtol=0.0):
        return "Leaves the object, basis directions, and lattice unchanged."
    if "diagonal scaling" in label or (is_diagonal and not is_orthogonal):
        return "Stretches or compresses independently along the coordinate directions."
    if "reflection" in label or (is_orthogonal and determinant < 0.0):
        return "Mirrors the shape across an axis or plane and reverses orientation."
    if "shear" in label:
        return "Slides one direction along another, turning the grid and shape into a slanted form."
    if "rotation" in label or (is_orthogonal and determinant > 0.0):
        return "Turns the shape while preserving lengths and angles."
    if abs(determinant) <= tolerance:
        return "Collapses area or volume toward a lower-dimensional output."
    if has_off_diagonal_mix:
        return "Mixes coordinate directions, so the output rotates, slants, or skews."
    return "Applies a custom linear transformation to the reference shape and basis vectors."


def _presets_for_dimension(dimension: int) -> tuple[tuple[str, np.ndarray], ...]:
    if dimension == 2:
        return _PRESETS_2D
    if dimension == 3:
        return _PRESETS_3D
    raise ValueError("Only 2x2 and 3x3 matrices are supported.")


def _is_row_like(value: object) -> bool:
    return hasattr(value, "__iter__") and not isinstance(value, (str, bytes))


def parse_matrix_entries(entries: Sequence[object]) -> tuple[np.ndarray | None, tuple[str, ...]]:
    raw_items = list(entries)
    if raw_items and all(_is_row_like(item) for item in raw_items):
        rows = [list(row) for row in raw_items]
        if len(rows) not in {2, 3} or any(len(row) != len(rows) for row in rows):
            return None, ("Matrix input must be either 2x2 or 3x3.",)
    else:
        if len(raw_items) not in {4, 9}:
            return None, ("Matrix input must contain either 4 or 9 values.",)
        size = 2 if len(raw_items) == 4 else 3
        rows = [raw_items[index : index + size] for index in range(0, len(raw_items), size)]

    numbers: list[float] = []
    errors: list[str] = []

    for row_index, row in enumerate(rows, start=1):
        for column_index, raw_value in enumerate(row, start=1):
            value = raw_value.strip() if isinstance(raw_value, str) else raw_value
            if value == "":
                errors.append(f"Entry ({row_index}, {column_index}) is empty.")
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors.append(f"Entry ({row_index}, {column_index}) must be a real number.")
                continue
            if not np.isfinite(number):
                errors.append(f"Entry ({row_index}, {column_index}) must be finite.")
                continue
            numbers.append(number)

    if errors:
        return None, tuple(errors)

    size = len(rows)
    matrix = np.array(numbers, dtype=float).reshape(size, size)
    return matrix, ()


def _normalize_output_variable(raw_name: str) -> str | None:
    cleaned = (
        raw_name.strip()
        .lower()
        .replace(" ", "")
        .replace("′", "'")
        .replace("’", "'")
        .replace("`", "'")
    )
    if cleaned.endswith("'"):
        cleaned = cleaned[:-1]
    if cleaned in {"x", "y", "z"}:
        return cleaned
    return None


def _parse_linear_expression(
    expression: str,
    variables: Sequence[str],
) -> tuple[np.ndarray | None, str | None]:
    text = expression.strip().lower().replace(" ", "").replace("−", "-")
    if not text:
        return None, "Right-hand side is empty."
    if text == "0":
        return np.zeros(len(variables), dtype=float), None

    normalized = text.replace("-", "+-")
    if normalized.startswith("+-"):
        normalized = normalized[1:]
    if normalized.startswith("+"):
        normalized = normalized[1:]

    coefficients = np.zeros(len(variables), dtype=float)
    terms = [term for term in normalized.split("+") if term]

    for term in terms:
        matching_variable = next((variable for variable in variables if term.endswith(variable)), None)
        if matching_variable is None:
            return None, f"Term '{term}' is not a valid linear term."

        prefix = term[: -len(matching_variable)]
        if prefix.endswith("*"):
            prefix = prefix[:-1]

        if prefix in {"", "+"}:
            coefficient = 1.0
        elif prefix == "-":
            coefficient = -1.0
        else:
            try:
                coefficient = float(prefix)
            except ValueError:
                return None, f"Term '{term}' has an invalid coefficient."

        coefficients[variables.index(matching_variable)] += coefficient

    return coefficients, None


def parse_linear_equations(
    equations: Sequence[str],
    dimension: int | None = None,
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    raw_equations = [str(equation).strip() for equation in equations]
    expected_dimension = dimension or len(raw_equations)
    if expected_dimension not in VARIABLE_NAMES:
        return None, ("Only 2x2 and 3x3 equation systems are supported.",)
    if len(raw_equations) != expected_dimension:
        return None, (f"Provide exactly {expected_dimension} equations.",)

    variables = VARIABLE_NAMES[expected_dimension]
    rows_by_output: dict[str, np.ndarray] = {}
    errors: list[str] = []

    for equation_index, equation in enumerate(raw_equations, start=1):
        if not equation:
            errors.append(f"Equation {equation_index} is empty.")
            continue
        if equation.count("=") != 1:
            errors.append(f"Equation {equation_index} must contain exactly one '=' sign.")
            continue

        lhs, rhs = equation.split("=", 1)
        output_variable = _normalize_output_variable(lhs)
        if output_variable not in variables:
            expected = ", ".join(f"{variable}'" for variable in variables)
            errors.append(
                f"Equation {equation_index} must define one of: {expected}."
            )
            continue
        if output_variable in rows_by_output:
            errors.append(f"Output '{output_variable}' is defined more than once.")
            continue

        row, error = _parse_linear_expression(rhs, variables)
        if error is not None:
            errors.append(f"Equation {equation_index}: {error}")
            continue
        rows_by_output[output_variable] = row

    for variable in variables:
        if variable not in rows_by_output:
            errors.append(f"Missing equation for {variable}'.")

    if errors:
        return None, tuple(errors)

    matrix = np.vstack([rows_by_output[variable] for variable in variables])
    return matrix, ()


def _format_equation_coefficient(value: float, digits: int = 3) -> str:
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text or "0"


def format_linear_equations(
    matrix: Sequence[Sequence[float]],
    digits: int = 3,
) -> tuple[str, ...]:
    candidate = validate_matrix(matrix)
    variables = VARIABLE_NAMES[candidate.shape[0]]
    equations: list[str] = []

    for output_variable, row in zip(variables, candidate):
        terms: list[str] = []
        for coefficient, variable in zip(row, variables):
            if abs(coefficient) <= FLOAT_TOLERANCE:
                continue

            sign = "-" if coefficient < 0 else "+"
            magnitude = abs(float(coefficient))
            coefficient_text = "" if abs(magnitude - 1.0) <= FLOAT_TOLERANCE else _format_equation_coefficient(magnitude, digits)
            term_body = f"{coefficient_text}{variable}" if coefficient_text else variable

            if not terms:
                terms.append(f"-{term_body}" if sign == "-" else term_body)
            else:
                terms.append(f"{sign} {term_body}")

        right_hand_side = " ".join(terms) if terms else "0"
        equations.append(f"{output_variable}' = {right_hand_side}")

    return tuple(equations)


def validate_matrix(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Return a finite 2x2 or 3x3 matrix as a NumPy array.

    Accepted inputs include nested Python sequences and NumPy arrays whose
    shape is exactly `(2, 2)` or `(3, 3)`.
    """
    candidate = np.asarray(matrix, dtype=float)
    if candidate.shape not in {(2, 2), (3, 3)}:
        raise ValueError("Matrix must be 2x2 or 3x3.")
    if not np.isfinite(candidate).all():
        raise ValueError("Matrix entries must be finite.")
    return candidate


def embed_for_display(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    candidate = validate_matrix(matrix)
    if candidate.shape == (3, 3):
        return candidate.copy()

    embedded = np.eye(3, dtype=float)
    embedded[:2, :2] = candidate
    return embedded


def transform_points(points: Sequence[Sequence[float]], matrix: Sequence[Sequence[float]]) -> np.ndarray:
    transform = embed_for_display(matrix)
    point_array = np.asarray(points, dtype=float)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("Points must be an array of shape (n, 3).")
    return point_array @ transform.T


def _is_effectively_real(value: complex, tolerance: float = FLOAT_TOLERANCE) -> bool:
    return abs(complex(value).imag) <= tolerance


def _coerce_real_vector(vector: Iterable[complex], tolerance: float = FLOAT_TOLERANCE) -> np.ndarray | None:
    vector_array = np.asarray(tuple(vector), dtype=complex)
    if np.max(np.abs(vector_array.imag)) > tolerance:
        return None
    return vector_array.real.astype(float)


def _orient_vector(vector: np.ndarray) -> np.ndarray:
    oriented = vector.astype(float, copy=True)
    dominant_index = int(np.argmax(np.abs(oriented)))
    if oriented[dominant_index] < 0:
        oriented *= -1.0
    return oriented


def _embed_vector(vector: np.ndarray) -> np.ndarray:
    if vector.shape == (3,):
        return vector.astype(float, copy=True)
    if vector.shape == (2,):
        return np.array([vector[0], vector[1], 0.0], dtype=float)
    raise ValueError("Eigenvectors must have length 2 or 3.")


def _build_status_messages(real_count: int, hidden_count: int, determinant: float) -> tuple[str, ...]:
    messages: list[str] = []
    if abs(determinant) <= FLOAT_TOLERANCE:
        messages.append(
            "Matrix is singular or nearly singular; the transformed output may collapse into a line, plane, or lower-dimensional shape."
        )

    if real_count == 0:
        messages.append("No real eigenvectors can be drawn for this matrix.")
    elif hidden_count == 0:
        messages.append(f"All {real_count} eigenvectors are real and shown.")
    else:
        vector_word = "eigenvector" if real_count == 1 else "eigenvectors"
        pair_word = "eigenpair" if hidden_count == 1 else "eigenpairs"
        messages.append(
            f"{real_count} real {vector_word} shown; remaining {hidden_count} {pair_word} are complex or not visually representable."
        )
    return tuple(messages)


def analyze_matrix(
    matrix: Sequence[Sequence[float]],
    transformation_label: str | None = None,
    tolerance: float = FLOAT_TOLERANCE,
) -> MatrixState:
    analysis_matrix = validate_matrix(matrix)
    display_matrix = embed_for_display(analysis_matrix)
    determinant = float(np.linalg.det(analysis_matrix))
    eigenvalues, eigenvectors = np.linalg.eig(analysis_matrix)

    real_pairs: list[tuple[complex, np.ndarray]] = []
    hidden_count = 0

    ordered_pairs = list(zip(eigenvalues, eigenvectors.T))
    ordered_pairs.sort(key=lambda pair: (not _is_effectively_real(pair[0], tolerance), -abs(pair[0].real)))

    for eigenvalue, eigenvector in ordered_pairs:
        if not _is_effectively_real(eigenvalue, tolerance):
            hidden_count += 1
            continue

        real_vector = _coerce_real_vector(eigenvector, tolerance)
        if real_vector is None:
            hidden_count += 1
            continue

        norm = float(np.linalg.norm(real_vector))
        if norm <= tolerance:
            hidden_count += 1
            continue

        normalized_vector = _orient_vector(_embed_vector(real_vector / norm))
        real_pairs.append((complex(float(np.real(eigenvalue)), 0.0), normalized_vector))

    eigen_items = tuple(
        EigenDisplayItem(
            eigenvalue=eigenvalue,
            eigenvector=eigenvector,
            is_real=True,
            label=f"lambda = {format_scalar(eigenvalue.real)}",
            color=EIGEN_COLORS[index % len(EIGEN_COLORS)],
        )
        for index, (eigenvalue, eigenvector) in enumerate(real_pairs)
    )

    sorted_eigenvalues = tuple(complex(value) for value, _ in ordered_pairs)
    resolved_label = transformation_label or identify_preset_name(analysis_matrix, tolerance=tolerance) or "Custom matrix"
    warnings = _build_status_messages(len(eigen_items), hidden_count, determinant)

    return MatrixState(
        matrix=display_matrix,
        analysis_matrix=analysis_matrix.copy(),
        dimension=int(analysis_matrix.shape[0]),
        determinant=determinant,
        eigenvalues=sorted_eigenvalues,
        eigen_items=eigen_items,
        transformation_label=resolved_label,
        geometric_description=describe_transformation(analysis_matrix, resolved_label, tolerance=tolerance),
        errors=(),
        warnings=warnings,
    )


def build_scene_state(matrix_state: MatrixState) -> SceneState:
    if matrix_state.dimension == 2:
        original_vertices = UNIT_SQUARE_VERTICES
    else:
        original_vertices = CUBE_VERTICES

    return SceneState(
        dimension=matrix_state.dimension,
        original_vertices=original_vertices.copy(),
        transformed_vertices=transform_points(original_vertices, matrix_state.matrix),
        original_basis=BASIS_VECTORS.copy(),
        transformed_basis=transform_points(BASIS_VECTORS, matrix_state.matrix),
        eigen_items=matrix_state.eigen_items,
        animation_progress=1.0,
    )


def interpolate_arrays(start: np.ndarray, end: np.ndarray, progress: float) -> np.ndarray:
    clipped = float(np.clip(progress, 0.0, 1.0))
    return (1.0 - clipped) * start + clipped * end


def format_scalar(value: complex | float, digits: int = 3) -> str:
    candidate = complex(value)
    if abs(candidate.imag) <= FLOAT_TOLERANCE:
        return f"{candidate.real:.{digits}f}"

    real_part = f"{candidate.real:.{digits}f}"
    imag_part = f"{abs(candidate.imag):.{digits}f}"
    sign = "+" if candidate.imag >= 0 else "-"
    return f"{real_part} {sign} {imag_part}i"


def format_eigenvalues(values: Sequence[complex]) -> str:
    lines = [f"lambda{index}: {format_scalar(value)}" for index, value in enumerate(values, start=1)]
    return "\n".join(lines)


__all__ = [
    "BASIS_VECTORS",
    "CUBE_FACES",
    "CUBE_VERTICES",
    "EIGEN_COLORS",
    "EigenDisplayItem",
    "FLOAT_TOLERANCE",
    "MatrixState",
    "SceneState",
    "UNIT_SQUARE_FACES",
    "UNIT_SQUARE_VERTICES",
    "analyze_matrix",
    "build_scene_state",
    "describe_transformation",
    "embed_for_display",
    "format_eigenvalues",
    "format_linear_equations",
    "format_scalar",
    "get_preset_matrix",
    "get_presets",
    "identify_preset_name",
    "interpolate_arrays",
    "parse_linear_equations",
    "parse_matrix_entries",
    "preset_names",
    "transform_points",
    "validate_matrix",
]
