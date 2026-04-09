from __future__ import annotations

import unittest

import numpy as np

from core import (
    FLOAT_TOLERANCE,
    analyze_matrix,
    build_scene_state,
    format_linear_equations,
    get_preset_matrix,
    parse_linear_equations,
    parse_matrix_entries,
    validate_matrix,
)


class CoreMathTests(unittest.TestCase):
    def test_parse_matrix_entries_rejects_invalid_values(self) -> None:
        matrix, errors = parse_matrix_entries(
            [
                ["1", "0", "0"],
                ["0", "bad", "0"],
                ["0", "0", ""],
            ]
        )

        self.assertIsNone(matrix)
        self.assertGreaterEqual(len(errors), 2)
        self.assertIn("Entry (2, 2) must be a real number.", errors)
        self.assertIn("Entry (3, 3) is empty.", errors)

    def test_identity_matrix_leaves_cube_unchanged(self) -> None:
        state = analyze_matrix(np.eye(3))
        scene = build_scene_state(state)

        np.testing.assert_allclose(scene.original_vertices, scene.transformed_vertices)
        np.testing.assert_allclose(scene.original_basis, scene.transformed_basis)
        self.assertEqual(state.transformation_label, "Identity")
        self.assertIn("unchanged", state.geometric_description.lower())

    def test_diagonal_scaling_produces_axis_aligned_eigenvectors(self) -> None:
        state = analyze_matrix(get_preset_matrix("Diagonal scaling"))

        self.assertEqual(len(state.eigen_items), 3)
        actual_vectors = {tuple(np.round(np.abs(item.eigenvector), 6)) for item in state.eigen_items}
        expected_vectors = {
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        }
        self.assertEqual(actual_vectors, expected_vectors)
        self.assertEqual(state.transformation_label, "Diagonal scaling")
        self.assertIn("stretches", state.geometric_description.lower())

    def test_rotation_about_z_has_only_real_z_axis_eigenvector(self) -> None:
        state = analyze_matrix(get_preset_matrix("Rotation about z"))

        self.assertEqual(len(state.eigen_items), 1)
        np.testing.assert_allclose(np.abs(state.eigen_items[0].eigenvector), np.array([0.0, 0.0, 1.0]))
        self.assertIn("1 real eigenvector shown", state.warnings[-1])

    def test_two_by_two_mode_uses_two_eigenvalues_and_embeds_for_display(self) -> None:
        state = analyze_matrix(get_preset_matrix("Diagonal scaling", 2))
        scene = build_scene_state(state)

        self.assertEqual(state.dimension, 2)
        self.assertEqual(scene.dimension, 2)
        self.assertEqual(len(state.eigenvalues), 2)
        self.assertEqual(len(state.eigen_items), 2)
        self.assertEqual(scene.original_vertices.shape[0], 4)
        np.testing.assert_allclose(scene.transformed_basis[2], np.array([0.0, 0.0, 1.0]))

    def test_singular_matrix_reports_warning_without_crashing(self) -> None:
        state = analyze_matrix(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        )

        self.assertTrue(any("singular" in message.lower() for message in state.warnings))

    def test_custom_matrix_gets_generic_transformation_summary(self) -> None:
        state = analyze_matrix(np.array([[1.0, 0.25], [0.1, 1.2]]))

        self.assertEqual(state.transformation_label, "Custom matrix")
        self.assertTrue(state.geometric_description)

    def test_validate_matrix_rejects_invalid_dimension(self) -> None:
        with self.assertRaises(ValueError):
            validate_matrix(np.eye(4))

    def test_validate_matrix_rejects_non_finite_entries(self) -> None:
        with self.assertRaises(ValueError):
            validate_matrix([[1.0, np.inf], [0.0, 1.0]])

        with self.assertRaises(ValueError):
            validate_matrix([[1.0, 0.0], [np.nan, 1.0]])

    def test_near_singular_matrix_uses_tolerance_for_warning(self) -> None:
        state = analyze_matrix(np.diag([FLOAT_TOLERANCE / 10.0, 1.0]))

        self.assertTrue(any("singular" in message.lower() for message in state.warnings))
        self.assertEqual(len(state.eigen_items), 2)

    def test_parse_linear_equations_builds_matrix_from_equation_rules(self) -> None:
        matrix, errors = parse_linear_equations(
            [
                "x' = 2x + y",
                "y' = -x + 3y",
            ],
            dimension=2,
        )

        self.assertEqual(errors, ())
        np.testing.assert_allclose(matrix, np.array([[2.0, 1.0], [-1.0, 3.0]]))

    def test_parse_linear_equations_accepts_output_rows_in_any_order(self) -> None:
        matrix, errors = parse_linear_equations(
            [
                "z' = z - x",
                "x' = x + 2y",
                "y' = -y",
            ],
            dimension=3,
        )

        self.assertEqual(errors, ())
        np.testing.assert_allclose(
            matrix,
            np.array(
                [
                    [1.0, 2.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [-1.0, 0.0, 1.0],
                ]
            ),
        )

    def test_parse_linear_equations_rejects_non_linear_or_constant_terms(self) -> None:
        matrix, errors = parse_linear_equations(
            [
                "x' = 2x + 1",
                "y' = y",
            ],
            dimension=2,
        )

        self.assertIsNone(matrix)
        self.assertTrue(any("not a valid linear term" in error for error in errors))

    def test_format_linear_equations_round_trips_matrix(self) -> None:
        source_matrix = np.array([[1.5, -1.0], [0.0, 2.25]])
        equations = format_linear_equations(source_matrix)
        parsed_matrix, errors = parse_linear_equations(equations, dimension=2)

        self.assertEqual(errors, ())
        np.testing.assert_allclose(parsed_matrix, source_matrix)


if __name__ == "__main__":
    unittest.main()
