# Project Report: TransformLab

## 1. Project Idea

Our project is an **animated linear transformation visualizer**. The user selects a matrix or a preset transformation, and the program shows how that matrix changes a reference shape, the basis vectors, and a coordinate lattice.

The core idea is not just to compute numbers. The matrix is the main input, and the animated visual output depends directly on the kind of transformation the user chooses.

## 2. Main Objective

The goal is to make linear transformations feel concrete. A matrix can act like a rule that stretches, reflects, shears, or rotates a shape, and each kind of matrix produces a visibly different result.

This project makes that idea visible by showing:

- a reference shape,
- the transformed shape,
- the original basis vectors,
- the transformed basis vectors,
- a coordinate lattice in `2 x 2` mode,
- and optional eigenvector overlays for advanced discussion.

## 3. Tools Used

- Python
- NumPy
- PyVista
- pyvistaqt
- PyQt5

## 4. How the Program Works

1. The user selects `2 x 2` or `3 x 3` mode.
2. The user chooses a preset such as scaling, reflection, shear, or rotation, or enters a custom matrix.
3. The program applies that matrix to a reference shape and the standard basis vectors.
4. In `2 x 2` mode, the app also transforms a visible lattice on the `xy` plane so the effect is easier to see.
5. The transformation can be previewed instantly or animated smoothly.
6. The app also computes the determinant and eigenvalues.
7. If real eigenvectors exist, they can be shown as an optional overlay.

## 5. Mathematical Concepts Shown

### Linear Transformation

If `A` is a matrix and `v` is a vector, then the transformed vector is:

`A v`

This changes the direction, orientation, or size of the object depending on the matrix.

### Eigenvector and Eigenvalue

A non-zero vector `v` is an eigenvector of `A` if:

`A v = lambda v`

Here `lambda` is the eigenvalue. This means the vector keeps the same line of action and is only stretched, shrunk, or flipped. In our project, this is an optional advanced layer, not the main feature.

## 6. Presets Demonstrated

### Identity

- The shape and lattice stay unchanged.
- This shows what it means for a matrix to leave the visual output fixed.

### Diagonal Scaling

- The shape stretches differently along the coordinate directions.
- The transformed output clearly depends on the scaling factors in the matrix.

### Reflection

- The shape flips across an axis or plane.
- This is the clearest example of orientation reversal.

### Shear

- The square or cube becomes slanted.
- The lattice makes the shearing effect very easy to see.

### Rotation

- The shape turns while keeping its size.
- This shows that not every transformation is just stretching.
- In some rotation cases, the eigen overlay also shows that not all eigenvectors are real.

## 7. Screenshots to Add

- Screenshot 1: `2 x 2` Identity
- Screenshot 2: `2 x 2` Shear
- Screenshot 3: `2 x 2` Rotation
- Screenshot 4: `3 x 3` extension mode

## 8. Conclusion

This project turns linear transformations into motion and geometry. The matrix is the main driver of the output, and different kinds of matrices create clearly different visual results. That makes the project suitable both as a math demo and as a matrix-driven graphics application.
