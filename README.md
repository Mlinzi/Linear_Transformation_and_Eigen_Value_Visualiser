# TransformLab: Animated Linear Transformation Visualizer

This project is a local desktop visualizer for a math group presentation. The main experience is matrix-driven animation: choose a linear transformation, apply it to a reference shape, and watch the visual output change. The app starts in a `2 x 2` plane-based mode for the clearest demos and keeps `3 x 3` mode as an advanced extension.

## Features

- Single-window desktop app with a PyVista viewport and Qt control panel
- `2 x 2` transformation mode with a unit square, transformed basis vectors, and animated `xy` lattice
- `3 x 3` extension mode with a transformed cube and basis vectors in full 3D
- Visible preset buttons for scaling, reflection, shear, and rotation
- Alternative equation input such as `x' = 2x + y` and `y' = -x + 3y`
- Smooth in-place animation plus `Apply`, `Reset All`, and `Home View`
- Compact transformation summary with optional eigenvector overlay

## Project Files

- `main.py`: startup entrypoint
- `core.py`: matrix validation, transformation math, preset metadata, eigenvalue/eigenvector analysis
- `viewer.py`: PyQt5 and PyVista interface, scene construction, animation, and rendering
- `PROJECT_REPORT.md`: short write-up scaffold for the presentation/report

## Setup

1. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Install the GUI dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

## Equation Input

You can enter linear transformation rules directly in the equation fields, for example:

```text
x' = 2x + y
y' = -x + 3y
```

or in `3 x 3` mode:

```text
x' = x + y
y' = y - z
z' = z
```

After editing the equations, press `Apply`, `Animate`, or `Use Equations`. The app converts the equations into the matching matrix automatically.

## Notes

- The app only draws eigenvectors that are real and visually representable.
- For matrices with complex eigenvalues, the advanced panel explains why not all eigenvectors are drawn.
- Singular matrices are allowed; the app warns when the transformed output collapses toward a lower-dimensional shape.
- PyVista needs a normal desktop OpenGL session. If you launch from a headless shell or restricted remote environment, the renderer may fail even though the Python code is correct.
