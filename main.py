from __future__ import annotations


def main() -> int:
    try:
        from viewer import launch_app
    except ImportError as exc:
        print(exc)
        print("Install the GUI dependencies with: .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt")
        return 1

    try:
        return launch_app()
    except Exception as exc:
        print(f"Application failed to start: {exc}")
        print("If you see OpenGL or pixel-format errors, run the app in a normal local desktop session.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
