"""Entry point for running the BCF application locally.

This module is used when cloning the repository without installing it as a
package.  The original script expected the package name ``bcf_app`` to be
available which is not the case in a fresh checkout, leading to a
``ModuleNotFoundError``.  We keep compatibility with the installed package
while providing a graceful fallback to the local sources.
"""

try:
    # Preferred path when the project is installed as a package.
    from bcf_app.main import launch
except ModuleNotFoundError:  # pragma: no cover - depends on runtime context
    # Fallback for running straight from the repository checkout.
    from importlib import import_module
    from pathlib import Path
    import sys

    repo_dir = Path(__file__).resolve().parent
    package_parent = repo_dir.parent
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

    try:
        launch = import_module(f"{repo_dir.name}.ui.app").launch
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime context
        raise ModuleNotFoundError(
            "Unable to locate the UI module. Ensure project dependencies are installed."
        ) from exc

if __name__ == "__main__":
    launch()
