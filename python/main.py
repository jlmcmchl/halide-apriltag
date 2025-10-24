#!/usr/bin/env python3
"""Minimal Halide GPU pipeline starter for experimentation."""

from __future__ import annotations

import sys

try:
    import halide as hl
    from halide import tools
except ImportError as exc:
    sys.stderr.write(
        "Halide python bindings are missing. Activate the venv and run "
        "'pip install -r requirements.txt' before executing this script.\n"
    )
    raise


def build_gradient() -> tuple[hl.Func, hl.Var, hl.Var]:
    """Return a simple gradient pipeline that can be GPU scheduled."""
    x, y = hl.Var("x"), hl.Var("y")
    gradient = hl.Func("gradient")
    gradient[x, y] = hl.cast(hl.UInt(8), x + y)
    return gradient, x, y


def main() -> None:
    target = hl.get_target_from_environment()
    gradient, x, y = build_gradient()

    if target.has_gpu_feature():
        xo, yo = hl.Var("xo"), hl.Var("yo")
        xi, yi = hl.Var("xi"), hl.Var("yi")
        gradient.gpu_tile(x, y, xo, yo, xi, yi, 16, 16)
        print(f"Using GPU schedule with target: {target}")
    else:
        gradient.parallel(y).vectorize(x, 16)
        print(
            f"No GPU features detected in target {target}. "
            "Set HL_TARGET (e.g. host-cuda) before running to enable GPU scheduling."
        )

    output = gradient.realize(1024, 768, target=target)
    tools.save_image(output, "gradient_py.png")
    print("Wrote gradient_py.png")


if __name__ == "__main__":
    main()
