"""Reference values for simpson / trapz from scipy.integrate (scipy >= 1.11: Cartwright correction for an
even number of samples). Writes tests/fixtures/quadrature-scipy.json.

    py -3 packages/advanced/tests/refs/quadrature_ref.py
"""
import json
import pathlib

import numpy as np
import scipy
from scipy.integrate import simpson, trapezoid

rng = np.random.default_rng(20260924)
cases = []


def add(name, y, x=None, dx=None):
    y = np.asarray(y, dtype=float)
    kw = {}
    if x is not None:
        kw["x"] = np.asarray(x, dtype=float)
    elif dx is not None:
        kw["dx"] = dx
    cases.append({
        "name": name,
        "y": y.tolist(),
        "x": None if x is None else np.asarray(x, dtype=float).tolist(),
        "dx": dx,
        "simpson": float(simpson(y, **kw)),
        "trapz": float(trapezoid(y, **kw)),
    })


# small fixed cases, both parities, uniform and not
add("two points", [1.0, 2.0])
add("three points", [0.0, 1.0, 0.0])
add("four points (even N)", [0.0, 1.0, 4.0, 9.0])
add("four points uneven x", [0.0, 1.0, 4.0, 9.0], x=[0.0, 0.5, 1.7, 3.0])
add("three points uneven", [1.0, 2.0, 3.0], x=[0.0, 0.5, 2.0])
add("dx scaling", [1.0, 4.0, 9.0, 16.0, 25.0], dx=0.25)
add("decreasing x", [1.0, 2.0, 5.0, 3.0, 0.5], x=[4.0, 3.0, 2.5, 1.0, 0.0])
# random functions on random grids
for n in [5, 6, 7, 10, 11, 50, 51]:
    for kind in ["uniform", "uneven"]:
        if kind == "uniform":
            x = np.linspace(-1.3, 2.9, n)
        else:
            x = np.sort(rng.uniform(-2.0, 3.0, n))
            x[0], x[-1] = -2.0, 3.0
        y = np.sin(3 * x) + 0.3 * x ** 2 - np.exp(-x)
        add(f"smooth n={n} {kind}", y, x=x)
        add(f"noise n={n} {kind}", rng.normal(size=n), x=x)

out = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "quadrature-scipy.json"
out.write_text(json.dumps({"scipy": scipy.__version__, "cases": cases}, indent=1))
print(f"{len(cases)} cases -> {out}")
