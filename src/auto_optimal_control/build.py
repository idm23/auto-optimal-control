"""Compile the generated C kernels into an importable module, with a cache.

:func:`build_kernels` prints a :class:`~auto_optimal_control.pontryagin.PontryaginSystem`
as C (:mod:`auto_optimal_control.codegen`) and adds a Cython wrapper that plugs
the kernels into fast-numerical-integration's GIL-free C API. It compiles both
and returns :class:`Kernels`.

Builds are cached by a hash of the generated code and of the toolchain
(Python, Cython, FNI's ``.pxd``). The first build of a problem takes a few
seconds; later calls, in this process or another, load the cached module.
The cache lives in ``$AUTO_OPTIMAL_CONTROL_CACHE``, or else
``$XDG_CACHE_HOME/auto-optimal-control`` (``~/.cache/auto-optimal-control``).

A C compiler is required, as for any Cython extension.
"""

import glob
import hashlib
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import Cython
import numpy as np

import fast_numerical_integration
from auto_optimal_control.codegen import c_identifier, generate_c
from fast_numerical_integration.integrators import solve_hybrid

_TEMPLATE = Path(__file__).with_name("wrapper_template.pyx")

_SETUP = '''\
import os

from Cython.Build import cythonize
from setuptools import Extension, setup

MOD = {mod!r}
FNI_INCLUDE = {fni_include!r}
# As in fast-numerical-integration: no FMA contraction, so results do not
# depend on the compiler's choice to fuse multiply-adds.
FLAGS = ["/O2", "/fp:precise"] if os.name == "nt" else ["-O3", "-ffp-contract=off"]

ext = Extension(MOD, [MOD + ".pyx", MOD + "_kernels.c"], extra_compile_args=FLAGS)
setup(
    name=MOD,
    ext_modules=cythonize([ext], include_path=[FNI_INCLUDE], quiet=True,
                          compiler_directives={{"language_level": 3}}),
    script_args=["build_ext", "--inplace"],
)
'''


class BuildError(RuntimeError):
    """The generated kernels failed to compile; the message holds the build log."""


def cache_root():
    env = os.environ.get("AUTO_OPTIMAL_CONTROL_CACHE")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "auto-optimal-control"


def _fni_include():
    return os.path.dirname(os.path.dirname(fast_numerical_integration.__file__))


def _render_wrapper(pmp, prefix, header_name, macros):
    params = tuple(s.name for s in pmp.params)
    values = {
        "name!r": repr(pmp.problem.name),
        "header": header_name,
        "P": prefix,
        "param_names": repr(params),
        "constraint_names": repr(tuple(c.name for c in pmp.path_constraints)),
        "constraint_orders": repr(tuple(c.order for c in pmp.path_constraints)),
        "exact_arcs": repr(tuple(c.boundary is not None for c in pmp.path_constraints)),
        "NC1": str(max(macros["NC"], 1)),
        "NM1": str(max(macros["NM"], 1)),
        "NK1": str(max(macros["NK"], 1)),
        "free_tf": repr(bool(macros["FREE_TF"])),
        "NP1": str(max(macros["NP"], 1)),
        "NSW1": str(max(macros["NSW"], 1)),
        "NNU1": str(max(macros["NNU"], 1)),
    }
    values.update({k: str(v) for k, v in macros.items()})
    text = _TEMPLATE.read_text()
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def _fingerprint(*texts):
    h = hashlib.sha256()
    pxd = Path(fast_numerical_integration.__file__).with_name("integrators.pxd")
    for part in (*texts, pxd.read_text(), Cython.__version__, sys.version, platform.machine(),
                 platform.system()):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


def _find_extension(directory, mod):
    hits = glob.glob(os.path.join(directory, mod + "*.so")) + glob.glob(
        os.path.join(directory, mod + "*.pyd"))
    return hits[0] if hits else None


def _compile(directory, mod, files, verbose):
    for fname, text in files.items():
        Path(directory, fname).write_text(text)
    proc = subprocess.run([sys.executable, "setup.py"], cwd=directory, capture_output=True,
                          text=True)
    if verbose:
        print(proc.stdout, proc.stderr, sep="\n")
    if proc.returncode != 0 or _find_extension(directory, mod) is None:
        log = (proc.stdout + "\n" + proc.stderr)[-8000:]
        raise BuildError(f"building {mod} failed in {directory}:\n{log}")


def _load(mod, path):
    if mod in sys.modules:
        return sys.modules[mod]
    spec = importlib.util.spec_from_file_location(mod, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[mod] = module
    return module


def build_kernels(pmp, name=None, *, cache_dir=None, force=False, verbose=False):
    """Generate, compile (or load from the cache) and return the kernels of ``pmp``."""
    ident = c_identifier(name or pmp.problem.name)
    prefix, header, source, macros = generate_c(pmp, ident)
    probe = "aoc_" + ident
    wrapper = _render_wrapper(pmp, prefix, probe + "_kernels.h", macros)
    key = _fingerprint(header, source, wrapper)
    mod = f"aoc_{ident}_{key[:16]}"
    header_name = mod + "_kernels.h"
    wrapper = wrapper.replace(probe + "_kernels.h", header_name)
    source = source.replace("{header}", header_name)

    root = Path(cache_dir) if cache_dir is not None else cache_root()
    root.mkdir(parents=True, exist_ok=True)
    final = root / mod
    path = None if force else _find_extension(final, mod)
    if path is None:
        if force and final.exists():
            shutil.rmtree(final)
        tmp = tempfile.mkdtemp(prefix=mod + ".", dir=root)
        try:
            _compile(tmp, mod, {
                mod + ".pyx": wrapper,
                header_name: header,
                mod + "_kernels.c": source,
                "setup.py": _SETUP.format(mod=mod, fni_include=_fni_include()),
            }, verbose)
            try:
                os.rename(tmp, final)
            except OSError:              # another process finished first
                shutil.rmtree(tmp, ignore_errors=True)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        path = _find_extension(final, mod)
    module = _load(mod, path)
    return Kernels(pmp, module, prefix, header, source, Path(path))


class Kernels:
    """Compiled kernels of one :class:`~auto_optimal_control.pontryagin.PontryaginSystem`.

    The generated module's functions are available as attributes (``rhs``,
    ``rhs_arc``, ``jacobian``, ``dfdp``, ``control``, ``hamiltonian``,
    ``cost_rate``, ``switching``, ``switching_gradient``, ``constraint``,
    ``modes``, ``constraint_rate``, ``initial_arc``, ``residual``,
    ``residual_jacobian``, ``stm_jump``, ``propagate``, ``propagate_batch``,
    and for exact boundary arcs ``propagate_boundary``, ``boundary_rhs``,
    ``boundary_jacobian``, ``boundary_control``, ``boundary_multiplier``,
    ``boundary_hamiltonian``, ``hamiltonian_gradient``, ``tangency``,
    ``tangency_gradient``) and so are its FNI types (``Dynamics``,
    ``ArcDynamics``, ``BoundaryDynamics``, ``Switching``, ``Constraint``,
    ``ConstraintRate``). All of them take ``z = [x, lambda]`` and
    the parameter vector in the orders of ``system.z`` and ``system.params``.
    """

    def __init__(self, system, module, prefix, header, source, path):
        self.system = system
        self.module = module
        self.prefix = prefix
        self.header = header
        self.source = source
        self.path = path

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError(attr)
        return getattr(self.module, attr)

    def __repr__(self):
        return f"<Kernels {self.module.__name__} at {self.path}>"

    @property
    def param_names(self):
        return self.module.PARAM_NAMES

    def write_c(self, directory):
        """Write the plain-C kernels (``<prefix>kernels.h`` / ``.c``) to ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem = self.prefix + "kernels"
        source = self.source.replace(f'#include "{self.module.__name__}_kernels.h"',
                                     f'#include "{stem}.h"')
        (directory / f"{stem}.h").write_text(self.header)
        (directory / f"{stem}.c").write_text(source)
        return directory / f"{stem}.h", directory / f"{stem}.c"

    def initial_state(self, params=()):
        """The fixed initial state, with NaN for free components."""
        problem = self.system.problem
        values = dict(zip(self.system.params, np.asarray(params, dtype=float).reshape(-1)))
        t0 = float(problem.initial_time.subs(values))
        out = np.full(len(self.system.states), np.nan)
        for i, s in enumerate(self.system.states):
            if s in problem.initial_state:
                out[i] = float(problem.initial_state[s].subs({**values, problem.time: t0}))
        return out

    def z0(self, lam0, params=(), x0=None):
        """``[x0, lam0]``; ``x0`` defaults to the problem's fixed initial state."""
        x = self.initial_state(params) if x0 is None else np.asarray(x0, dtype=float)
        lam = np.asarray(lam0, dtype=float)
        if np.isnan(x).any():
            raise ValueError("x0 has free components; pass x0 explicitly")
        if x.shape != lam.shape:
            raise ValueError(f"x0 and lam0 must both have {len(self.system.states)} entries")
        return np.concatenate([x, lam])

    def switching_events(self, params=None, terminal=False, direction=0.0):
        """One native FNI event per switching function."""
        return [self.module.Switching(k, params, terminal, direction)
                for k in range(self.module.N_SWITCHING)]

    def constraint_events(self, params=None, terminal=False, direction=0.0):
        """One native FNI event per path constraint (value ``g``)."""
        return [self.module.Constraint(c, params, terminal, direction)
                for c in range(self.module.N_CONSTRAINTS)]

    def trajectory(self, t0, z0, t1, params=None, *, t_eval=None, n_points=201, arc=None,
                   method="DOP853", rtol=1e-10, atol=1e-12, max_switches=10000,
                   terminate=False, n_interior=4, constraint_events=True):
        """The extremal from ``z0``, sampled: states, costates, controls, ``H`` and ``g``.

        Integrates like :func:`propagate`, as one FNI hybrid integration
        (``solve_hybrid`` on :class:`ArcDynamics`, with each mode function a
        guard whose reset map flips the arc). Then it samples the piecewise
        dense output at ``t_eval`` (default: ``n_points`` evenly spaced
        times). With ``terminate`` (as in :func:`propagate`) the integration
        stops at a violation and only the times reached are sampled.
        ``n_interior`` and ``constraint_events`` are as in :func:`propagate`.

        Returns a dict with ``t``, ``z`` (len, NZ), ``u`` (len, NU), ``H``
        (len,), ``g`` (len, NC), ``switch_times``, ``switch_index``,
        ``constraint_times``, ``constraint_index`` and ``status`` (-4 if it
        stopped at a violation).
        """
        mod = self.module
        nsw, nm = mod.N_SWITCHING, mod.N_MODES
        p = None if params is None else np.asarray(params, dtype=float)
        z = np.asarray(z0, dtype=float)
        arc = mod.initial_arc(t0, z, p) if arc is None else np.array(arc, dtype=np.intc)
        nc = mod.N_CONSTRAINTS if constraint_events else 0
        if not constraint_events:
            arc[nsw:] = 0
        stop = np.zeros(mod.N_CONSTRAINTS, dtype=bool)
        if nc:
            stop[:] = mod._stop_mask(terminate)[:mod.N_CONSTRAINTS]
        dynamics = mod.ArcDynamics(p, arc if nm else None)
        violated = []

        rates = [mod.ConstraintRate(c, p, arc) for c in range(nc)]
        kinks = [mod.Kink(i, p, arc, True, 0.0) for i in range(mod.N_KINKS)]
        unchanged = []                   # resets that left the arc as it was

        def flip(k, t, y):
            # As the compiled reset map: a flag follows the sign of its mode
            # function's rate at the crossing, and a tangential zero flips it.
            # A saturation kink only restarts the step.
            if k >= nsw + nc:
                return y
            a = dynamics.arc
            flag = 1 - a[k]
            if k < nsw:
                # As for constraints: the side S moves to (upper where it falls).
                St, Sz, _ = mod.switching_gradient(t, y[:mod.N_Z], p)
                rate = St[k] + Sz[k] @ mod.rhs_arc(t, y[:mod.N_Z], p, a)
                flag = 1 if rate < 0 else 0 if rate > 0 else flag
                if flag == a[k]:
                    unchanged.append((t, k))
                    return y
            if k >= nsw:
                rate = mod.constraint_rate(t, y[:mod.N_Z], p, a)[k - nsw]
                flag = 1 if rate > 0 else 0 if rate < 0 else flag
                if flag == 1 and stop[k - nsw]:
                    violated.append((t, k))
                    return None
            a[k] = flag
            dynamics.arc = a
            for r in rates + kinks:
                r.arc = a
            return y

        status = 0
        if any(stop[c] and arc[nsw + c] for c in range(mod.N_CONSTRAINTS)):
            t_end, sol = t0, None
            status = mod.STATUS_VIOLATED
        else:
            detection = ({"event_detection": "interior", "n_interior": int(n_interior)}
                         if nc and n_interior else {})
            events = ([mod.Switching(k, p, True, 0.0) for k in range(nsw)]
                      + [mod.Constraint(c, p, True, 0.0) for c in range(nc)]
                      + kinks + rates)
            # Constraint events report g, the mode function is -g: same zeros.
            # The rate events (peaks and troughs of g) only split steps.
            # Kinks come before them so the guards' indices stay contiguous.
            sol = solve_hybrid(dynamics, (t0, t1), z, events or None, flip if events else None,
                               method=method, dense_output=True, rtol=rtol, atol=atol,
                               max_resets=max_switches, **detection)
            t_end = sol.t[-1]
            status = mod.STATUS_VIOLATED if violated else sol.status
        if t_eval is None:
            t_eval = np.linspace(t0, t_end, n_points)
        t_eval = np.asarray(t_eval, dtype=float)
        if sol is None:
            t_eval = np.array([t0])
            Z = z[None, :].copy()
        else:
            lo, hi = sorted((t0, t_end))
            t_eval = t_eval[(t_eval >= lo) & (t_eval <= hi)]
            Z = sol.sol(t_eval).T.reshape(len(t_eval), mod.N_Z)
        U = np.array([mod.control(ti, zi, p) for ti, zi in zip(t_eval, Z)]).reshape(
            len(t_eval), mod.N_CONTROLS)
        H = np.array([mod.hamiltonian(ti, zi, p) for ti, zi in zip(t_eval, Z)])
        G = np.array([mod.constraint(ti, zi, p) for ti, zi in zip(t_eval, Z)]).reshape(
            len(t_eval), mod.N_CONSTRAINTS)
        times = np.zeros(0) if sol is None else np.asarray(sol.t_resets, dtype=float)
        index = (np.zeros(0, dtype=np.intp) if sol is None
                 else np.asarray(sol.reset_events, dtype=np.intp))
        keep = (index < nsw + nc) & ~np.array(   # drop the kinks and the no-op resets
            [(ti, ki) in unchanged for ti, ki in zip(times, index)], dtype=bool).reshape(-1)
        times, index = times[keep], index[keep]
        if violated:                     # a stopping reset is not recorded by FNI
            times = np.append(times, violated[0][0])
            index = np.append(index, violated[0][1])
        sw = index < nsw
        c_times, c_index = times[~sw], index[~sw] - nsw
        return {"t": t_eval, "z": Z, "u": U, "H": H, "g": G, "switch_times": times[sw],
                "switch_index": index[sw], "constraint_times": c_times,
                "constraint_index": c_index, "status": status}
