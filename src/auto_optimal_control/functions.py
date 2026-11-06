"""Min, max and the unit step, left unevaluated for large symbolic expressions.

SymPy's ``Min`` and ``Max`` re-simplify their arguments (``gcd_terms`` and
argument collapsing) each time an expression holding them is rebuilt, by
``subs``, ``xreplace`` or ``diff``. As a ``Piecewise``, the result is folded
and its conditions simplified. On the clipped boundary-arc laws of a two-link
arm either takes minutes. These functions do nothing on construction. They
differentiate into :class:`Step`, print as ``fmin``/``fmax`` in C and as
``numpy.minimum``/``numpy.maximum`` through ``lambdify``.
"""

import sympy as sp


class Step(sp.Function):
    """The unit step, 1/2 at zero. Its derivative is taken as zero: it only appears
    in the first derivatives of clipped laws."""

    nargs = 1

    def fdiff(self, argindex=1):
        return sp.S.Zero

    def _ccode(self, printer):
        a = printer._print(self.args[0])
        return f"((({a}) > 0) ? 1.0 : ((({a}) < 0) ? 0.0 : 0.5))"

    def _numpycode(self, printer):
        return f"numpy.heaviside({printer._print(self.args[0])}, 0.5)"


class Lesser(sp.Function):
    """``min(a, b)``, unevaluated."""

    nargs = 2

    def fdiff(self, argindex=1):
        a, b = self.args
        return Step(b - a) if argindex == 1 else Step(a - b)

    def _ccode(self, printer):
        return f"fmin({printer._print(self.args[0])}, {printer._print(self.args[1])})"

    def _numpycode(self, printer):
        return f"numpy.minimum({printer._print(self.args[0])}, {printer._print(self.args[1])})"


class Greater(sp.Function):
    """``max(a, b)``, unevaluated."""

    nargs = 2

    def fdiff(self, argindex=1):
        a, b = self.args
        return Step(a - b) if argindex == 1 else Step(b - a)

    def _ccode(self, printer):
        return f"fmax({printer._print(self.args[0])}, {printer._print(self.args[1])})"

    def _numpycode(self, printer):
        return f"numpy.maximum({printer._print(self.args[0])}, {printer._print(self.args[1])})"


def lesser(*args):
    """The smallest of ``args``, as nested :class:`Lesser`."""
    out = sp.sympify(args[0])
    for a in args[1:]:
        out = Lesser(out, a)
    return out


def greater(*args):
    """The largest of ``args``, as nested :class:`Greater`."""
    out = sp.sympify(args[0])
    for a in args[1:]:
        out = Greater(out, a)
    return out
